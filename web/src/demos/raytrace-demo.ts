/**
 * Ray Tracing Demo — renders 100 Cornell Boxes at random positions using
 * TLAS/BLAS two-level CWBVH traversal on the GPU via WebGPU compute shader.
 * Data layout aligns with tray_racing (rt_gpu_software_query_tlas.hlsl).
 *
 * Each Cornell Box has:
 *   - Triangle mesh BLAS: walls + boxes (standard BVH traversal)
 *   - Procedural sphere BLAS: sphere SDF intersection (ray marching)
 *
 * The sphere_data buffer is an application-layer per-instance buffer that stores
 * [center.x, center.y, center.z, radius] for each instance. Procedural spheres
 * read their radius from this buffer via instance_id, validating that instance_id
 * propagation through TLAS traversal works correctly.
 */
import { BufferManager } from '../gpu/buffer-manager';
import { ComputePipeline } from '../gpu/compute-pipeline';
import { RenderPipeline } from '../gpu/render-pipeline';
import { BvhManager, BuildQuality } from '../bvh/bvh-manager';
import { OrbitCamera } from '../utils/camera';
import { generateCornellBoxAt, CornellBoxResult } from '../utils/geometry';

// Shader sources (loaded via ?raw in Vite)
import cwbvhCommonSrc from '../shaders/cwbvh_common.wgsl?raw';
import cwbvhTraverseSrc from '../shaders/cwbvh_traverse.wgsl?raw';
import cwbvhTlasTraverseSrc from '../shaders/cwbvh_tlas_traverse.wgsl?raw';
import rayTraceSrc from '../shaders/ray_trace.wgsl?raw';
import fullscreenSrc from '../shaders/fullscreen.wgsl?raw';

export class RaytraceDemo {
  private device: GPUDevice;
  private context: GPUCanvasContext;
  private format: GPUTextureFormat;
  private canvas: HTMLCanvasElement;

  private bufferManager: BufferManager;
  private bvhManager: BvhManager;
  private computePipeline: ComputePipeline;
  private renderPipeline: RenderPipeline;
  private camera: OrbitCamera;

  private outputTexture: GPUTexture | null = null;
  private computeBindGroups: GPUBindGroup[] = [];
  private frame = 0;
  private animId = 0;
  private resolutionScale = 1.0;
  private tlasStart = 0;

  // Stats callback
  public onStats?: (stats: {
    buildTimeMs: number;
    gpuTimeMs: number;
    fps: number;
    triangles: number;
    nodes: number;
  }) => void;

  private lastFrameTime = 0;
  private fpsAccum = 0;
  private fpsCount = 0;
  private currentFps = 0;

  constructor(
    device: GPUDevice,
    context: GPUCanvasContext,
    format: GPUTextureFormat,
    canvas: HTMLCanvasElement,
    bvhManager: BvhManager
  ) {
    this.device = device;
    this.context = context;
    this.format = format;
    this.canvas = canvas;
    this.bufferManager = new BufferManager(device);
    this.bvhManager = bvhManager;
    this.computePipeline = new ComputePipeline(device);
    this.renderPipeline = new RenderPipeline(device);
    this.camera = new OrbitCamera(canvas);
  }

  async init(): Promise<void> {
    // ── Scene Generation ──
    // 100 Cornell Boxes, each producing:
    //   - 1 triangle mesh BLAS (walls + boxes)
    //   - 1 procedural sphere BLAS (sphere SDF)
    // Total TLAS instances: 200 (100 mesh + 100 procedural), interleaved.
    const INSTANCE_COUNT = 100;
    const SPREAD = 30;

    // Seeded random for reproducibility
    let seed = 42;
    const seededRandom = () => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return seed / 0x7fffffff;
    };

    // Generate Cornell Box data
    const cornellBoxes: CornellBoxResult[] = [];
    const positions: [number, number, number][] = [];
    for (let i = 0; i < INSTANCE_COUNT; i++) {
      const x = (seededRandom() - 0.5) * SPREAD;
      const y = (seededRandom() - 0.5) * SPREAD;
      const z = (seededRandom() - 0.5) * SPREAD;
      positions.push([x, y, z]);
      cornellBoxes.push(generateCornellBoxAt([x, y, z]));
    }

    // ── Instance Layout ──
    // Interleave: [mesh_0, proc_0, mesh_1, proc_1, ..., mesh_99, proc_99]
    // mesh instance indices: 0, 2, 4, ..., 198
    // procedural instance indices: 1, 3, 5, ..., 199
    const totalInstances = INSTANCE_COUNT * 2;
    const blasTriArrays: Float32Array[] = [];
    const proceduralAabbsList: number[] = [];
    const proceduralInstanceIndicesList: number[] = [];

    // ── Per-instance sphere_data buffer (APPLICATION-LAYER data) ──
    // sphere_data[instance_id * 4 + 0..3] = [center.x, center.y, center.z, radius]
    // For triangle mesh instances, the values don't matter (won't be read).
    // For procedural instances, this is where the sphere parameters live.
    // Each procedural sphere gets a RANDOM radius to test instance_id correctness.
    const sphereDataArray = new Float32Array(totalInstances * 4);

    for (let i = 0; i < INSTANCE_COUNT; i++) {
      const cb = cornellBoxes[i];

      // Triangle mesh BLAS (walls + boxes)
      const meshInstanceIdx = i * 2;      // 0, 2, 4, ...
      blasTriArrays.push(cb.triangles);

      // For mesh instances, write zeros to sphere_data (unused but must exist)
      sphereDataArray[meshInstanceIdx * 4 + 0] = 0;
      sphereDataArray[meshInstanceIdx * 4 + 1] = 0;
      sphereDataArray[meshInstanceIdx * 4 + 2] = 0;
      sphereDataArray[meshInstanceIdx * 4 + 3] = 0;

      // Procedural sphere BLAS
      const procInstanceIdx = i * 2 + 1;  // 1, 3, 5, ...
      proceduralInstanceIndicesList.push(procInstanceIdx);

      // Random radius between 0.12 and 0.35 for visual variety
      const randomRadius = 0.12 + seededRandom() * 0.23;

      // Sphere center from Cornell Box generation
      const [scx, scy, scz] = cb.sphereCenter;

      // Build the AABB for this procedural sphere (using the random radius!)
      proceduralAabbsList.push(
        scx - randomRadius, scy - randomRadius, scz - randomRadius,  // min
        scx + randomRadius, scy + randomRadius, scz + randomRadius   // max
      );

      // Write sphere data: center + randomized radius
      // The shader will read this via instance_id to get the correct sphere parameters.
      sphereDataArray[procInstanceIdx * 4 + 0] = scx;
      sphereDataArray[procInstanceIdx * 4 + 1] = scy;
      sphereDataArray[procInstanceIdx * 4 + 2] = scz;
      sphereDataArray[procInstanceIdx * 4 + 3] = randomRadius;
    }

    const proceduralAabbs = new Float32Array(proceduralAabbsList);
    const proceduralInstanceIndices = new Uint32Array(proceduralInstanceIndicesList);

    // ── Build TLAS scene (tray_racing layout with mixed BLAS types) ──
    this.bvhManager.buildTlasScene(
      blasTriArrays,
      proceduralAabbs,
      proceduralInstanceIndices,
      BuildQuality.Medium
    );
    this.tlasStart = this.bvhManager.tlasStart;

    // ── Upload TLAS scene to GPU ──
    this.bvhManager.uploadTlasToGpu(this.bufferManager);

    // ── Upload sphere_data buffer (application-layer per-instance data) ──
    // This buffer is NOT part of the BVH library — it's pure app data.
    // The shader uses instance_id (from TLAS traversal) to index into it.
    this.bufferManager.uploadBuffer(
      'sphere_data',
      new Uint8Array(sphereDataArray.buffer)
    );

    // ── Combine shader sources ──
    const traceImpl = `
fn trace_ray(ray: Ray) -> RayHit {
    return traverse_tlas_closest(ray, camera.tlas_start);
}
fn trace_shadow_ray(ray: Ray) -> bool {
    return traverse_tlas_any(ray, camera.tlas_start);
}
`;
    const computeShader = (cwbvhCommonSrc + '\n' + cwbvhTraverseSrc + '\n' + cwbvhTlasTraverseSrc + '\n' + rayTraceSrc)
      .replace('/*TRACE_IMPL*/', traceImpl);

    // Initialize pipelines
    await this.computePipeline.init(computeShader, 'main');
    await this.renderPipeline.init(fullscreenSrc, this.format);

    // Setup camera — pull back to see the whole scene
    this.camera.target = [0, 0, 0];
    this.camera.distance = 25;
    this.camera.phi = Math.PI * 0.4;

    // Create output texture and bind groups
    this.recreateOutputTexture();
  }

  setResolutionScale(scale: number): void {
    this.resolutionScale = scale;
    this.recreateOutputTexture();
    this.camera.markDirty();
  }

  private recreateOutputTexture(): void {
    this.outputTexture?.destroy();

    const w = Math.max(1, Math.floor(this.canvas.width * this.resolutionScale));
    const h = Math.max(1, Math.floor(this.canvas.height * this.resolutionScale));

    this.outputTexture = this.device.createTexture({
      label: 'ray-trace-output',
      size: [w, h],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });

    this.renderPipeline.updateTexture(this.outputTexture);
    this.rebuildBindGroups(w, h);
  }

  private rebuildBindGroups(width: number, height: number): void {
    const nodesBuffer = this.bufferManager.getBuffer('bvh_nodes')!;
    const indicesBuffer = this.bufferManager.getBuffer('primitive_indices')!;
    const trianglesBuffer = this.bufferManager.getBuffer('triangles')!;
    const blasOffsetsBuffer = this.bufferManager.getBuffer('blas_offsets')!;
    const sphereDataBuffer = this.bufferManager.getBuffer('sphere_data')!;

    // BVH bind group (group 0) — aligns with tray_racing bindings:
    // binding(0) = bvh_nodes, binding(1) = primitive_indices,
    // binding(2) = triangles, binding(3) = blas_offsets (INSTANCES_BINDING),
    // binding(4) = sphere_data (app-layer per-instance data)
    // Note: blas_types and blas_instance_ids are eliminated —
    // procedural detection uses blas_offsets == INVALID_U32 sentinel,
    // and instance_id is read from primitive_indices directly.
    const bvhBindGroup = this.computePipeline.createBindGroup([
      { binding: 0, resource: { buffer: nodesBuffer } },
      { binding: 1, resource: { buffer: indicesBuffer } },
      { binding: 2, resource: { buffer: trianglesBuffer } },
      { binding: 3, resource: { buffer: blasOffsetsBuffer } },
      { binding: 4, resource: { buffer: sphereDataBuffer } },
    ], 0);

    // Camera + output texture bind group (group 1)
    const cameraBuffer = this.bufferManager.createUniformBuffer(
      'camera_uniforms',
      this.camera.toUniformData(width, height, this.frame, this.tlasStart)
    );

    const cameraBindGroup = this.computePipeline.createBindGroup([
      { binding: 0, resource: { buffer: cameraBuffer } },
      { binding: 1, resource: this.outputTexture!.createView() },
    ], 1);

    this.computeBindGroups = [bvhBindGroup, cameraBindGroup];
  }

  start(): void {
    this.lastFrameTime = performance.now();
    const loop = () => {
      this.animId = requestAnimationFrame(loop);
      this.renderFrame();
    };
    loop();
  }

  stop(): void {
    cancelAnimationFrame(this.animId);
  }

  private renderFrame(): void {
    const now = performance.now();
    const dt = now - this.lastFrameTime;
    this.lastFrameTime = now;

    // FPS tracking
    this.fpsAccum += dt;
    this.fpsCount++;
    if (this.fpsAccum >= 1000) {
      this.currentFps = Math.round((this.fpsCount / this.fpsAccum) * 1000);
      this.fpsAccum = 0;
      this.fpsCount = 0;
    }

    // Handle canvas resize
    const cw = this.canvas.clientWidth * devicePixelRatio;
    const ch = this.canvas.clientHeight * devicePixelRatio;
    if (this.canvas.width !== cw || this.canvas.height !== ch) {
      this.canvas.width = cw;
      this.canvas.height = ch;
      this.recreateOutputTexture();
    }

    this.frame++;

    const w = Math.max(1, Math.floor(this.canvas.width * this.resolutionScale));
    const h = Math.max(1, Math.floor(this.canvas.height * this.resolutionScale));

    // Update camera uniform
    const cameraBuffer = this.bufferManager.getBuffer('camera_uniforms')!;
    this.device.queue.writeBuffer(
      cameraBuffer,
      0,
      new Uint8Array(this.camera.toUniformData(w, h, this.frame, this.tlasStart))
    );

    // Rebuild bind group 1 with updated texture view (needed if texture was recreated)
    // We re-use existing bind groups if texture hasn't changed, else they were rebuilt in recreateOutputTexture

    const encoder = this.device.createCommandEncoder({ label: 'raytrace-frame' });

    // Dispatch compute
    const wgX = Math.ceil(w / 8);
    const wgY = Math.ceil(h / 8);
    this.computePipeline.dispatch(encoder, this.computeBindGroups, wgX, wgY);

    // Blit to canvas
    const textureView = this.context.getCurrentTexture().createView();
    this.renderPipeline.render(encoder, textureView);

    this.device.queue.submit([encoder.finish()]);

    // Report stats
    this.onStats?.({
      buildTimeMs: this.bvhManager.lastBuildTimeMs,
      gpuTimeMs: dt, // approximate
      fps: this.currentFps,
      triangles: this.bvhManager.triangleCount,
      nodes: this.bvhManager.nodeCount,
    });
  }

  destroy(): void {
    this.stop();
    this.outputTexture?.destroy();
    this.bufferManager.destroy();
  }
}
