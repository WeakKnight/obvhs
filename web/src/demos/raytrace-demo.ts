/**
 * Ray Tracing Demo — renders a Cornell Box or icosphere scene using
 * CWBVH traversal on the GPU via WebGPU compute shader.
 */
import { BufferManager } from '../gpu/buffer-manager';
import { ComputePipeline } from '../gpu/compute-pipeline';
import { RenderPipeline } from '../gpu/render-pipeline';
import { BvhManager, BuildQuality } from '../bvh/bvh-manager';
import { OrbitCamera } from '../utils/camera';
import { generateCornellBox, generateIcosphere, mergeTriangleArrays } from '../utils/geometry';

// Shader sources (loaded via ?raw in Vite)
import cwbvhCommonSrc from '../shaders/cwbvh_common.wgsl?raw';
import cwbvhTraverseSrc from '../shaders/cwbvh_traverse.wgsl?raw';
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
    // Generate scene geometry
    const cornell = generateCornellBox();
    const sphere = generateIcosphere([0, -0.5, 0], 0.35, 4);
    const triangles = mergeTriangleArrays(cornell, sphere);

    // Build BVH
    this.bvhManager.buildCwBvh(triangles, BuildQuality.Medium, false);

    // Upload BVH to GPU
    this.bvhManager.uploadToGpu(this.bufferManager);

    // Combine shader sources
    const computeShader = cwbvhCommonSrc + '\n' + cwbvhTraverseSrc + '\n' + rayTraceSrc;

    // Initialize pipelines
    await this.computePipeline.init(computeShader, 'main');
    await this.renderPipeline.init(fullscreenSrc, this.format);

    // Setup camera
    this.camera.target = [0, 0, 0];
    this.camera.distance = 3.5;
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

    // BVH bind group (group 0)
    const bvhBindGroup = this.computePipeline.createBindGroup([
      { binding: 0, resource: { buffer: nodesBuffer } },
      { binding: 1, resource: { buffer: indicesBuffer } },
      { binding: 2, resource: { buffer: trianglesBuffer } },
    ], 0);

    // Camera + output texture bind group (group 1)
    const cameraBuffer = this.bufferManager.createUniformBuffer(
      'camera_uniforms',
      this.camera.toUniformData(width, height, this.frame)
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
      new Uint8Array(this.camera.toUniformData(w, h, this.frame))
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
