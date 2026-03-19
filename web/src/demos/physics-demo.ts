/**
 * Physics Demo — real-time sphere collision simulation with dynamic BVH updates.
 * Simulates N bouncing spheres, uses BVH broadphase for collision detection,
 * and renders the scene with ray tracing.
 */
import { BufferManager } from '../gpu/buffer-manager';
import { ComputePipeline } from '../gpu/compute-pipeline';
import { RenderPipeline } from '../gpu/render-pipeline';
import { BvhManager, BuildQuality } from '../bvh/bvh-manager';
import { OrbitCamera } from '../utils/camera';
import { generateSphere } from '../utils/geometry';

import cwbvhCommonSrc from '../shaders/cwbvh_common.wgsl?raw';
import cwbvhTraverseSrc from '../shaders/cwbvh_traverse.wgsl?raw';
import rayTraceSrc from '../shaders/ray_trace.wgsl?raw';
import fullscreenSrc from '../shaders/fullscreen.wgsl?raw';

interface Sphere {
  position: [number, number, number];
  velocity: [number, number, number];
  radius: number;
  mass: number;
  triOffset: number; // offset into triangle array (in triangle count)
  triCount: number;  // number of triangles for this sphere
}

// Room bounds
const ROOM_MIN: [number, number, number] = [-3, -2, -3];
const ROOM_MAX: [number, number, number] = [3, 4, 3];
const GRAVITY = -9.8;
const RESTITUTION = 0.7;
const DAMPING = 0.998;

export class PhysicsDemo {
  private device: GPUDevice;
  private context: GPUCanvasContext;
  private format: GPUTextureFormat;
  private canvas: HTMLCanvasElement;

  private bufferManager: BufferManager;
  private bvhManager: BvhManager;
  private computePipeline: ComputePipeline;
  private renderPipeline: RenderPipeline;
  private camera: OrbitCamera;

  private spheres: Sphere[] = [];
  private sphereTemplate: Float32Array; // unit sphere triangles
  private allTriangles: Float32Array = new Float32Array(0);

  private outputTexture: GPUTexture | null = null;
  private computeBindGroups: GPUBindGroup[] = [];
  private frame = 0;
  private animId = 0;
  private lastTime = 0;

  private updateStrategy: 'rebuild' | 'reinsert' | 'partial_rebuild' = 'rebuild';

  public onStats?: (stats: {
    buildTimeMs: number;
    gpuTimeMs: number;
    fps: number;
    triangles: number;
    nodes: number;
  }) => void;

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

    // Pre-generate unit sphere mesh (low-poly for performance)
    this.sphereTemplate = generateSphere([0, 0, 0], 1, 8, 6);
  }

  async init(sphereCount: number = 64): Promise<void> {
    // Initialize spheres with random positions/velocities
    this.spheres = [];
    for (let i = 0; i < sphereCount; i++) {
      this.addSphere(
        [
          (Math.random() - 0.5) * 4,
          Math.random() * 3,
          (Math.random() - 0.5) * 4,
        ],
        [
          (Math.random() - 0.5) * 2,
          Math.random() * 2,
          (Math.random() - 0.5) * 2,
        ],
        0.15 + Math.random() * 0.15,
        1.0
      );
    }

    // Build initial triangle data
    this.rebuildTriangleData();

    // Build BVH (keep BVH2 for dynamic updates)
    this.bvhManager.buildCwBvh(this.allTriangles, BuildQuality.Fast, true);
    this.bvhManager.uploadToGpu(this.bufferManager);

    // Initialize pipelines
    const computeShader = cwbvhCommonSrc + '\n' + cwbvhTraverseSrc + '\n' + rayTraceSrc;
    await this.computePipeline.init(computeShader, 'main');
    await this.renderPipeline.init(fullscreenSrc, this.format);

    // Camera
    this.camera.target = [0, 1, 0];
    this.camera.distance = 8;
    this.camera.phi = Math.PI * 0.35;

    this.recreateOutputTexture();
  }

  private addSphere(
    position: [number, number, number],
    velocity: [number, number, number],
    radius: number,
    mass: number
  ): void {
    this.spheres.push({
      position,
      velocity,
      radius,
      mass,
      triOffset: 0,
      triCount: this.sphereTemplate.length / 9,
    });
  }

  addNewSphere(): void {
    this.addSphere(
      [
        (Math.random() - 0.5) * 2,
        3 + Math.random() * 2,
        (Math.random() - 0.5) * 2,
      ],
      [(Math.random() - 0.5) * 3, 0, (Math.random() - 0.5) * 3],
      0.15 + Math.random() * 0.15,
      1.0
    );
    this.rebuildTriangleData();
  }

  /** Rebuild the complete triangle data from all spheres at their current positions. */
  private rebuildTriangleData(): void {
    const triPerSphere = this.sphereTemplate.length / 9;
    const totalTris = this.spheres.length * triPerSphere;
    this.allTriangles = new Float32Array(totalTris * 9);

    for (let si = 0; si < this.spheres.length; si++) {
      const s = this.spheres[si];
      s.triOffset = si * triPerSphere;
      s.triCount = triPerSphere;

      const srcLen = this.sphereTemplate.length;
      const dstOffset = si * srcLen;

      for (let i = 0; i < srcLen; i += 3) {
        this.allTriangles[dstOffset + i + 0] = this.sphereTemplate[i + 0] * s.radius + s.position[0];
        this.allTriangles[dstOffset + i + 1] = this.sphereTemplate[i + 1] * s.radius + s.position[1];
        this.allTriangles[dstOffset + i + 2] = this.sphereTemplate[i + 2] * s.radius + s.position[2];
      }
    }
  }

  setUpdateStrategy(strategy: 'rebuild' | 'reinsert' | 'partial_rebuild'): void {
    this.updateStrategy = strategy;
  }

  private stepPhysics(dt: number): void {
    const dtClamped = Math.min(dt, 0.02); // Cap at 50fps equivalent

    for (const s of this.spheres) {
      // Gravity
      s.velocity[1] += GRAVITY * dtClamped;

      // Integrate position
      s.position[0] += s.velocity[0] * dtClamped;
      s.position[1] += s.velocity[1] * dtClamped;
      s.position[2] += s.velocity[2] * dtClamped;

      // Damping
      s.velocity[0] *= DAMPING;
      s.velocity[1] *= DAMPING;
      s.velocity[2] *= DAMPING;

      // Wall collision (room bounds)
      for (let axis = 0; axis < 3; axis++) {
        if (s.position[axis] - s.radius < ROOM_MIN[axis]) {
          s.position[axis] = ROOM_MIN[axis] + s.radius;
          s.velocity[axis] = Math.abs(s.velocity[axis]) * RESTITUTION;
        }
        if (s.position[axis] + s.radius > ROOM_MAX[axis]) {
          s.position[axis] = ROOM_MAX[axis] - s.radius;
          s.velocity[axis] = -Math.abs(s.velocity[axis]) * RESTITUTION;
        }
      }
    }

    // Sphere-sphere collision (CPU broadphase + narrowphase)
    for (let i = 0; i < this.spheres.length; i++) {
      for (let j = i + 1; j < this.spheres.length; j++) {
        const a = this.spheres[i];
        const b = this.spheres[j];

        const dx = b.position[0] - a.position[0];
        const dy = b.position[1] - a.position[1];
        const dz = b.position[2] - a.position[2];
        const dist2 = dx * dx + dy * dy + dz * dz;
        const minDist = a.radius + b.radius;

        if (dist2 < minDist * minDist && dist2 > 0.0001) {
          const dist = Math.sqrt(dist2);
          const nx = dx / dist;
          const ny = dy / dist;
          const nz = dz / dist;

          // Separate
          const overlap = (minDist - dist) * 0.5;
          a.position[0] -= nx * overlap;
          a.position[1] -= ny * overlap;
          a.position[2] -= nz * overlap;
          b.position[0] += nx * overlap;
          b.position[1] += ny * overlap;
          b.position[2] += nz * overlap;

          // Elastic collision response
          const dvx = a.velocity[0] - b.velocity[0];
          const dvy = a.velocity[1] - b.velocity[1];
          const dvz = a.velocity[2] - b.velocity[2];
          const dvn = dvx * nx + dvy * ny + dvz * nz;

          if (dvn > 0) continue; // Already separating

          const totalMass = a.mass + b.mass;
          const impulse = (-(1 + RESTITUTION) * dvn) / totalMass;

          a.velocity[0] += impulse * b.mass * nx;
          a.velocity[1] += impulse * b.mass * ny;
          a.velocity[2] += impulse * b.mass * nz;
          b.velocity[0] -= impulse * a.mass * nx;
          b.velocity[1] -= impulse * a.mass * ny;
          b.velocity[2] -= impulse * a.mass * nz;
        }
      }
    }
  }

  private updateBvh(): void {
    // Rebuild triangle data with new positions
    this.rebuildTriangleData();

    // Full rebuild each frame (simplest strategy)
    this.bvhManager.buildCwBvh(this.allTriangles, BuildQuality.Fastest, true);
    this.bvhManager.uploadToGpu(this.bufferManager);

    // Rebuild bind groups since buffers may have changed
    this.rebuildBindGroups();
  }

  private recreateOutputTexture(): void {
    this.outputTexture?.destroy();

    const w = Math.max(1, this.canvas.width);
    const h = Math.max(1, this.canvas.height);

    this.outputTexture = this.device.createTexture({
      label: 'physics-ray-output',
      size: [w, h],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });

    this.renderPipeline.updateTexture(this.outputTexture);
    this.rebuildBindGroups();
  }

  private rebuildBindGroups(): void {
    const nodesBuffer = this.bufferManager.getBuffer('bvh_nodes');
    const indicesBuffer = this.bufferManager.getBuffer('primitive_indices');
    const trianglesBuffer = this.bufferManager.getBuffer('triangles');

    if (!nodesBuffer || !indicesBuffer || !trianglesBuffer || !this.outputTexture) return;

    const w = this.outputTexture.width;
    const h = this.outputTexture.height;

    const bvhBindGroup = this.computePipeline.createBindGroup([
      { binding: 0, resource: { buffer: nodesBuffer } },
      { binding: 1, resource: { buffer: indicesBuffer } },
      { binding: 2, resource: { buffer: trianglesBuffer } },
    ], 0);

    const cameraBuffer = this.bufferManager.createUniformBuffer(
      'camera_uniforms',
      this.camera.toUniformData(w, h, this.frame)
    );

    const cameraBindGroup = this.computePipeline.createBindGroup([
      { binding: 0, resource: { buffer: cameraBuffer } },
      { binding: 1, resource: this.outputTexture.createView() },
    ], 1);

    this.computeBindGroups = [bvhBindGroup, cameraBindGroup];
  }

  start(): void {
    this.lastTime = performance.now();
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
    const dt = (now - this.lastTime) / 1000;
    this.lastTime = now;

    // FPS tracking
    this.fpsAccum += dt * 1000;
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

    // Physics step
    this.stepPhysics(dt);

    // Update BVH
    const buildStart = performance.now();
    this.updateBvh();
    const buildTime = performance.now() - buildStart;

    this.frame++;

    const w = this.outputTexture?.width ?? 1;
    const h = this.outputTexture?.height ?? 1;

    // Update camera uniform
    const cameraBuffer = this.bufferManager.getBuffer('camera_uniforms');
    if (cameraBuffer) {
      this.device.queue.writeBuffer(
        cameraBuffer,
        0,
        new Uint8Array(this.camera.toUniformData(w, h, this.frame))
      );
    }

    if (this.computeBindGroups.length < 2 || !this.outputTexture) return;

    const encoder = this.device.createCommandEncoder({ label: 'physics-frame' });

    // Ray trace
    const wgX = Math.ceil(w / 8);
    const wgY = Math.ceil(h / 8);
    this.computePipeline.dispatch(encoder, this.computeBindGroups, wgX, wgY);

    // Blit
    const textureView = this.context.getCurrentTexture().createView();
    this.renderPipeline.render(encoder, textureView);

    this.device.queue.submit([encoder.finish()]);

    this.onStats?.({
      buildTimeMs: buildTime,
      gpuTimeMs: dt * 1000,
      fps: this.currentFps,
      triangles: this.allTriangles.length / 9,
      nodes: this.bvhManager.nodeCount,
    });
  }

  destroy(): void {
    this.stop();
    this.outputTexture?.destroy();
    this.bufferManager.destroy();
  }
}
