/**
 * obvhs — Compressed Wide BVH (CWBVH) for WebGPU
 *
 * High-performance BVH construction (via WASM) and GPU traversal shaders
 * for ray tracing, collision detection, and spatial queries on the web.
 *
 * @packageDocumentation
 */

// ── Core BVH ──
export {
  BvhManager,
  BuildQuality,
  type BvhManagerInitOptions,
  type WasmCwBvhInstance,
  type WasmTlasSceneInstance,
  type WasmModule,
} from './bvh/bvh-manager';

// ── GPU Buffer Management ──
export { BufferManager } from './gpu/buffer-manager';

// ── WGSL Shader Sources ──
export {
  cwbvhCommonSrc,
  cwbvhTraverseSrc,
  cwbvhTlasTraverseSrc,
  rayTraceSrc,
  fullscreenSrc,
  collisionSrc,
  singleBvhShaderSrc,
  tlasBvhShaderSrc,
  collisionShaderSrc,
} from './shaders/index';

// ── Geometry Utilities ──
export {
  generateSphere,
  generateIcosphere,
  generateCornellBox,
  generateCornellBoxAt,
  mergeTriangleArrays,
  type CornellBoxResult,
} from './utils/geometry';
