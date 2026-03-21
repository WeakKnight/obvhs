/**
 * WGSL Shader Sources — exported as string constants.
 * These are the BVH traversal and utility shaders for use in WebGPU compute/render pipelines.
 */

// NOTE: In library build mode, Vite inlines these via ?raw imports.
// When consumed as an npm package, these are pre-bundled string constants.

import cwbvhCommon from './cwbvh_common.wgsl?raw';
import cwbvhTraverse from './cwbvh_traverse.wgsl?raw';
import cwbvhTlasTraverse from './cwbvh_tlas_traverse.wgsl?raw';
import rayTrace from './ray_trace.wgsl?raw';
import fullscreen from './fullscreen.wgsl?raw';
import collision from './collision.wgsl?raw';

export const cwbvhCommonSrc = cwbvhCommon;
export const cwbvhTraverseSrc = cwbvhTraverse;
export const cwbvhTlasTraverseSrc = cwbvhTlasTraverse;
export const rayTraceSrc = rayTrace;
export const fullscreenSrc = fullscreen;
export const collisionSrc = collision;

/**
 * Pre-combined shader source for single-layer BVH ray tracing.
 * Includes: cwbvh_common + cwbvh_traverse + ray_trace
 * You need to replace `/*TRACE_IMPL*​/` with your trace function implementations.
 */
export const singleBvhShaderSrc = cwbvhCommon + '\n' + cwbvhTraverse + '\n' + rayTrace;

/**
 * Pre-combined shader source for TLAS/BLAS two-level ray tracing.
 * Includes: cwbvh_common + cwbvh_traverse + cwbvh_tlas_traverse + ray_trace
 * You need to replace `/*TRACE_IMPL*​/` with your trace function implementations.
 */
export const tlasBvhShaderSrc = cwbvhCommon + '\n' + cwbvhTraverse + '\n' + cwbvhTlasTraverse + '\n' + rayTrace;

/**
 * Pre-combined shader source for BVH AABB collision detection.
 * Includes: cwbvh_common + collision
 */
export const collisionShaderSrc = cwbvhCommon + '\n' + collision;
