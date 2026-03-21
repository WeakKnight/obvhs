# obvhs

A **Compressed Wide BVH (CWBVH)** acceleration structure library for WebGPU + WASM. Built on the [obvhs](https://github.com/DGriffin91/obvhs) Rust library, compiled to WebAssembly via `wasm-pack` for high-performance BVH construction and GPU traversal in the browser.

## Features

- **CWBVH (8-wide)** — Compressed Wide BVH storing quantized AABBs for 8 child nodes per node
- **TLAS / BLAS** — Two-level acceleration structure supporting multi-instance scenes
- **Procedural Geometry** — Procedural geometry (sphere SDFs, etc.) via `blas_offsets` sentinel values
- **Dynamic BVH** — Runtime insert / delete / rebuild / refit
- **GPU Collision Detection** — BVH-based AABB broad-phase collision detection
- **Zero-copy Upload** — WASM linear memory → GPU Buffer zero-copy upload
- **WGSL Shaders** — Built-in CWBVH traversal, ray tracing, and collision detection shader sources

## Install

```bash
npm install obvhs
```

> **Prerequisites:** Before use, you need to build the WASM module (requires [Rust](https://rustup.rs/) + [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/)):
>
> ```bash
> cd crates/obvhs-wasm
> wasm-pack build --target web --out-dir ../../web/src/wasm/pkg
> ```

## Quick Start

### Option 1: Using Vite (Recommended)

Works seamlessly with `vite-plugin-wasm` — WASM is handled automatically:

```typescript
import { BvhManager, BufferManager, BuildQuality } from 'obvhs';

const bvhManager = new BvhManager();
await bvhManager.init(); // Automatically loads bundled WASM
```

### Option 2: Manual WASM Loading

For webpack / esbuild / native ESM and other environments:

```typescript
import { BvhManager, BufferManager, BuildQuality } from 'obvhs';
import initWasm, * as wasmExports from 'obvhs/wasm';

// 1. Manually initialize WASM (provide your .wasm file URL)
await initWasm('/path/to/obvhs_wasm_bg.wasm');

// 2. Pass the initialized module
const bvhManager = new BvhManager();
await bvhManager.init({ wasmModule: wasmExports as any });
```

---

## Basic Example

### Single-level BVH — Build + GPU Upload

```typescript
import { BvhManager, BufferManager, BuildQuality } from 'obvhs';

// 1. Initialize
const bvhManager = new BvhManager();
await bvhManager.init();

// 2. Prepare triangle data (every 9 floats = 1 triangle: v0.xyz, v1.xyz, v2.xyz)
const triangles = new Float32Array([
  -1, 0, 0,   1, 0, 0,   0, 1, 0,  // Triangle 0
   0, 0, -1,  0, 0, 1,   0, 1, 0,  // Triangle 1
]);

// 3. Build CWBVH
bvhManager.buildCwBvh(triangles, BuildQuality.Medium);
console.log(`Build: ${bvhManager.lastBuildTimeMs}ms, ${bvhManager.nodeCount} nodes`);

// 4. Upload to GPU
const bufferManager = new BufferManager(device);
const { nodes, indices, triangles: triBuf } = bvhManager.uploadToGpu(bufferManager);

// 5. Bind these buffers in your compute shader:
//    @group(0) @binding(0) bvh_nodes
//    @group(0) @binding(1) primitive_indices
//    @group(0) @binding(2) triangles
```

### TLAS / BLAS — Multi-instance Scene + Procedural Geometry

```typescript
import { BvhManager, BufferManager, BuildQuality, generateCornellBoxAt } from 'obvhs';

// 1. Prepare multiple BLAS instances
const blasTriArrays: Float32Array[] = [];
const proceduralAabbs: number[] = [];
const proceduralInstanceIndices: number[] = [];

for (let i = 0; i < 10; i++) {
  const center: [number, number, number] = [i * 3, 0, 0];
  const box = generateCornellBoxAt(center);

  // Triangle mesh BLAS (instance_index = i * 2)
  blasTriArrays.push(box.triangles);

  // Procedural sphere BLAS (instance_index = i * 2 + 1)
  const radius = 0.2 + Math.random() * 0.15;
  const sc = box.sphereCenter;
  proceduralAabbs.push(
    sc[0] - radius, sc[1] - radius, sc[2] - radius,
    sc[0] + radius, sc[1] + radius, sc[2] + radius,
  );
  proceduralInstanceIndices.push(i * 2 + 1);
}

// 2. Build TLAS
const bvhManager = new BvhManager();
await bvhManager.init();

bvhManager.buildTlasScene(
  blasTriArrays,
  new Float32Array(proceduralAabbs),
  new Uint32Array(proceduralInstanceIndices),
  BuildQuality.Medium
);

// 3. Upload to GPU
const bufferManager = new BufferManager(device);
const { nodes, indices, triangles, blasOffsets } = bvhManager.uploadTlasToGpu(bufferManager);
```

### Dynamic BVH — Runtime Updates

```typescript
import { BvhManager, BufferManager, BuildQuality } from 'obvhs';

const bvhManager = new BvhManager();
await bvhManager.init();

// Keep BVH2 to enable dynamic operations
bvhManager.buildCwBvh(triangles, BuildQuality.Medium, /* keepBvh2 */ true);

const bvh = bvhManager.instance!;

// Insert a new triangle
const primId = bvh.insert_triangle(0, 0, 0, 1, 0, 0, 0, 1, 0);

// Resize primitive AABB and reinsert
bvh.resize_primitive(primId, -0.5, -0.5, -0.5, 1.5, 1.5, 0.5);
bvh.reinsert_primitive(primId);

// Batch update
const ids = new Uint32Array([0, 1, 2]);
const aabbs = new Float32Array([
  -1, -1, -1, 1, 1, 1,   // id0
  -2, -2, -2, 2, 2, 2,   // id1
  -3, -3, -3, 3, 3, 3,   // id2
]);
bvh.batch_update_and_reinsert(ids, aabbs);

// Convert back to CWBVH and upload to GPU
bvhManager.convertBvh2ToCwbvh();

const bufferManager = new BufferManager(device);
bvhManager.uploadToGpu(bufferManager);
```

### Using Built-in Shaders

```typescript
import {
  singleBvhShaderSrc,  // Single-level BVH: common + traverse + ray_trace
  tlasBvhShaderSrc,     // TLAS/BLAS: common + traverse + tlas_traverse + ray_trace
  collisionShaderSrc,   // Collision detection: common + collision
} from 'obvhs';

// Inject trace function implementation, replacing the placeholder in the shader
const singleBvhShader = singleBvhShaderSrc.replace('/*TRACE_IMPL*/', `
fn trace_ray(ray: Ray) -> RayHit {
    return traverse_cwbvh_closest(ray);
}
fn trace_shadow_ray(ray: Ray) -> bool {
    return traverse_cwbvh_any(ray);
}
`);

// You can also use individual shader sources to compose your own
import { cwbvhCommonSrc, cwbvhTraverseSrc, collisionSrc } from 'obvhs';
const customShader = cwbvhCommonSrc + '\n' + cwbvhTraverseSrc + '\n' + yourCustomShader;
```

---

## API Reference

### `BvhManager`

Core BVH management class wrapping WASM module loading, building, and GPU upload.

```typescript
import { BvhManager, BuildQuality } from 'obvhs';
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `init()` | `(options?: BvhManagerInitOptions) → Promise<void>` | Load and initialize the WASM module |
| `buildCwBvh()` | `(triangles: Float32Array, quality?: BuildQuality, keepBvh2?: boolean) → void` | Build CWBVH from a triangle array |
| `uploadToGpu()` | `(bufferManager: BufferManager) → { nodes, indices, triangles }` | Upload single-level BVH to GPU |
| `rebuild()` | `(quality?: BuildQuality) → void` | Fully rebuild the CWBVH |
| `convertBvh2ToCwbvh()` | `() → void` | Convert a dynamically modified BVH2 to CWBVH |
| `getTotalAabb()` | `() → { min: [x,y,z], max: [x,y,z] }` | Get the bounding box of the entire BVH |
| `buildTlasScene()` | `(blasTriArrays, proceduralAabbs?, proceduralInstanceIndices?, quality?) → void` | Build a TLAS multi-instance scene |
| `uploadTlasToGpu()` | `(bufferManager: BufferManager) → { nodes, indices, triangles, blasOffsets }` | Upload TLAS data to GPU |
| `destroy()` | `() → void` | Release all WASM resources |

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `lastBuildTimeMs` | `number` | Last build time in milliseconds |
| `nodeCount` | `number` | Number of BVH nodes |
| `triangleCount` | `number` | Number of triangles |
| `tlasStart` | `number` | Starting offset of TLAS nodes in the buffer |
| `instance` | `WasmCwBvhInstance \| null` | Underlying WASM BVH instance (for direct dynamic operations) |

#### `BvhManagerInitOptions`

```typescript
interface BvhManagerInitOptions {
  /** Pre-loaded WASM module exports. If provided, skips default WASM import. */
  wasmModule?: WasmModule;
}
```

#### `BuildQuality`

```typescript
enum BuildQuality {
  Fastest  = 0,  // SAH binning, fastest
  Fast     = 1,
  Medium   = 2,  // Recommended default
  Slow     = 3,
  VerySlow = 4,  // Highest quality
}
```

---

### `BufferManager`

GPU Storage Buffer manager with zero-copy upload from WASM linear memory.

```typescript
import { BufferManager } from 'obvhs';
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `constructor()` | `(device: GPUDevice)` | |
| `uploadBuffer()` | `(name: string, data: Uint8Array, usage?: number) → GPUBuffer` | Upload / update a storage buffer |
| `uploadFromWasm()` | `(name, wasmMemory, ptr, byteLen, usage?) → GPUBuffer` | Zero-copy upload from WASM memory |
| `getBuffer()` | `(name: string) → GPUBuffer \| undefined` | Get an existing buffer by name |
| `createUniformBuffer()` | `(name: string, data: ArrayBuffer) → GPUBuffer` | Create a uniform buffer |
| `createStorageBuffer()` | `(name: string, size: number, readWrite?: boolean) → GPUBuffer` | Create an empty storage buffer |
| `destroy()` | `() → void` | Destroy all buffers |

---

### WGSL Shader Sources

All shaders are exported as string constants, ready for `device.createShaderModule({ code })`:

```typescript
import {
  // ─── Individual shader modules ───
  cwbvhCommonSrc,       // CWBVH common definitions + node unpacking
  cwbvhTraverseSrc,     // Single-level BVH traversal (closest + any hit)
  cwbvhTlasTraverseSrc, // TLAS/BLAS two-level traversal
  rayTraceSrc,          // Ray generation + shading
  fullscreenSrc,        // Fullscreen texture blit (vertex + fragment)
  collisionSrc,         // AABB collision detection

  // ─── Pre-composed shaders ───
  singleBvhShaderSrc,   // = common + traverse + ray_trace
  tlasBvhShaderSrc,     // = common + traverse + tlas_traverse + ray_trace
  collisionShaderSrc,   // = common + collision
} from 'obvhs';
```

---

### Geometry Utilities

Procedural geometry generators outputting a unified `Float32Array` (every 9 floats = 1 triangle).

```typescript
import {
  generateSphere,
  generateIcosphere,
  generateCornellBox,
  generateCornellBoxAt,
  mergeTriangleArrays,
} from 'obvhs';
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `generateSphere()` | `(center, radius, segments?, rings?) → Float32Array` | UV sphere mesh |
| `generateIcosphere()` | `(center?, radius?, subdivisions?) → Float32Array` | Icosphere subdivision mesh |
| `generateCornellBox()` | `() → CornellBoxResult` | Standard Cornell Box scene |
| `generateCornellBoxAt()` | `(center: [x,y,z]) → CornellBoxResult` | Generate Cornell Box at a specified position |
| `mergeTriangleArrays()` | `(...arrays: Float32Array[]) → Float32Array` | Merge multiple triangle arrays |

---

## GPU Shader Bindings

### Single-level BVH (Group 0)

| Binding | Buffer | Type | Description |
|---------|--------|------|-------------|
| 0 | `bvh_nodes` | `array<u32>` | CWBVH nodes (80 bytes/node, quantized AABBs) |
| 1 | `primitive_indices` | `array<u32>` | BVH leaf → triangle index mapping |
| 2 | `triangles` | `array<f32>` | Triangle vertex data (9 floats/tri) |

### TLAS / BLAS (Group 0)

| Binding | Buffer | Type | Description |
|---------|--------|------|-------------|
| 0 | `bvh_nodes` | `array<u32>` | All BLAS + TLAS nodes (concatenated) |
| 1 | `primitive_indices` | `array<u32>` | All BLAS indices + TLAS instance_id mapping |
| 2 | `triangles` | `array<f32>` | All BLAS triangles (concatenated) |
| 3 | `blas_offsets` | `array<u32>` | `instance_id → BLAS node offset` (`0xFFFFFFFF` = procedural) |
| 4+ | *app data* | *user-defined* | Application-layer per-instance data (e.g., `sphere_data`) |

### Data Layout

```
bvh_nodes:         [BLAS_0 nodes | BLAS_1 nodes | ... | TLAS nodes]
                                                        ↑ tlasStart
primitive_indices: [BLAS_0 indices | BLAS_1 indices | ... | TLAS instance_ids]
triangles:         [BLAS_0 tris | BLAS_1 tris | ...]
blas_offsets:      [offset_0, offset_1, ..., 0xFFFFFFFF, ...]  (per instance_id)
```

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────┐
│  Geometry Utils  │────→│   BvhManager     │────→│ BufferManager  │
│  (CPU meshes)    │     │  (WASM bridge)   │     │ (GPU upload)   │
└─────────────────┘     └──────────────────┘     └────────────────┘
                                │                        │
                        WASM Linear Memory          GPU Buffers
                                │                        │
                                ▼                        ▼
                        ┌──────────────┐        ┌────────────────┐
                        │  obvhs_wasm  │        │    WebGPU      │
                        │  (Rust BVH)  │        │  Compute /     │
                        │  build core  │        │  Render Pass   │
                        └──────────────┘        └────────────────┘
                                                        │
                                                ┌───────┴────────┐
                                                │  WGSL Shaders  │
                                                │  (pre-bundled  │
                                                │   as strings)  │
                                                └────────────────┘
```

## Development

```bash
# Build WASM
npm run build:wasm

# Start demo dev server
npm run dev

# Build npm library
npm run build:lib          # Linux/macOS
npm run build:lib:win      # Windows
```

## License

Dual-licensed under [MIT](../LICENSE-MIT) / [Apache 2.0](../LICENSE-APACHE).
