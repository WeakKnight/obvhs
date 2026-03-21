# obvhs

WebGPU + WASM 实现的 **Compressed Wide BVH (CWBVH)** 加速结构库。基于 [obvhs](https://github.com/DGriffin91/obvhs) Rust 库，通过 `wasm-pack` 编译为 WebAssembly，在浏览器中实现高性能 BVH 构建与 GPU 遍历。

## Features

- **CWBVH (8-wide)** — 压缩宽 BVH，单节点存储 8 个子节点的量化 AABB
- **TLAS / BLAS** — 双层加速结构，支持多实例场景
- **Procedural Geometry** — 程序化几何（球体 SDF 等），通过 `blas_offsets` sentinel 区分
- **Dynamic BVH** — 运行时插入 / 删除 / 重建 / refit
- **GPU Collision Detection** — 基于 BVH 的 AABB 宽相碰撞检测
- **Zero-copy Upload** — WASM 线性内存 → GPU Buffer 零拷贝上传
- **WGSL Shaders** — 内置 CWBVH 遍历、光线追踪、碰撞检测 shader 源码

## Install

```bash
npm install obvhs
```

> **前置条件：** 使用前需要构建 WASM 模块（需要 [Rust](https://rustup.rs/) + [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/)）：
>
> ```bash
> cd crates/obvhs-wasm
> wasm-pack build --target web --out-dir ../../web/src/wasm/pkg
> ```

## Quick Start

### 方式一：使用 Vite（推荐）

配合 `vite-plugin-wasm` 使用，WASM 自动处理：

```typescript
import { BvhManager, BufferManager, BuildQuality } from 'obvhs';

const bvhManager = new BvhManager();
await bvhManager.init(); // 自动加载 bundled WASM
```

### 方式二：自行管理 WASM 加载

适用于 webpack / esbuild / 原生 ESM 等：

```typescript
import { BvhManager, BufferManager, BuildQuality } from 'obvhs';
import initWasm, * as wasmExports from 'obvhs/wasm';

// 1. 手动初始化 WASM（传入你的 .wasm 文件 URL）
await initWasm('/path/to/obvhs_wasm_bg.wasm');

// 2. 传入已初始化的模块
const bvhManager = new BvhManager();
await bvhManager.init({ wasmModule: wasmExports as any });
```

---

## Basic Example

### 单层 BVH — 构建 + GPU 上传

```typescript
import { BvhManager, BufferManager, BuildQuality } from 'obvhs';

// 1. 初始化
const bvhManager = new BvhManager();
await bvhManager.init();

// 2. 准备三角形数据 (每 9 个 float = 1 个三角形: v0.xyz, v1.xyz, v2.xyz)
const triangles = new Float32Array([
  -1, 0, 0,   1, 0, 0,   0, 1, 0,  // Triangle 0
   0, 0, -1,  0, 0, 1,   0, 1, 0,  // Triangle 1
]);

// 3. 构建 CWBVH
bvhManager.buildCwBvh(triangles, BuildQuality.Medium);
console.log(`Build: ${bvhManager.lastBuildTimeMs}ms, ${bvhManager.nodeCount} nodes`);

// 4. 上传到 GPU
const bufferManager = new BufferManager(device);
const { nodes, indices, triangles: triBuf } = bvhManager.uploadToGpu(bufferManager);

// 5. 在 compute shader 中绑定这些 buffer:
//    @group(0) @binding(0) bvh_nodes
//    @group(0) @binding(1) primitive_indices
//    @group(0) @binding(2) triangles
```

### TLAS / BLAS — 多实例场景 + 程序化几何

```typescript
import { BvhManager, BufferManager, BuildQuality, generateCornellBoxAt } from 'obvhs';

// 1. 准备多个 BLAS 实例
const blasTriArrays: Float32Array[] = [];
const proceduralAabbs: number[] = [];
const proceduralInstanceIndices: number[] = [];

for (let i = 0; i < 10; i++) {
  const center: [number, number, number] = [i * 3, 0, 0];
  const box = generateCornellBoxAt(center);

  // 三角形网格 BLAS（instance_index = i * 2）
  blasTriArrays.push(box.triangles);

  // 程序化球体 BLAS（instance_index = i * 2 + 1）
  const radius = 0.2 + Math.random() * 0.15;
  const sc = box.sphereCenter;
  proceduralAabbs.push(
    sc[0] - radius, sc[1] - radius, sc[2] - radius,
    sc[0] + radius, sc[1] + radius, sc[2] + radius,
  );
  proceduralInstanceIndices.push(i * 2 + 1);
}

// 2. 构建 TLAS
const bvhManager = new BvhManager();
await bvhManager.init();

bvhManager.buildTlasScene(
  blasTriArrays,
  new Float32Array(proceduralAabbs),
  new Uint32Array(proceduralInstanceIndices),
  BuildQuality.Medium
);

// 3. 上传到 GPU
const bufferManager = new BufferManager(device);
const { nodes, indices, triangles, blasOffsets } = bvhManager.uploadTlasToGpu(bufferManager);
```

### 动态 BVH — 运行时更新

```typescript
import { BvhManager, BufferManager, BuildQuality } from 'obvhs';

const bvhManager = new BvhManager();
await bvhManager.init();

// 保留 BVH2 以支持动态操作
bvhManager.buildCwBvh(triangles, BuildQuality.Medium, /* keepBvh2 */ true);

const bvh = bvhManager.instance!;

// 插入新三角形
const primId = bvh.insert_triangle(0, 0, 0, 1, 0, 0, 0, 1, 0);

// 修改图元 AABB 并重插入
bvh.resize_primitive(primId, -0.5, -0.5, -0.5, 1.5, 1.5, 0.5);
bvh.reinsert_primitive(primId);

// 批量更新
const ids = new Uint32Array([0, 1, 2]);
const aabbs = new Float32Array([
  -1, -1, -1, 1, 1, 1,   // id0
  -2, -2, -2, 2, 2, 2,   // id1
  -3, -3, -3, 3, 3, 3,   // id2
]);
bvh.batch_update_and_reinsert(ids, aabbs);

// 转换回 CWBVH 并上传 GPU
bvhManager.convertBvh2ToCwbvh();

const bufferManager = new BufferManager(device);
bvhManager.uploadToGpu(bufferManager);
```

### 使用内置 Shader

```typescript
import {
  singleBvhShaderSrc,  // 单层 BVH: common + traverse + ray_trace
  tlasBvhShaderSrc,     // TLAS/BLAS: common + traverse + tlas_traverse + ray_trace
  collisionShaderSrc,   // 碰撞检测: common + collision
} from 'obvhs';

// 注入 trace 函数实现，替换 shader 中的占位符
const singleBvhShader = singleBvhShaderSrc.replace('/*TRACE_IMPL*/', `
fn trace_ray(ray: Ray) -> RayHit {
    return traverse_cwbvh_closest(ray);
}
fn trace_shadow_ray(ray: Ray) -> bool {
    return traverse_cwbvh_any(ray);
}
`);

// 也可以使用独立的 shader 源码自行组合
import { cwbvhCommonSrc, cwbvhTraverseSrc, collisionSrc } from 'obvhs';
const customShader = cwbvhCommonSrc + '\n' + cwbvhTraverseSrc + '\n' + yourCustomShader;
```

---

## API Reference

### `BvhManager`

BVH 核心管理类，封装 WASM 模块的加载、构建和 GPU 上传。

```typescript
import { BvhManager, BuildQuality } from 'obvhs';
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `init()` | `(options?: BvhManagerInitOptions) → Promise<void>` | 加载并初始化 WASM 模块 |
| `buildCwBvh()` | `(triangles: Float32Array, quality?: BuildQuality, keepBvh2?: boolean) → void` | 从三角形数组构建 CWBVH |
| `uploadToGpu()` | `(bufferManager: BufferManager) → { nodes, indices, triangles }` | 上传单层 BVH 到 GPU |
| `rebuild()` | `(quality?: BuildQuality) → void` | 完全重建 CWBVH |
| `convertBvh2ToCwbvh()` | `() → void` | 将动态修改后的 BVH2 转换为 CWBVH |
| `getTotalAabb()` | `() → { min: [x,y,z], max: [x,y,z] }` | 获取整棵 BVH 的包围盒 |
| `buildTlasScene()` | `(blasTriArrays, proceduralAabbs?, proceduralInstanceIndices?, quality?) → void` | 构建 TLAS 多实例场景 |
| `uploadTlasToGpu()` | `(bufferManager: BufferManager) → { nodes, indices, triangles, blasOffsets }` | 上传 TLAS 数据到 GPU |
| `destroy()` | `() → void` | 释放所有 WASM 资源 |

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `lastBuildTimeMs` | `number` | 上次构建耗时（毫秒） |
| `nodeCount` | `number` | BVH 节点数量 |
| `triangleCount` | `number` | 三角形数量 |
| `tlasStart` | `number` | TLAS 节点在 buffer 中的起始偏移 |
| `instance` | `WasmCwBvhInstance \| null` | 底层 WASM BVH 实例（可直接调用动态操作） |

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
  Fastest  = 0,  // SAH binning, 最快
  Fast     = 1,
  Medium   = 2,  // 推荐默认值
  Slow     = 3,
  VerySlow = 4,  // 最高质量
}
```

---

### `BufferManager`

GPU Storage Buffer 管理器，支持从 WASM 线性内存零拷贝上传。

```typescript
import { BufferManager } from 'obvhs';
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `constructor()` | `(device: GPUDevice)` | |
| `uploadBuffer()` | `(name: string, data: Uint8Array, usage?: number) → GPUBuffer` | 上传 / 更新 storage buffer |
| `uploadFromWasm()` | `(name, wasmMemory, ptr, byteLen, usage?) → GPUBuffer` | 从 WASM 内存零拷贝上传 |
| `getBuffer()` | `(name: string) → GPUBuffer \| undefined` | 按名称获取已有 buffer |
| `createUniformBuffer()` | `(name: string, data: ArrayBuffer) → GPUBuffer` | 创建 uniform buffer |
| `createStorageBuffer()` | `(name: string, size: number, readWrite?: boolean) → GPUBuffer` | 创建空 storage buffer |
| `destroy()` | `() → void` | 销毁所有 buffer |

---

### WGSL Shader Sources

所有 shader 作为字符串常量导出，可直接用于 `device.createShaderModule({ code })`:

```typescript
import {
  // ─── 独立 shader 模块 ───
  cwbvhCommonSrc,       // CWBVH 公共定义 + 节点解包
  cwbvhTraverseSrc,     // 单层 BVH 遍历（closest + any hit）
  cwbvhTlasTraverseSrc, // TLAS/BLAS 双层遍历
  rayTraceSrc,          // 光线生成 + 着色
  fullscreenSrc,        // 全屏纹理 blit（vertex + fragment）
  collisionSrc,         // AABB 碰撞检测

  // ─── 预组合 shader ───
  singleBvhShaderSrc,   // = common + traverse + ray_trace
  tlasBvhShaderSrc,     // = common + traverse + tlas_traverse + ray_trace
  collisionShaderSrc,   // = common + collision
} from 'obvhs';
```

---

### Geometry Utilities

程序化几何生成工具，输出统一的 `Float32Array`（每 9 个 float = 1 个三角形）。

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
| `generateSphere()` | `(center, radius, segments?, rings?) → Float32Array` | UV 球体网格 |
| `generateIcosphere()` | `(center?, radius?, subdivisions?) → Float32Array` | Icosphere 细分网格 |
| `generateCornellBox()` | `() → CornellBoxResult` | 标准 Cornell Box 场景 |
| `generateCornellBoxAt()` | `(center: [x,y,z]) → CornellBoxResult` | 在指定位置生成 Cornell Box |
| `mergeTriangleArrays()` | `(...arrays: Float32Array[]) → Float32Array` | 合并多个三角形数组 |

---

## GPU Shader Bindings

### 单层 BVH (Group 0)

| Binding | Buffer | Type | Description |
|---------|--------|------|-------------|
| 0 | `bvh_nodes` | `array<u32>` | CWBVH 节点（80 bytes/node，量化 AABB） |
| 1 | `primitive_indices` | `array<u32>` | BVH 叶节点 → 三角形索引映射 |
| 2 | `triangles` | `array<f32>` | 三角形顶点数据（9 floats/tri） |

### TLAS / BLAS (Group 0)

| Binding | Buffer | Type | Description |
|---------|--------|------|-------------|
| 0 | `bvh_nodes` | `array<u32>` | 所有 BLAS + TLAS 节点（拼接） |
| 1 | `primitive_indices` | `array<u32>` | 所有 BLAS 索引 + TLAS instance_id 映射 |
| 2 | `triangles` | `array<f32>` | 所有 BLAS 三角形（拼接） |
| 3 | `blas_offsets` | `array<u32>` | `instance_id → BLAS 节点偏移`（`0xFFFFFFFF` = procedural） |
| 4+ | *app data* | *user-defined* | 应用层 per-instance 数据（如 `sphere_data`） |

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
# 构建 WASM
npm run build:wasm

# 启动 demo dev server
npm run dev

# 构建 npm 库
npm run build:lib          # Linux/macOS
npm run build:lib:win      # Windows
```

## License

Dual-licensed under [MIT](../LICENSE-MIT) / [Apache 2.0](../LICENSE-APACHE).
