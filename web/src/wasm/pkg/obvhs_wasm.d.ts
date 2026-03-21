/* tslint:disable */
/* eslint-disable */

/**
 * JS-friendly build quality enum.
 */
export enum BuildQuality {
    Fastest = 0,
    Fast = 1,
    Medium = 2,
    Slow = 3,
    VerySlow = 4,
}

/**
 * The main WASM-exported BVH handle.
 * Holds both the CwBvh (for GPU upload) and optionally a Bvh2 (for dynamic updates).
 */
export class WasmCwBvh {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Batch reinsert: resize multiple primitives, then run the reinsertion optimizer.
     * `prim_ids` — list of primitive IDs that have moved.
     * `aabbs_flat` — flat array of new AABBs (6 floats each: min_xyz, max_xyz).
     */
    batch_update_and_reinsert(prim_ids: Uint32Array, aabbs_flat: Float32Array): void;
    /**
     * Build time in milliseconds.
     */
    build_time_ms(): number;
    /**
     * Convert the current BVH2 to CWBVH (after dynamic updates).
     * Only valid if keep_bvh2 was true during construction.
     */
    convert_bvh2_to_cwbvh(): void;
    /**
     * Free this handle explicitly. Also called automatically when GC collects.
     */
    free(): void;
    /**
     * Byte length of primitive_indices (len * 4).
     */
    indices_byte_len(): number;
    /**
     * Number of primitive indices.
     */
    indices_count(): number;
    /**
     * Pointer to the primitive_indices array.
     */
    indices_ptr(): number;
    /**
     * Insert a new triangle into the BVH.
     * Returns the primitive_id of the inserted triangle.
     * After inserting, call `convert_bvh2_to_cwbvh()` to update GPU data.
     */
    insert_triangle(v0x: number, v0y: number, v0z: number, v1x: number, v1y: number, v1z: number, v2x: number, v2y: number, v2z: number): number;
    /**
     * Build a CWBVH from a flat f32 array of triangle vertices.
     * `triangles_flat` must have length divisible by 9 (3 vertices × 3 floats).
     * `quality` selects a build quality preset.
     * `keep_bvh2` — if true, retains the intermediate BVH2 for dynamic updates.
     */
    constructor(triangles_flat: Float32Array, quality: BuildQuality, keep_bvh2: boolean);
    /**
     * Byte length of the CwBvhNode array (nodes.len() * 80).
     */
    nodes_byte_len(): number;
    /**
     * Number of CwBvh nodes.
     */
    nodes_count(): number;
    /**
     * Pointer to the CwBvhNode array in WASM linear memory.
     */
    nodes_ptr(): number;
    /**
     * Partial rebuild: mark moved primitives and rebuild only affected subtrees.
     * `prim_ids` — list of primitive IDs that have moved.
     * `aabbs_flat` — flat array of new AABBs (6 floats each).
     */
    partial_rebuild(prim_ids: Uint32Array, aabbs_flat: Float32Array): void;
    /**
     * Full rebuild of the CWBVH from the current triangle set.
     */
    rebuild(quality: BuildQuality): void;
    /**
     * Refit all nodes in the BVH2 (bottom-up AABB recalculation).
     */
    refit_all(): void;
    /**
     * Reinsert a primitive's node to a potentially better position in the tree.
     * Call after `resize_primitive` for moved objects.
     */
    reinsert_primitive(prim_id: number): void;
    /**
     * Remove a triangle from the BVH by its primitive_id.
     * After removing, call `convert_bvh2_to_cwbvh()` to update GPU data.
     */
    remove_triangle(prim_id: number): void;
    /**
     * Update the AABB of a primitive and refit up the tree.
     * Useful when an object moves — update its bounding box, then refit.
     * `aabb_flat` must be [min_x, min_y, min_z, max_x, max_y, max_z].
     */
    resize_primitive(prim_id: number, min_x: number, min_y: number, min_z: number, max_x: number, max_y: number, max_z: number): void;
    /**
     * Total AABB of the BVH (6 floats: min_x, min_y, min_z, max_x, max_y, max_z).
     */
    total_aabb(): Float32Array;
    /**
     * Byte length of the triangles array.
     * Each Triangle is 48 bytes (3 × Vec3A at 16 bytes each).
     */
    triangles_byte_len(): number;
    /**
     * Number of triangles.
     */
    triangles_count(): number;
    /**
     * Pointer to the Triangle array (for GPU upload).
     */
    triangles_ptr(): number;
}

export class WasmTlasScene {
    free(): void;
    [Symbol.dispose](): void;
    blas_offsets_byte_len(): number;
    blas_offsets_ptr(): number;
    build_time_ms(): number;
    free(): void;
    indices_byte_len(): number;
    indices_ptr(): number;
    /**
     * Build a TLAS scene from mixed BLAS types: triangle meshes + procedural AABBs.
     *
     * **Triangle mesh BLAS:**
     * - `blas_data_flat`: all BLAS triangles concatenated (f32, every 9 = 1 triangle)
     * - `blas_tri_counts`: number of triangles per triangle-mesh BLAS instance (u32 array)
     *
     * **Procedural BLAS (AABB-only, like DXR PROCEDURAL_PRIMITIVE_AABBS):**
     * - `procedural_aabbs_flat`: flat f32 array, every 6 floats = 1 AABB (min_xyz, max_xyz)
     * - `procedural_instance_indices`: for each procedural BLAS, its position in the final
     *   TLAS instance list. E.g., if we have 100 triangle-mesh + 100 procedural instances
     *   interleaved, this array tells us where each procedural one sits.
     *
     * The total TLAS instance count = blas_tri_counts.len() + procedural count.
     * Procedural detection uses blas_offsets sentinel (INVALID_U32).
     * instance_id is stored in primitive_indices at TLAS level.
     *
     * `quality`: build quality for both BLAS and TLAS construction
     */
    constructor(blas_data_flat: Float32Array, blas_tri_counts: Uint32Array, procedural_aabbs_flat: Float32Array, procedural_instance_indices: Uint32Array, quality: BuildQuality);
    nodes_byte_len(): number;
    nodes_ptr(): number;
    tlas_start(): number;
    total_node_count(): number;
    total_triangle_count(): number;
    triangles_byte_len(): number;
    triangles_ptr(): number;
}

/**
 * Initialize panic hook for better error messages in the browser console.
 */
export function init(): void;

/**
 * Return a reference to the WebAssembly linear memory.
 * This allows JS to create zero-copy views into WASM heap data.
 */
export function wasm_memory(): any;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_wasmcwbvh_free: (a: number, b: number) => void;
    readonly __wbg_wasmtlasscene_free: (a: number, b: number) => void;
    readonly wasmcwbvh_batch_update_and_reinsert: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly wasmcwbvh_build_time_ms: (a: number) => number;
    readonly wasmcwbvh_convert_bvh2_to_cwbvh: (a: number) => [number, number];
    readonly wasmcwbvh_free: (a: number) => void;
    readonly wasmcwbvh_indices_byte_len: (a: number) => number;
    readonly wasmcwbvh_indices_count: (a: number) => number;
    readonly wasmcwbvh_indices_ptr: (a: number) => number;
    readonly wasmcwbvh_insert_triangle: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number) => [number, number, number];
    readonly wasmcwbvh_new: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly wasmcwbvh_nodes_byte_len: (a: number) => number;
    readonly wasmcwbvh_nodes_count: (a: number) => number;
    readonly wasmcwbvh_nodes_ptr: (a: number) => number;
    readonly wasmcwbvh_partial_rebuild: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly wasmcwbvh_rebuild: (a: number, b: number) => void;
    readonly wasmcwbvh_refit_all: (a: number) => [number, number];
    readonly wasmcwbvh_reinsert_primitive: (a: number, b: number) => [number, number];
    readonly wasmcwbvh_remove_triangle: (a: number, b: number) => [number, number];
    readonly wasmcwbvh_resize_primitive: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number];
    readonly wasmcwbvh_total_aabb: (a: number) => [number, number];
    readonly wasmcwbvh_triangles_byte_len: (a: number) => number;
    readonly wasmcwbvh_triangles_count: (a: number) => number;
    readonly wasmcwbvh_triangles_ptr: (a: number) => number;
    readonly wasmtlasscene_blas_offsets_byte_len: (a: number) => number;
    readonly wasmtlasscene_blas_offsets_ptr: (a: number) => number;
    readonly wasmtlasscene_build_time_ms: (a: number) => number;
    readonly wasmtlasscene_free: (a: number) => void;
    readonly wasmtlasscene_indices_byte_len: (a: number) => number;
    readonly wasmtlasscene_indices_ptr: (a: number) => number;
    readonly wasmtlasscene_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => [number, number, number];
    readonly wasmtlasscene_nodes_byte_len: (a: number) => number;
    readonly wasmtlasscene_nodes_ptr: (a: number) => number;
    readonly wasmtlasscene_tlas_start: (a: number) => number;
    readonly wasmtlasscene_total_node_count: (a: number) => number;
    readonly wasmtlasscene_total_triangle_count: (a: number) => number;
    readonly wasmtlasscene_triangles_byte_len: (a: number) => number;
    readonly wasmtlasscene_triangles_ptr: (a: number) => number;
    readonly init: () => void;
    readonly wasm_memory: () => any;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
