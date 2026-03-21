/* @ts-self-types="./obvhs_wasm.d.ts" */

/**
 * JS-friendly build quality enum.
 * @enum {0 | 1 | 2 | 3 | 4}
 */
export const BuildQuality = Object.freeze({
    Fastest: 0, "0": "Fastest",
    Fast: 1, "1": "Fast",
    Medium: 2, "2": "Medium",
    Slow: 3, "3": "Slow",
    VerySlow: 4, "4": "VerySlow",
});

/**
 * The main WASM-exported BVH handle.
 * Holds both the CwBvh (for GPU upload) and optionally a Bvh2 (for dynamic updates).
 */
export class WasmCwBvh {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmCwBvhFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmcwbvh_free(ptr, 0);
    }
    /**
     * Batch reinsert: resize multiple primitives, then run the reinsertion optimizer.
     * `prim_ids` — list of primitive IDs that have moved.
     * `aabbs_flat` — flat array of new AABBs (6 floats each: min_xyz, max_xyz).
     * @param {Uint32Array} prim_ids
     * @param {Float32Array} aabbs_flat
     */
    batch_update_and_reinsert(prim_ids, aabbs_flat) {
        const ptr0 = passArray32ToWasm0(prim_ids, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(aabbs_flat, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmcwbvh_batch_update_and_reinsert(this.__wbg_ptr, ptr0, len0, ptr1, len1);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Build time in milliseconds.
     * @returns {number}
     */
    build_time_ms() {
        const ret = wasm.wasmcwbvh_build_time_ms(this.__wbg_ptr);
        return ret;
    }
    /**
     * Convert the current BVH2 to CWBVH (after dynamic updates).
     * Only valid if keep_bvh2 was true during construction.
     */
    convert_bvh2_to_cwbvh() {
        const ret = wasm.wasmcwbvh_convert_bvh2_to_cwbvh(this.__wbg_ptr);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Free this handle explicitly. Also called automatically when GC collects.
     */
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.wasmcwbvh_free(ptr);
    }
    /**
     * Byte length of primitive_indices (len * 4).
     * @returns {number}
     */
    indices_byte_len() {
        const ret = wasm.wasmcwbvh_indices_byte_len(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Number of primitive indices.
     * @returns {number}
     */
    indices_count() {
        const ret = wasm.wasmcwbvh_indices_count(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Pointer to the primitive_indices array.
     * @returns {number}
     */
    indices_ptr() {
        const ret = wasm.wasmcwbvh_indices_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Insert a new triangle into the BVH.
     * Returns the primitive_id of the inserted triangle.
     * After inserting, call `convert_bvh2_to_cwbvh()` to update GPU data.
     * @param {number} v0x
     * @param {number} v0y
     * @param {number} v0z
     * @param {number} v1x
     * @param {number} v1y
     * @param {number} v1z
     * @param {number} v2x
     * @param {number} v2y
     * @param {number} v2z
     * @returns {number}
     */
    insert_triangle(v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z) {
        const ret = wasm.wasmcwbvh_insert_triangle(this.__wbg_ptr, v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] >>> 0;
    }
    /**
     * Build a CWBVH from a flat f32 array of triangle vertices.
     * `triangles_flat` must have length divisible by 9 (3 vertices × 3 floats).
     * `quality` selects a build quality preset.
     * `keep_bvh2` — if true, retains the intermediate BVH2 for dynamic updates.
     * @param {Float32Array} triangles_flat
     * @param {BuildQuality} quality
     * @param {boolean} keep_bvh2
     */
    constructor(triangles_flat, quality, keep_bvh2) {
        const ptr0 = passArrayF32ToWasm0(triangles_flat, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmcwbvh_new(ptr0, len0, quality, keep_bvh2);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        WasmCwBvhFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Byte length of the CwBvhNode array (nodes.len() * 80).
     * @returns {number}
     */
    nodes_byte_len() {
        const ret = wasm.wasmcwbvh_nodes_byte_len(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Number of CwBvh nodes.
     * @returns {number}
     */
    nodes_count() {
        const ret = wasm.wasmcwbvh_nodes_count(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Pointer to the CwBvhNode array in WASM linear memory.
     * @returns {number}
     */
    nodes_ptr() {
        const ret = wasm.wasmcwbvh_nodes_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Partial rebuild: mark moved primitives and rebuild only affected subtrees.
     * `prim_ids` — list of primitive IDs that have moved.
     * `aabbs_flat` — flat array of new AABBs (6 floats each).
     * @param {Uint32Array} prim_ids
     * @param {Float32Array} aabbs_flat
     */
    partial_rebuild(prim_ids, aabbs_flat) {
        const ptr0 = passArray32ToWasm0(prim_ids, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(aabbs_flat, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmcwbvh_partial_rebuild(this.__wbg_ptr, ptr0, len0, ptr1, len1);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Full rebuild of the CWBVH from the current triangle set.
     * @param {BuildQuality} quality
     */
    rebuild(quality) {
        wasm.wasmcwbvh_rebuild(this.__wbg_ptr, quality);
    }
    /**
     * Refit all nodes in the BVH2 (bottom-up AABB recalculation).
     */
    refit_all() {
        const ret = wasm.wasmcwbvh_refit_all(this.__wbg_ptr);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Reinsert a primitive's node to a potentially better position in the tree.
     * Call after `resize_primitive` for moved objects.
     * @param {number} prim_id
     */
    reinsert_primitive(prim_id) {
        const ret = wasm.wasmcwbvh_reinsert_primitive(this.__wbg_ptr, prim_id);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Remove a triangle from the BVH by its primitive_id.
     * After removing, call `convert_bvh2_to_cwbvh()` to update GPU data.
     * @param {number} prim_id
     */
    remove_triangle(prim_id) {
        const ret = wasm.wasmcwbvh_remove_triangle(this.__wbg_ptr, prim_id);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Update the AABB of a primitive and refit up the tree.
     * Useful when an object moves — update its bounding box, then refit.
     * `aabb_flat` must be [min_x, min_y, min_z, max_x, max_y, max_z].
     * @param {number} prim_id
     * @param {number} min_x
     * @param {number} min_y
     * @param {number} min_z
     * @param {number} max_x
     * @param {number} max_y
     * @param {number} max_z
     */
    resize_primitive(prim_id, min_x, min_y, min_z, max_x, max_y, max_z) {
        const ret = wasm.wasmcwbvh_resize_primitive(this.__wbg_ptr, prim_id, min_x, min_y, min_z, max_x, max_y, max_z);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Total AABB of the BVH (6 floats: min_x, min_y, min_z, max_x, max_y, max_z).
     * @returns {Float32Array}
     */
    total_aabb() {
        const ret = wasm.wasmcwbvh_total_aabb(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Byte length of the triangles array.
     * Each Triangle is 48 bytes (3 × Vec3A at 16 bytes each).
     * @returns {number}
     */
    triangles_byte_len() {
        const ret = wasm.wasmcwbvh_triangles_byte_len(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Number of triangles.
     * @returns {number}
     */
    triangles_count() {
        const ret = wasm.wasmcwbvh_triangles_count(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Pointer to the Triangle array (for GPU upload).
     * @returns {number}
     */
    triangles_ptr() {
        const ret = wasm.wasmcwbvh_triangles_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) WasmCwBvh.prototype[Symbol.dispose] = WasmCwBvh.prototype.free;

export class WasmTlasScene {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmTlasSceneFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmtlasscene_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    blas_offsets_byte_len() {
        const ret = wasm.wasmtlasscene_blas_offsets_byte_len(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    blas_offsets_ptr() {
        const ret = wasm.wasmtlasscene_blas_offsets_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    build_time_ms() {
        const ret = wasm.wasmtlasscene_build_time_ms(this.__wbg_ptr);
        return ret;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.wasmtlasscene_free(ptr);
    }
    /**
     * @returns {number}
     */
    indices_byte_len() {
        const ret = wasm.wasmtlasscene_indices_byte_len(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    indices_ptr() {
        const ret = wasm.wasmtlasscene_indices_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
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
     * @param {Float32Array} blas_data_flat
     * @param {Uint32Array} blas_tri_counts
     * @param {Float32Array} procedural_aabbs_flat
     * @param {Uint32Array} procedural_instance_indices
     * @param {BuildQuality} quality
     */
    constructor(blas_data_flat, blas_tri_counts, procedural_aabbs_flat, procedural_instance_indices, quality) {
        const ptr0 = passArrayF32ToWasm0(blas_data_flat, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray32ToWasm0(blas_tri_counts, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passArrayF32ToWasm0(procedural_aabbs_flat, wasm.__wbindgen_malloc);
        const len2 = WASM_VECTOR_LEN;
        const ptr3 = passArray32ToWasm0(procedural_instance_indices, wasm.__wbindgen_malloc);
        const len3 = WASM_VECTOR_LEN;
        const ret = wasm.wasmtlasscene_new(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, quality);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        WasmTlasSceneFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {number}
     */
    nodes_byte_len() {
        const ret = wasm.wasmtlasscene_nodes_byte_len(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    nodes_ptr() {
        const ret = wasm.wasmtlasscene_nodes_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    tlas_start() {
        const ret = wasm.wasmtlasscene_tlas_start(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    total_node_count() {
        const ret = wasm.wasmtlasscene_total_node_count(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    total_triangle_count() {
        const ret = wasm.wasmtlasscene_total_triangle_count(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    triangles_byte_len() {
        const ret = wasm.wasmtlasscene_triangles_byte_len(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    triangles_ptr() {
        const ret = wasm.wasmtlasscene_triangles_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) WasmTlasScene.prototype[Symbol.dispose] = WasmTlasScene.prototype.free;

/**
 * Initialize panic hook for better error messages in the browser console.
 */
export function init() {
    wasm.init();
}

/**
 * Return a reference to the WebAssembly linear memory.
 * This allows JS to create zero-copy views into WASM heap data.
 * @returns {any}
 */
export function wasm_memory() {
    const ret = wasm.wasm_memory();
    return ret;
}

function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg_Error_83742b46f01ce22d: function(arg0, arg1) {
            const ret = Error(getStringFromWasm0(arg0, arg1));
            return ret;
        },
        __wbg___wbindgen_memory_edb3f01e3930bbf6: function() {
            const ret = wasm.memory;
            return ret;
        },
        __wbg___wbindgen_throw_6ddd609b62940d55: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg_error_a6fa202b58aa1cd3: function(arg0, arg1) {
            let deferred0_0;
            let deferred0_1;
            try {
                deferred0_0 = arg0;
                deferred0_1 = arg1;
                console.error(getStringFromWasm0(arg0, arg1));
            } finally {
                wasm.__wbindgen_free(deferred0_0, deferred0_1, 1);
            }
        },
        __wbg_new_227d7c05414eb861: function() {
            const ret = new Error();
            return ret;
        },
        __wbg_now_16f0c993d5dd6c27: function() {
            const ret = Date.now();
            return ret;
        },
        __wbg_stack_3b0d974bbf31e44f: function(arg0, arg1) {
            const ret = arg1.stack;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./obvhs_wasm_bg.js": import0,
    };
}

const WasmCwBvhFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmcwbvh_free(ptr >>> 0, 1));
const WasmTlasSceneFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmtlasscene_free(ptr >>> 0, 1));

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

let cachedUint32ArrayMemory0 = null;
function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function passArray32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getUint32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayF32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getFloat32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasm;
function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    wasmModule = module;
    cachedDataViewMemory0 = null;
    cachedFloat32ArrayMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('obvhs_wasm_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
