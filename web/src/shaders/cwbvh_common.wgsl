// ============================================================================
// cwbvh_common.wgsl — CWBVH common definitions and node unpacking utilities
// Ported from obvhs (Rust) — CwBvhNode layout: 80 bytes = 20 u32 words
// ============================================================================

// ─── Constants ───
const BRANCHING: u32 = 8u;
const EPSILON: f32 = 0.0001;
const TRAVERSAL_STACK_SIZE: u32 = 32u;
const INVALID_U32: u32 = 0xFFFFFFFFu;

// ─── Data Types ───

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
    inv_direction: vec3<f32>,
    tmin: f32,
    tmax: f32,
};

struct RayHit {
    primitive_id: u32,
    t: f32,
};

struct Aabb {
    aabb_min: vec3<f32>,
    aabb_max: vec3<f32>,
};

// ─── Bindings ───
// BVH nodes as flat u32 array: each node = 20 consecutive u32s (80 bytes)
@group(0) @binding(0) var<storage, read> bvh_nodes: array<u32>;
// Primitive index remapping
@group(0) @binding(1) var<storage, read> primitive_indices: array<u32>;
// Triangles as flat f32 array: each triangle = 12 floats (3 vertices × vec3 + padding)
// Layout: [v0.x, v0.y, v0.z, pad, v1.x, v1.y, v1.z, pad, v2.x, v2.y, v2.z, pad]
// This matches Vec3A (16-byte aligned) layout from Rust
@group(0) @binding(2) var<storage, read> triangles: array<f32>;

// ─── Ray Creation ───

fn make_ray(origin: vec3<f32>, direction: vec3<f32>, tmin: f32, tmax: f32) -> Ray {
    var ray: Ray;
    ray.origin = origin;
    ray.direction = direction;
    ray.inv_direction = 1.0 / direction;
    ray.tmin = tmin;
    ray.tmax = tmax;
    return ray;
}

// ─── Node Field Access (from flat u32 array) ───
// CwBvhNode memory layout (80 bytes = 20 u32 words):
//   word  0-2:  p (vec3<f32>)           — node AABB minimum point
//   word  3:    e[0..3] + imask         — 3 exponent bytes + internal mask byte
//   word  4:    child_base_idx          — base index for child internal nodes
//   word  5:    primitive_base_idx      — base index for primitive_indices
//   word  6-7:  child_meta[0..8]        — 8 bytes of child metadata, packed as 2 u32
//   word  8-9:  child_min_x[0..8]       — quantized child AABB min x
//   word 10-11: child_max_x[0..8]
//   word 12-13: child_min_y[0..8]
//   word 14-15: child_max_y[0..8]
//   word 16-17: child_min_z[0..8]
//   word 18-19: child_max_z[0..8]

fn node_offset(node_index: u32) -> u32 {
    return node_index * 20u;
}

fn load_node_p(off: u32) -> vec3<f32> {
    return vec3<f32>(
        bitcast<f32>(bvh_nodes[off + 0u]),
        bitcast<f32>(bvh_nodes[off + 1u]),
        bitcast<f32>(bvh_nodes[off + 2u])
    );
}

fn load_node_e(off: u32) -> vec3<u32> {
    let packed = bvh_nodes[off + 3u];
    return vec3<u32>(
        packed & 0xFFu,
        (packed >> 8u) & 0xFFu,
        (packed >> 16u) & 0xFFu
    );
}

fn load_node_imask(off: u32) -> u32 {
    return (bvh_nodes[off + 3u] >> 24u) & 0xFFu;
}

fn load_child_base_idx(off: u32) -> u32 {
    return bvh_nodes[off + 4u];
}

fn load_primitive_base_idx(off: u32) -> u32 {
    return bvh_nodes[off + 5u];
}

/// Load child_meta byte for child index [0..8)
fn load_child_meta(off: u32, child: u32) -> u32 {
    // child_meta is at words 6-7 (bytes 24-31)
    let word_idx = off + 6u + (child >> 2u);     // word 6 or 7
    let byte_shift = (child & 3u) * 8u;
    return (bvh_nodes[word_idx] >> byte_shift) & 0xFFu;
}

/// Load a quantized AABB byte for a given child.
/// axis: 0=x, 1=y, 2=z; is_max: 0=min, 1=max
fn load_child_q(off: u32, child: u32, axis: u32, is_max: u32) -> u32 {
    // Quantized bytes start at word 8
    // Layout: min_x(8-9), max_x(10-11), min_y(12-13), max_y(14-15), min_z(16-17), max_z(18-19)
    let base_word = off + 8u + axis * 4u + is_max * 2u;
    let word_idx = base_word + (child >> 2u);
    let byte_shift = (child & 3u) * 8u;
    return (bvh_nodes[word_idx] >> byte_shift) & 0xFFu;
}

// ─── Utility Functions ───

/// Decode compressed exponent to float: e[i] << 23 → float with only exponent bits set
/// Result is 2^(e_i - 127) — a power-of-two scale factor
fn compute_extent(e: vec3<u32>) -> vec3<f32> {
    return vec3<f32>(
        bitcast<f32>(e.x << 23u),
        bitcast<f32>(e.y << 23u),
        bitcast<f32>(e.z << 23u)
    );
}

/// Compute the ray octant encoding (replicated 4 times in a u32).
/// Each byte encodes: bit2 = (dir.x >= 0), bit1 = (dir.y >= 0), bit0 = (dir.z >= 0)
fn ray_get_octant_inv4(dir: vec3<f32>) -> u32 {
    return select(0u, 0x04040404u, dir.x >= 0.0)
         | select(0u, 0x02020202u, dir.y >= 0.0)
         | select(0u, 0x01010101u, dir.z >= 0.0);
}

// ─── Triangle Access ───
// Triangle in WASM memory: 3 × Vec3A = 3 × 16 bytes = 48 bytes = 12 floats
// Layout per triangle: [v0.x, v0.y, v0.z, pad, v1.x, v1.y, v1.z, pad, v2.x, v2.y, v2.z, pad]

fn load_triangle_v0(prim_id: u32) -> vec3<f32> {
    let base = prim_id * 12u;
    return vec3<f32>(triangles[base + 0u], triangles[base + 1u], triangles[base + 2u]);
}

fn load_triangle_v1(prim_id: u32) -> vec3<f32> {
    let base = prim_id * 12u;
    return vec3<f32>(triangles[base + 4u], triangles[base + 5u], triangles[base + 6u]);
}

fn load_triangle_v2(prim_id: u32) -> vec3<f32> {
    let base = prim_id * 12u;
    return vec3<f32>(triangles[base + 8u], triangles[base + 9u], triangles[base + 10u]);
}
