// ============================================================================
// collision.wgsl — CWBVH AABB broadphase collision detection
// Each thread = one query object; traverses BVH to find overlapping primitives.
// ============================================================================

// NOTE: Must be concatenated after cwbvh_common.wgsl

// ─── Collision-specific bindings ───

struct CollisionUniforms {
    query_count: u32,
    max_pairs_per_query: u32,
    _pad0: u32,
    _pad1: u32,
};

struct CollisionPair {
    query_id: u32,
    prim_id: u32,
};

@group(1) @binding(0) var<uniform> collision_uniforms: CollisionUniforms;
// Query AABBs: flat f32 array, 6 floats per query (min_x, min_y, min_z, max_x, max_y, max_z)
@group(1) @binding(1) var<storage, read> query_aabbs: array<f32>;
// Output collision pairs
@group(1) @binding(2) var<storage, read_write> collision_pairs: array<CollisionPair>;
// Atomic counter for the number of output pairs
@group(1) @binding(3) var<storage, read_write> pair_count: atomic<u32>;

// ─── AABB-Node Intersection ───
// Tests query AABB against all 8 quantized child AABBs of a CwBvhNode.
// Returns hit_mask (same encoding as ray version).
fn intersect_aabb_node(off: u32, query: Aabb, oct_inv4: u32) -> u32 {
    let p = load_node_p(off);
    let e = load_node_e(off);
    let extent = compute_extent(e);
    let extent_rcp = 1.0 / extent;

    // Transform query AABB to node's quantized local space
    let local_min = (query.aabb_min - p) * extent_rcp;
    let local_max = (query.aabb_max - p) * extent_rcp;

    var hit_mask: u32 = 0u;

    for (var child: u32 = 0u; child < 8u; child = child + 1u) {
        let cmeta = load_child_meta(off, child);
        if cmeta == 0u { continue; }

        // Load child quantized AABB
        let c_min_x = f32(load_child_q(off, child, 0u, 0u));
        let c_max_x = f32(load_child_q(off, child, 0u, 1u));
        let c_min_y = f32(load_child_q(off, child, 1u, 0u));
        let c_max_y = f32(load_child_q(off, child, 1u, 1u));
        let c_min_z = f32(load_child_q(off, child, 2u, 0u));
        let c_max_z = f32(load_child_q(off, child, 2u, 1u));

        // AABB-AABB overlap test
        let overlaps = local_min.x <= c_max_x && local_max.x >= c_min_x
                     && local_min.y <= c_max_y && local_max.y >= c_min_y
                     && local_min.z <= c_max_z && local_max.z >= c_min_z;

        if overlaps {
            let child_bits = (cmeta >> 5u) & 0x07u;
            var bit_index = cmeta & 0x1Fu;
            let is_inner = ((cmeta & 0x18u) == 0x18u);
            if is_inner {
                bit_index = bit_index ^ (oct_inv4 & 0xFFu);
            }
            hit_mask |= child_bits << bit_index;
        }
    }

    return hit_mask;
}

// ─── CWBVH AABB Traversal ───
// Traverses the BVH to find all primitives whose AABBs overlap the query AABB.
// Writes collision pairs to the output buffer.
fn traverse_cwbvh_aabb(query_id: u32, query: Aabb, max_pairs: u32) {
    // Use a fixed oct_inv4 for AABB queries (no directional ordering needed)
    let oct_inv4: u32 = 0u;

    var current_group = vec2<u32>(0u, 0x80000000u);
    var primitive_group = vec2<u32>(0u, 0u);
    var stack: array<vec2<u32>, 32>;
    var stack_ptr: u32 = 0u;
    var pairs_written: u32 = 0u;

    loop {
        // Process leaf primitives
        while primitive_group.y != 0u {
            let local_prim_idx = firstLeadingBit(primitive_group.y);
            primitive_group.y &= ~(1u << local_prim_idx);

            let global_prim_idx = primitive_group.x + local_prim_idx;
            let prim_id = primitive_indices[global_prim_idx];

            // Check triangle AABB vs query AABB for precise test
            let v0 = load_triangle_v0(prim_id);
            let v1 = load_triangle_v1(prim_id);
            let v2 = load_triangle_v2(prim_id);
            let tri_min = min(v0, min(v1, v2));
            let tri_max = max(v0, max(v1, v2));

            let overlaps = query.aabb_min.x <= tri_max.x && query.aabb_max.x >= tri_min.x
                         && query.aabb_min.y <= tri_max.y && query.aabb_max.y >= tri_min.y
                         && query.aabb_min.z <= tri_max.z && query.aabb_max.z >= tri_min.z;

            if overlaps && pairs_written < max_pairs {
                let idx = atomicAdd(&pair_count, 1u);
                collision_pairs[idx] = CollisionPair(query_id, prim_id);
                pairs_written += 1u;
            }
        }

        // Process internal nodes
        if (current_group.y & 0xFF000000u) != 0u {
            let hits_imask = current_group.y;
            let child_index_offset = firstLeadingBit(hits_imask);
            current_group.y &= ~(1u << child_index_offset);

            if (current_group.y & 0xFF000000u) != 0u {
                stack[stack_ptr] = current_group;
                stack_ptr += 1u;
            }

            let slot_index = (child_index_offset - 24u) ^ (oct_inv4 & 0xFFu);
            let imask = hits_imask & 0xFFu;
            let mask_below = imask & ((1u << slot_index) - 1u);
            let relative_index = countOneBits(mask_below);
            let child_node_index = current_group.x + relative_index;

            let child_off = node_offset(child_node_index);
            let child_imask = load_node_imask(child_off);
            let hitmask = intersect_aabb_node(child_off, query, oct_inv4);

            current_group = vec2<u32>(
                load_child_base_idx(child_off),
                (hitmask & 0xFF000000u) | child_imask
            );
            primitive_group = vec2<u32>(
                load_primitive_base_idx(child_off),
                hitmask & 0x00FFFFFFu
            );
        } else {
            primitive_group = current_group;
            current_group = vec2<u32>(0u, 0u);
        }

        // Stack
        if primitive_group.y == 0u && (current_group.y & 0xFF000000u) == 0u {
            if stack_ptr == 0u { break; }
            stack_ptr -= 1u;
            current_group = stack[stack_ptr];
        }
    }
}

// ─── Compute Entry Point ───
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let query_id = gid.x;
    if query_id >= collision_uniforms.query_count {
        return;
    }

    let base = query_id * 6u;
    var query: Aabb;
    query.aabb_min = vec3<f32>(query_aabbs[base + 0u], query_aabbs[base + 1u], query_aabbs[base + 2u]);
    query.aabb_max = vec3<f32>(query_aabbs[base + 3u], query_aabbs[base + 4u], query_aabbs[base + 5u]);

    traverse_cwbvh_aabb(query_id, query, collision_uniforms.max_pairs_per_query);
}
