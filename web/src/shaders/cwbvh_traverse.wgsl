// ============================================================================
// cwbvh_traverse.wgsl — CWBVH ray traversal & triangle intersection
// Ported from obvhs (Rust) — intersect_ray_basic + traverse! macro logic
// ============================================================================

// NOTE: This file is meant to be concatenated after cwbvh_common.wgsl
// All common types, bindings and utility functions are defined there.

// ─── Moller-Trumbore Ray-Triangle Intersection ───
// Returns distance t on hit, or a value > ray.tmax on miss.
fn intersect_triangle(ray: Ray, prim_id: u32) -> f32 {
    let v0 = load_triangle_v0(prim_id);
    let v1 = load_triangle_v1(prim_id);
    let v2 = load_triangle_v2(prim_id);

    let e1 = v0 - v1;
    let e2 = v2 - v0;
    let n = cross(e1, e2);

    let c = v0 - ray.origin;
    let r = cross(ray.direction, c);
    let inv_det = 1.0 / dot(n, ray.direction);

    let u = dot(r, e2) * inv_det;
    let v = dot(r, e1) * inv_det;
    let w = 1.0 - u - v;

    // Check all barycentric coords >= 0 using sign bit trick
    let hit_bits = bitcast<u32>(u) | bitcast<u32>(v) | bitcast<u32>(w);
    let valid = (inv_det != 0.0) && ((hit_bits & 0x80000000u) == 0u);

    if valid {
        let t = dot(n, c) * inv_det;
        if t >= ray.tmin && t <= ray.tmax {
            return t;
        }
    }
    return 3.402823466e+38; // F32_MAX as miss sentinel
}

// ─── Ray-Node Intersection (basic / non-SIMD path) ───
// Tests ray against all 8 quantized child AABBs of a CwBvhNode.
// Returns hit_mask: upper 8 bits = hit internal children, lower 24 bits = hit leaf primitives.
fn intersect_ray_node(off: u32, ray: Ray, oct_inv4: u32) -> u32 {
    let p = load_node_p(off);
    let e = load_node_e(off);
    let extent = compute_extent(e);

    // Transform ray to node's quantized space
    let adjusted_ray_dir_inv = extent * ray.inv_direction;
    let adjusted_ray_origin = (p - ray.origin) * ray.inv_direction;

    let rdx = ray.direction.x < 0.0;
    let rdy = ray.direction.y < 0.0;
    let rdz = ray.direction.z < 0.0;

    var hit_mask: u32 = 0u;

    for (var child: u32 = 0u; child < 8u; child = child + 1u) {
        let cmeta = load_child_meta(off, child);
        if cmeta == 0u { continue; }

        // Load quantized AABB bounds
        let q_lo_x = load_child_q(off, child, 0u, 0u);
        let q_hi_x = load_child_q(off, child, 0u, 1u);
        let q_lo_y = load_child_q(off, child, 1u, 0u);
        let q_hi_y = load_child_q(off, child, 1u, 1u);
        let q_lo_z = load_child_q(off, child, 2u, 0u);
        let q_hi_z = load_child_q(off, child, 2u, 1u);

        // Swap min/max based on ray direction sign
        let x_min = select(q_lo_x, q_hi_x, rdx);
        let x_max = select(q_hi_x, q_lo_x, rdx);
        let y_min = select(q_lo_y, q_hi_y, rdy);
        let y_max = select(q_hi_y, q_lo_y, rdy);
        let z_min = select(q_lo_z, q_hi_z, rdz);
        let z_max = select(q_hi_z, q_lo_z, rdz);

        var tmin3 = vec3<f32>(f32(x_min), f32(y_min), f32(z_min));
        var tmax3 = vec3<f32>(f32(x_max), f32(y_max), f32(z_max));

        // Apply quantized-space transform
        tmin3 = tmin3 * adjusted_ray_dir_inv + adjusted_ray_origin;
        tmax3 = tmax3 * adjusted_ray_dir_inv + adjusted_ray_origin;

        // Slab test
        let tmin_val = max(max(tmin3.x, max(tmin3.y, tmin3.z)), EPSILON);
        let tmax_val = min(min(tmax3.x, min(tmax3.y, tmax3.z)), ray.tmax);

        if tmin_val <= tmax_val {
            // Decode child_meta bits
            // Inner node: high 3 bits = 0b001, low 5 bits = slot_index + 24
            // Leaf node: high 3 bits = unary prim count, low 5 bits = prim offset
            let child_bits = (cmeta >> 5u) & 0x07u;
            var bit_index = cmeta & 0x1Fu;

            // For inner nodes, XOR with oct_inv4 to get traversal order
            let is_inner = ((cmeta & 0x18u) == 0x18u); // bits 3&4 both set = inner
            if is_inner {
                bit_index = bit_index ^ (oct_inv4 & 0xFFu);
            }

            hit_mask |= child_bits << bit_index;
        }
    }

    return hit_mask;
}

// ─── CWBVH Traversal: Closest Hit ───
// Traverses the BVH to find the closest triangle intersection.
// Returns updated RayHit with primitive_id and t.
fn traverse_cwbvh_closest(ray_in: Ray) -> RayHit {
    var ray = ray_in;
    var hit: RayHit;
    hit.primitive_id = INVALID_U32;
    hit.t = ray.tmax;

    let oct_inv4 = ray_get_octant_inv4(ray.direction);

    // Traversal state
    // current_group.x = child_base_idx, current_group.y = internal hit mask | imask
    var current_group = vec2<u32>(0u, 0x80000000u); // Start at root
    var primitive_group = vec2<u32>(0u, 0u);

    // Stack
    var stack: array<vec2<u32>, 32>;
    var stack_ptr: u32 = 0u;

    loop {
        // ── Phase 1: Process all pending primitives ──
        while primitive_group.y != 0u {
            let local_prim_idx = firstLeadingBit(primitive_group.y);
            primitive_group.y &= ~(1u << local_prim_idx);

            let global_prim_idx = primitive_group.x + local_prim_idx;
            let prim_id = primitive_indices[global_prim_idx];

            let t = intersect_triangle(ray, prim_id);
            if t < ray.tmax {
                hit.primitive_id = prim_id;
                hit.t = t;
                ray.tmax = t;
            }
        }

        // ── Phase 2: Process internal node group ──
        if (current_group.y & 0xFF000000u) != 0u {
            // There are internal children to process
            let hits_imask = current_group.y;
            let child_index_offset = firstLeadingBit(hits_imask);

            // Remove this child from the group
            current_group.y &= ~(1u << child_index_offset);

            // If more internal children remain, push to stack
            if (current_group.y & 0xFF000000u) != 0u {
                stack[stack_ptr] = current_group;
                stack_ptr += 1u;
            }

            // Compute actual child node index
            let slot_index = (child_index_offset - 24u) ^ (oct_inv4 & 0xFFu);
            let imask = hits_imask & 0xFFu;
            // Count set bits below slot_index in imask to get relative offset
            let mask_below = imask & ((1u << slot_index) - 1u);
            let relative_index = countOneBits(mask_below);
            let child_node_index = current_group.x + relative_index;

            // Read child node and intersect
            let child_off = node_offset(child_node_index);
            let child_imask = load_node_imask(child_off);
            let hitmask = intersect_ray_node(child_off, ray, oct_inv4);

            // Split hitmask into internal (upper 8 bits) and leaf (lower 24 bits)
            current_group = vec2<u32>(
                load_child_base_idx(child_off),
                (hitmask & 0xFF000000u) | child_imask
            );
            primitive_group = vec2<u32>(
                load_primitive_base_idx(child_off),
                hitmask & 0x00FFFFFFu
            );
        } else {
            // No internal children — triangle postponing for GPU
            // Move any remaining leaf bits from current_group to primitive_group
            primitive_group = current_group;
            current_group = vec2<u32>(0u, 0u);
        }

        // ── Phase 3: Stack management ──
        if primitive_group.y == 0u && (current_group.y & 0xFF000000u) == 0u {
            if stack_ptr == 0u {
                break; // Traversal complete
            }
            stack_ptr -= 1u;
            current_group = stack[stack_ptr];
        }
    }

    return hit;
}

// ─── CWBVH Traversal: Any Hit (shadow ray) ───
// Returns true if any intersection is found.
fn traverse_cwbvh_any(ray_in: Ray) -> bool {
    var ray = ray_in;
    let oct_inv4 = ray_get_octant_inv4(ray.direction);

    var current_group = vec2<u32>(0u, 0x80000000u);
    var primitive_group = vec2<u32>(0u, 0u);
    var stack: array<vec2<u32>, 32>;
    var stack_ptr: u32 = 0u;

    loop {
        // Process primitives
        while primitive_group.y != 0u {
            let local_prim_idx = firstLeadingBit(primitive_group.y);
            primitive_group.y &= ~(1u << local_prim_idx);

            let global_prim_idx = primitive_group.x + local_prim_idx;
            let prim_id = primitive_indices[global_prim_idx];

            let t = intersect_triangle(ray, prim_id);
            if t < ray.tmax {
                return true; // Early exit on any hit
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
            let hitmask = intersect_ray_node(child_off, ray, oct_inv4);

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
            if stack_ptr == 0u {
                break;
            }
            stack_ptr -= 1u;
            current_group = stack[stack_ptr];
        }
    }

    return false;
}

// ─── Short Stack Traversal ───
// Based on "Wide BVH Traversal with a Short Stack" (Vaidyanathan et al., HPG 2019)
//
// Identical traversal logic to the full-stack version, but the stack is
// limited to SHORT_STACK_SIZE entries using a circular buffer. When the
// stack is full, the oldest (bottom) entry is silently overwritten.
//
// When the stack underflows on pop, we restart the entire traversal from
// the root. For closest-hit this is efficient because ray.tmax has been
// tightened by previous hits, causing most subtrees to be culled on the
// re-traversal. For any-hit, we typically find a hit before overflow.
//
// To guarantee termination and correctness even when entries are lost,
// we maintain a "restart trail": a per-level counter tracking how many
// internal children have been descended into. On restart we walk from the
// root, using the trail to skip already-processed subtrees and find the
// next unvisited one.

// ─── CWBVH Traversal: Closest Hit (Short Stack) ───
fn traverse_cwbvh_closest_short_stack(ray_in: Ray) -> RayHit {
    var ray = ray_in;
    var hit: RayHit;
    hit.primitive_id = INVALID_U32;
    hit.t = ray.tmax;

    let oct_inv4 = ray_get_octant_inv4(ray.direction);

    // Restart trail: trail[level] = # of internal children already descended
    // into at that tree depth. Packed as 4 bits per level in 4 × u32.
    var trail: array<u32, TRAIL_ARRAY_SIZE>;
    init_trail(&trail);

    // Circular-buffer short stack
    var stack: array<vec2<u32>, SHORT_STACK_SIZE>;
    var stack_top: u32 = 0u;
    var stack_bot: u32 = 0u;

    var current_group = vec2<u32>(0u, 0x80000000u); // sentinel: root
    var primitive_group = vec2<u32>(0u, 0u);
    var level: u32 = 0u;

    loop {
        // ── Phase 1: Process pending primitives ──
        while primitive_group.y != 0u {
            let idx = firstLeadingBit(primitive_group.y);
            primitive_group.y &= ~(1u << idx);

            let prim_id = primitive_indices[primitive_group.x + idx];
            let t = intersect_triangle(ray, prim_id);
            if t < ray.tmax {
                hit.primitive_id = prim_id;
                hit.t = t;
                ray.tmax = t;
            }
        }

        // ── Phase 2: Process internal node group ──
        if (current_group.y & 0xFF000000u) != 0u {
            let hits_imask = current_group.y;
            let child_index_offset = firstLeadingBit(hits_imask);

            // Remove this child from the group
            current_group.y &= ~(1u << child_index_offset);

            // Push remaining siblings to short stack (if any)
            if (current_group.y & 0xFF000000u) != 0u {
                stack[stack_top % SHORT_STACK_SIZE] = current_group;
                stack_top += 1u;
                if stack_top - stack_bot > SHORT_STACK_SIZE {
                    stack_bot = stack_top - SHORT_STACK_SIZE;
                }
            }

            // Compute child node index (identical to full-stack)
            let slot_index = (child_index_offset - 24u) ^ (oct_inv4 & 0xFFu);
            let imask = hits_imask & 0xFFu;
            let mask_below = imask & ((1u << slot_index) - 1u);
            let relative_index = countOneBits(mask_below);
            let child_node_index = current_group.x + relative_index;

            // Update trail: record that we descended one more child at this level
            trail_increment(&trail, level);
            level += 1u;

            // Read child node and intersect
            let child_off = node_offset(child_node_index);
            let child_imask = load_node_imask(child_off);
            let hitmask = intersect_ray_node(child_off, ray, oct_inv4);

            current_group = vec2<u32>(
                load_child_base_idx(child_off),
                (hitmask & 0xFF000000u) | child_imask
            );
            primitive_group = vec2<u32>(
                load_primitive_base_idx(child_off),
                hitmask & 0x00FFFFFFu
            );
        } else {
            // No internal children — handle leaf primitives
            primitive_group = current_group;
            current_group = vec2<u32>(0u, 0u);
        }

        // ── Phase 3: Stack management ──
        if primitive_group.y == 0u && (current_group.y & 0xFF000000u) == 0u {
            if stack_top > stack_bot {
                // Pop from circular stack
                stack_top -= 1u;
                current_group = stack[stack_top % SHORT_STACK_SIZE];
                if level > 0u { level -= 1u; }
            } else {
                // Stack empty — use trail to restart from root
                let restart = restart_from_trail(&trail, ray, oct_inv4);
                if restart.found == 0u {
                    break; // Traversal complete
                }
                level = restart.level;
                current_group = vec2<u32>(restart.child_base, restart.group_y);
                primitive_group = vec2<u32>(restart.prim_base, restart.prim_hits);
                // Reset stack
                stack_top = 0u;
                stack_bot = 0u;
            }
        }
    }

    return hit;
}

// ─── CWBVH Traversal: Any Hit (Short Stack) ───
fn traverse_cwbvh_any_short_stack(ray_in: Ray) -> bool {
    var ray = ray_in;
    let oct_inv4 = ray_get_octant_inv4(ray.direction);

    var trail: array<u32, TRAIL_ARRAY_SIZE>;
    init_trail(&trail);

    var stack: array<vec2<u32>, SHORT_STACK_SIZE>;
    var stack_top: u32 = 0u;
    var stack_bot: u32 = 0u;

    var current_group = vec2<u32>(0u, 0x80000000u);
    var primitive_group = vec2<u32>(0u, 0u);
    var level: u32 = 0u;

    loop {
        while primitive_group.y != 0u {
            let idx = firstLeadingBit(primitive_group.y);
            primitive_group.y &= ~(1u << idx);
            let prim_id = primitive_indices[primitive_group.x + idx];
            let t = intersect_triangle(ray, prim_id);
            if t < ray.tmax {
                return true;
            }
        }

        if (current_group.y & 0xFF000000u) != 0u {
            let hits_imask = current_group.y;
            let child_index_offset = firstLeadingBit(hits_imask);
            current_group.y &= ~(1u << child_index_offset);

            if (current_group.y & 0xFF000000u) != 0u {
                stack[stack_top % SHORT_STACK_SIZE] = current_group;
                stack_top += 1u;
                if stack_top - stack_bot > SHORT_STACK_SIZE {
                    stack_bot = stack_top - SHORT_STACK_SIZE;
                }
            }

            let slot_index = (child_index_offset - 24u) ^ (oct_inv4 & 0xFFu);
            let imask = hits_imask & 0xFFu;
            let mask_below = imask & ((1u << slot_index) - 1u);
            let relative_index = countOneBits(mask_below);
            let child_node_index = current_group.x + relative_index;

            trail_increment(&trail, level);
            level += 1u;

            let child_off = node_offset(child_node_index);
            let child_imask = load_node_imask(child_off);
            let hitmask = intersect_ray_node(child_off, ray, oct_inv4);

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

        if primitive_group.y == 0u && (current_group.y & 0xFF000000u) == 0u {
            if stack_top > stack_bot {
                stack_top -= 1u;
                current_group = stack[stack_top % SHORT_STACK_SIZE];
                if level > 0u { level -= 1u; }
            } else {
                let restart = restart_from_trail(&trail, ray, oct_inv4);
                if restart.found == 0u {
                    break;
                }
                level = restart.level;
                current_group = vec2<u32>(restart.child_base, restart.group_y);
                primitive_group = vec2<u32>(restart.prim_base, restart.prim_hits);
                stack_top = 0u;
                stack_bot = 0u;
            }
        }
    }

    return false;
}

// ─── Restart result struct ───
struct RestartResult {
    found: u32,      // 1 = found unvisited subtree, 0 = done
    level: u32,      // tree level of the found node
    child_base: u32, // child_base_idx of the found node
    group_y: u32,    // (remaining internal hits) | imask
    prim_base: u32,  // primitive_base_idx of the found node
    prim_hits: u32,  // leaf hit bits for the found node
};

// ─── Restart from trail ───
// Walk from root down the tree, following the trail to skip already-processed
// subtrees, until we find an internal node with unvisited children.
fn restart_from_trail(
    trail_ptr: ptr<function, array<u32, TRAIL_ARRAY_SIZE>>,
    ray: Ray,
    oct_inv4: u32
) -> RestartResult {
    var result: RestartResult;
    result.found = 0u;

    var cur_node_idx: u32 = 0u; // root = node index 0
    var cur_level: u32 = 0u;

    for (var iter: u32 = 0u; iter < MAX_BVH_DEPTH; iter = iter + 1u) {
        let off = node_offset(cur_node_idx);
        let hitmask = intersect_ray_node(off, ray, oct_inv4);
        let imask_val = load_node_imask(off);
        let child_base = load_child_base_idx(off);
        let prim_base = load_primitive_base_idx(off);

        let internal_hits = hitmask & 0xFF000000u;
        let prim_hits = hitmask & 0x00FFFFFFu;
        let n_internal_hits = countOneBits(internal_hits);
        let already_done = trail_get(*trail_ptr, cur_level);

        if already_done >= n_internal_hits {
            // All internal children at this level are done.
            // Mark this level as fully complete and search upward.
            trail_set(trail_ptr, cur_level, BRANCHING);

            // Walk backward to find a level with remaining work
            if cur_level == 0u {
                // Root is fully done — traversal complete
                return result;
            }

            // We can't easily walk up, so restart the whole descent
            // from root with updated trail. To avoid infinite loop,
            // we just return "not found" and let the outer loop handle it.
            // Actually, let's do a proper backward search and re-descent.
            var found_level = INVALID_U32;
            for (var l: i32 = i32(cur_level) - 1; l >= 0; l = l - 1) {
                if trail_get(*trail_ptr, u32(l)) < BRANCHING {
                    found_level = u32(l);
                    break;
                }
            }

            if found_level == INVALID_U32 {
                return result; // fully done
            }

            // Clear trail below found_level
            for (var c: u32 = found_level + 1u; c <= cur_level; c = c + 1u) {
                trail_set(trail_ptr, c, 0u);
            }

            // Re-descend from root to found_level
            cur_node_idx = 0u;
            for (var d: u32 = 0u; d < found_level; d = d + 1u) {
                let d_off = node_offset(cur_node_idx);
                let d_hitmask = intersect_ray_node(d_off, ray, oct_inv4);
                let d_internal = d_hitmask & 0xFF000000u;
                let d_imask = load_node_imask(d_off);
                let d_child_base = load_child_base_idx(d_off);
                let d_done = trail_get(*trail_ptr, d);

                // We need to find the (d_done)th internal child (0-indexed)
                // that was the *last* one descended into. That's child #(d_done - 1)
                // in traversal order.
                if d_done == 0u {
                    return result; // shouldn't happen
                }

                var d_remaining = d_internal;
                // Skip first (d_done - 1) hit bits
                for (var s: u32 = 0u; s < d_done - 1u; s = s + 1u) {
                    if d_remaining == 0u { return result; }
                    let bit = firstLeadingBit(d_remaining);
                    d_remaining &= ~(1u << bit);
                }
                if d_remaining == 0u { return result; }

                let child_bit = firstLeadingBit(d_remaining);
                let slot_idx = (child_bit - 24u) ^ (oct_inv4 & 0xFFu);
                let below = d_imask & ((1u << slot_idx) - 1u);
                cur_node_idx = d_child_base + countOneBits(below);
            }

            cur_level = found_level;
            continue; // re-intersect at found_level
        }

        // There are unvisited internal children at this level.
        // Skip the first `already_done` internal hit bits.
        var remaining = internal_hits;
        for (var s: u32 = 0u; s < already_done; s = s + 1u) {
            if remaining == 0u { break; }
            let bit = firstLeadingBit(remaining);
            remaining &= ~(1u << bit);
        }

        result.found = 1u;
        result.level = cur_level;
        result.child_base = child_base;
        result.group_y = remaining | imask_val;
        result.prim_base = prim_base;
        result.prim_hits = prim_hits;
        return result;
    }

    return result;
}
