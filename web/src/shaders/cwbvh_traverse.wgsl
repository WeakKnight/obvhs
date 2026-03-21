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

// ─── Ray-Node Intersection (optimized 2×4 batch path) ───
// Tests ray against all 8 quantized child AABBs of a CwBvhNode.
// Uses 2×4 structure: processes children in two batches of 4.
// Division by ray.direction is done directly (no precalculated inv_direction)
// to reduce register pressure (~2.5% faster per tray_racing measurements).
// Returns hit_mask: upper 8 bits = hit internal children, lower 24 bits = hit leaf primitives.
fn intersect_ray_node(off: u32, ray: Ray, oct_inv4: u32) -> u32 {
    let p = load_node_p(off);
    let e = load_node_e(off);
    let extent = compute_extent(e);

    // Direct division: reduces register pressure vs storing inv_direction
    let adjusted_ray_dir_inv = extent / ray.direction;
    let adjusted_ray_origin = (p - ray.origin) / ray.direction;

    let rdx = ray.direction.x < 0.0;
    let rdy = ray.direction.y < 0.0;
    let rdz = ray.direction.z < 0.0;

    var hit_mask: u32 = 0u;

    // Process 8 children in two batches of 4 (matching CWBVH packed layout)
    for (var i: u32 = 0u; i < 2u; i = i + 1u) {
        // Load packed meta for 4 children at once
        let meta4 = load_meta4(off, i);

        // Batch-compute inner mask and bit indices for all 4 children simultaneously
        // using packed byte arithmetic (same as tray_racing)
        let is_inner4 = (meta4 & (meta4 << 1u)) & 0x10101010u;
        let inner_mask4 = (is_inner4 >> 4u) * 0xFFu;
        let bit_index4 = (meta4 ^ (oct_inv4 & inner_mask4)) & 0x1F1F1F1Fu;
        let child_bits4 = (meta4 >> 5u) & 0x07070707u;

        // Load all quantized bounds for this half (4 children packed in each u32)
        let q_lo_x = load_q4(off, 0u, 0u, i);
        let q_hi_x = load_q4(off, 0u, 1u, i);
        let q_lo_y = load_q4(off, 1u, 0u, i);
        let q_hi_y = load_q4(off, 1u, 1u, i);
        let q_lo_z = load_q4(off, 2u, 0u, i);
        let q_hi_z = load_q4(off, 2u, 1u, i);

        // Select near/far planes based on ray direction sign
        let x_min = select(q_lo_x, q_hi_x, rdx);
        let x_max = select(q_hi_x, q_lo_x, rdx);
        let y_min = select(q_lo_y, q_hi_y, rdy);
        let y_max = select(q_hi_y, q_lo_y, rdy);
        let z_min = select(q_lo_z, q_hi_z, rdz);
        let z_max = select(q_hi_z, q_lo_z, rdz);

        // Test each of the 4 children
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            // Extract j-th byte from each packed u32
            var tmin3 = vec3<f32>(
                f32(extract_byte(x_min, j)),
                f32(extract_byte(y_min, j)),
                f32(extract_byte(z_min, j))
            );
            var tmax3 = vec3<f32>(
                f32(extract_byte(x_max, j)),
                f32(extract_byte(y_max, j)),
                f32(extract_byte(z_max, j))
            );

            // Apply quantized-space transform
            tmin3 = tmin3 * adjusted_ray_dir_inv + adjusted_ray_origin;
            tmax3 = tmax3 * adjusted_ray_dir_inv + adjusted_ray_origin;

            // Slab test
            let tmin_val = max(max(tmin3.x, max(tmin3.y, tmin3.z)), EPSILON);
            let tmax_val = min(min(tmax3.x, min(tmax3.y, tmax3.z)), ray.tmax);

            if tmin_val <= tmax_val {
                let child_bits = extract_byte(child_bits4, j);
                let bit_index = extract_byte(bit_index4, j);
                hit_mask |= child_bits << bit_index;
            }
        }
    }

    return hit_mask;
}

// ─── CWBVH Traversal: Closest Hit ───
// Traverses the BVH to find the closest triangle intersection.
// Loop order: node → triangle → stack (matches tray_racing for reduced loop overhead).
fn traverse_cwbvh_closest(ray_in: Ray) -> RayHit {
    var ray = ray_in;
    var hit: RayHit;
    hit.primitive_id = INVALID_U32;
    hit.t = ray.tmax;
    hit.instance_id = INVALID_U32;
    hit.is_procedural = 0u;

    let oct_inv4 = ray_get_octant_inv4(ray.direction);

    // Traversal state
    var current_group = vec2<u32>(0u, 0x80000000u); // Start at root
    var stack: array<vec2<u32>, 32>;
    var stack_ptr: u32 = 0u;

    loop {
        var triangle_group: vec2<u32>;

        // ── Phase 1: Process internal node group → produces triangle_group ──
        if (current_group.y & 0xFF000000u) != 0u {
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
            triangle_group = vec2<u32>(
                load_primitive_base_idx(child_off),
                hitmask & 0x00FFFFFFu
            );
        } else {
            // No internal children — triangle postponing for GPU
            triangle_group = current_group;
            current_group = vec2<u32>(0u, 0u);
        }

        // ── Phase 2: Process all pending primitives ──
        while triangle_group.y != 0u {
            let local_prim_idx = firstLeadingBit(triangle_group.y);
            triangle_group.y &= ~(1u << local_prim_idx);

            let global_prim_idx = triangle_group.x + local_prim_idx;
            let prim_id = primitive_indices[global_prim_idx];

            let t = intersect_triangle(ray, prim_id);
            if t < ray.tmax {
                hit.primitive_id = prim_id;
                hit.t = t;
                ray.tmax = t;
            }
        }

        // ── Phase 3: Stack management ──
        if (current_group.y & 0xFF000000u) == 0u {
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
// Loop order: node → triangle → stack.
fn traverse_cwbvh_any(ray_in: Ray) -> bool {
    var ray = ray_in;
    let oct_inv4 = ray_get_octant_inv4(ray.direction);

    var current_group = vec2<u32>(0u, 0x80000000u);
    var stack: array<vec2<u32>, 32>;
    var stack_ptr: u32 = 0u;

    loop {
        var triangle_group: vec2<u32>;

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
            triangle_group = vec2<u32>(
                load_primitive_base_idx(child_off),
                hitmask & 0x00FFFFFFu
            );
        } else {
            triangle_group = current_group;
            current_group = vec2<u32>(0u, 0u);
        }

        // Process triangles
        while triangle_group.y != 0u {
            let local_prim_idx = firstLeadingBit(triangle_group.y);
            triangle_group.y &= ~(1u << local_prim_idx);

            let global_prim_idx = triangle_group.x + local_prim_idx;
            let prim_id = primitive_indices[global_prim_idx];

            let t = intersect_triangle(ray, prim_id);
            if t < ray.tmax {
                return true; // Early exit on any hit
            }
        }

        // Stack
        if (current_group.y & 0xFF000000u) == 0u {
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
    hit.instance_id = INVALID_U32;
    hit.is_procedural = 0u;

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
    // Pending primitive group from restart (leaf hits at restart level)
    var pending_prim = vec2<u32>(0u, 0u);
    var level: u32 = 0u;

    loop {
        var triangle_group: vec2<u32>;

        // ── Phase 1: Process internal node group → produces triangle_group ──
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

            // Compute child node index
            let slot_index = (child_index_offset - 24u) ^ (oct_inv4 & 0xFFu);
            let imask = hits_imask & 0xFFu;
            let mask_below = imask & ((1u << slot_index) - 1u);
            let relative_index = countOneBits(mask_below);
            let child_node_index = current_group.x + relative_index;

            // Update trail
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
            triangle_group = vec2<u32>(
                load_primitive_base_idx(child_off),
                hitmask & 0x00FFFFFFu
            );
        } else {
            // No internal children — handle leaf primitives
            // Use pending_prim if available, otherwise use current_group (triangle postponing)
            if pending_prim.y != 0u {
                triangle_group = pending_prim;
                pending_prim = vec2<u32>(0u, 0u);
            } else {
                triangle_group = current_group;
            }
            current_group = vec2<u32>(0u, 0u);
        }

        // ── Phase 2: Process pending primitives ──
        while triangle_group.y != 0u {
            let idx = firstLeadingBit(triangle_group.y);
            triangle_group.y &= ~(1u << idx);

            let prim_id = primitive_indices[triangle_group.x + idx];
            let t = intersect_triangle(ray, prim_id);
            if t < ray.tmax {
                hit.primitive_id = prim_id;
                hit.t = t;
                ray.tmax = t;
            }
        }

        // ── Phase 3: Stack management ──
        if (current_group.y & 0xFF000000u) == 0u && pending_prim.y == 0u {
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
                pending_prim = vec2<u32>(restart.prim_base, restart.prim_hits);
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
    var pending_prim = vec2<u32>(0u, 0u);
    var level: u32 = 0u;

    loop {
        var triangle_group: vec2<u32>;

        // Process internal nodes
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
            triangle_group = vec2<u32>(
                load_primitive_base_idx(child_off),
                hitmask & 0x00FFFFFFu
            );
        } else {
            if pending_prim.y != 0u {
                triangle_group = pending_prim;
                pending_prim = vec2<u32>(0u, 0u);
            } else {
                triangle_group = current_group;
            }
            current_group = vec2<u32>(0u, 0u);
        }

        // Process triangles
        while triangle_group.y != 0u {
            let idx = firstLeadingBit(triangle_group.y);
            triangle_group.y &= ~(1u << idx);
            let prim_id = primitive_indices[triangle_group.x + idx];
            let t = intersect_triangle(ray, prim_id);
            if t < ray.tmax {
                return true;
            }
        }

        // Stack management
        if (current_group.y & 0xFF000000u) == 0u && pending_prim.y == 0u {
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
                pending_prim = vec2<u32>(restart.prim_base, restart.prim_hits);
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
