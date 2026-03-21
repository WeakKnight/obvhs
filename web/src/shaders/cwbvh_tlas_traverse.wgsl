// ============================================================================
// cwbvh_tlas_traverse.wgsl — TLAS/BLAS two-level CWBVH traversal
// Ported from tray_racing rt_gpu_software_query_tlas.hlsl
//
// Design alignment with tray_racing:
//   - Shared stack with tlas_stack_size sentinel (INVALID = in TLAS)
//   - current_bvh_offset tracks which BVH we're in (TLAS or a BLAS)
//   - blas_offsets[] maps TLAS primitive index → BLAS node offset
//   - Same traversal loop structure: node → triangle/TLAS-leaf → stack
// ============================================================================

// NOTE: This file is meant to be concatenated after cwbvh_common.wgsl
// and cwbvh_traverse.wgsl (for intersect_triangle and intersect_ray_node).
// Requires: camera.tlas_start uniform.

// BLAS offsets: maps TLAS primitive index → node offset of that BLAS instance
// Aligns with tray_racing's blas_offsets / INSTANCES_BINDING
@group(0) @binding(3) var<storage, read> blas_offsets: array<u32>;

// ─── TLAS/BLAS Traversal: Closest Hit ───
// Aligns with tray_racing traverse_bvh() in rt_gpu_software_query_tlas.hlsl
fn traverse_tlas_closest(ray_in: Ray, tlas_start_offset: u32) -> RayHit {
    var ray = ray_in;
    var hit: RayHit;
    hit.primitive_id = INVALID_U32;
    hit.t = ray.tmax;

    let oct_inv4 = ray_get_octant_inv4(ray.direction);

    // Traversal state — matches tray_racing
    var current_group = vec2<u32>(0u, 0x80000000u); // Start at root
    var stack: array<vec2<u32>, TRAVERSAL_STACK_SIZE>;
    var stack_size: u32 = 0u;

    // TLAS/BLAS switching state (tray_racing sentinel pattern)
    var tlas_stack_size: u32 = INVALID_U32;  // INVALID = currently in TLAS
    var current_bvh_offset: u32 = tlas_start_offset; // Start in TLAS

    loop {
        var triangle_group: vec2<u32>;

        // ── Phase 1: Process internal node group ──
        if (current_group.y & 0xFF000000u) != 0u {
            let hits_imask = current_group.y;
            let child_index_offset = firstLeadingBit(hits_imask);
            let child_index_base = current_group.x;

            // Remove node from current_group
            current_group.y &= ~(1u << child_index_offset);

            // If the node group is not yet empty, push it on the stack
            if (current_group.y & 0xFF000000u) != 0u {
                stack[stack_size] = current_group;
                stack_size += 1u;
            }

            let slot_index = (child_index_offset - 24u) ^ (oct_inv4 & 0xFFu);
            let imask = hits_imask & 0xFFu;
            let mask_below = imask & ((1u << slot_index) - 1u);
            let relative_index = countOneBits(mask_below);

            let child_node_index = child_index_base + relative_index;

            // ★ Key difference from single-BVH: offset node access by current_bvh_offset
            let child_off = node_offset(current_bvh_offset + child_node_index);
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
            // No internal children left
            triangle_group = current_group;
            current_group = vec2<u32>(0u, 0u);
        }

        // ── Phase 2: Process leaf primitives ──
        while triangle_group.y != 0u {

            // ★ TLAS leaf handling (tlas_stack_size == INVALID means we're in TLAS)
            // Aligns with tray_racing: "If tlas_stack_size is INVALID we are in the TLAS"
            if tlas_stack_size == INVALID_U32 {
                let local_triangle_index = firstLeadingBit(triangle_group.y);

                // Remove from current group
                triangle_group.y &= ~(1u << local_triangle_index);

                // Get the BLAS instance index (via primitive_indices remapping)
                let global_triangle_index = triangle_group.x + local_triangle_index;

                // Push remaining TLAS leaf hits and current node group onto stack
                if triangle_group.y != 0u {
                    stack[stack_size] = triangle_group;
                    stack_size += 1u;
                }

                if (current_group.y & 0xFF000000u) != 0u {
                    stack[stack_size] = current_group;
                    stack_size += 1u;
                }

                // Record stack depth to detect when BLAS traversal is done
                tlas_stack_size = stack_size;

                // TODO: transform ray according to the mesh transform
                // (same TODO as tray_racing)

                // Look up BLAS node offset via blas_offsets
                // Aligns with tray_racing: current_bvh_offset = blas_offsets[global_triangle_index]
                current_bvh_offset = blas_offsets[global_triangle_index];

                // Start BLAS traversal from its root node (index 0)
                current_group = vec2<u32>(0u, 0x80000000u);

                break; // Exit triangle loop, continue main loop in BLAS
            } else {
                // ★ BLAS leaf: normal triangle intersection
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
        }

        // ── Phase 3: Stack management ──
        if (current_group.y & 0xFF000000u) == 0u {
            if stack_size == 0u {
                break; // Traversal complete
            }

            // ★ Check if we've finished BLAS traversal (tray_racing sentinel check)
            // "The value of tlas_stack_size is used to determine when traversal of a
            //  BLAS is finished, and we should revert back to TLAS traversal."
            if stack_size == tlas_stack_size {
                tlas_stack_size = INVALID_U32;
                current_bvh_offset = tlas_start_offset;
                // TODO: Reset Ray to untransformed version
            }

            stack_size -= 1u;
            current_group = stack[stack_size];
        }
    }

    return hit;
}

// ─── TLAS/BLAS Traversal: Any Hit (shadow ray) ───
// Aligns with tray_racing traverse_bvh() but returns on first hit.
fn traverse_tlas_any(ray_in: Ray, tlas_start_offset: u32) -> bool {
    var ray = ray_in;
    let oct_inv4 = ray_get_octant_inv4(ray.direction);

    var current_group = vec2<u32>(0u, 0x80000000u);
    var stack: array<vec2<u32>, TRAVERSAL_STACK_SIZE>;
    var stack_size: u32 = 0u;

    var tlas_stack_size: u32 = INVALID_U32;
    var current_bvh_offset: u32 = tlas_start_offset;

    loop {
        var triangle_group: vec2<u32>;

        // Process internal nodes
        if (current_group.y & 0xFF000000u) != 0u {
            let hits_imask = current_group.y;
            let child_index_offset = firstLeadingBit(hits_imask);
            current_group.y &= ~(1u << child_index_offset);

            if (current_group.y & 0xFF000000u) != 0u {
                stack[stack_size] = current_group;
                stack_size += 1u;
            }

            let slot_index = (child_index_offset - 24u) ^ (oct_inv4 & 0xFFu);
            let imask = hits_imask & 0xFFu;
            let mask_below = imask & ((1u << slot_index) - 1u);
            let relative_index = countOneBits(mask_below);
            let child_node_index = current_group.x + relative_index;

            let child_off = node_offset(current_bvh_offset + child_node_index);
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

        // Process leaves
        while triangle_group.y != 0u {
            if tlas_stack_size == INVALID_U32 {
                // TLAS leaf: switch to BLAS
                let local_triangle_index = firstLeadingBit(triangle_group.y);
                triangle_group.y &= ~(1u << local_triangle_index);
                let global_triangle_index = triangle_group.x + local_triangle_index;

                if triangle_group.y != 0u {
                    stack[stack_size] = triangle_group;
                    stack_size += 1u;
                }
                if (current_group.y & 0xFF000000u) != 0u {
                    stack[stack_size] = current_group;
                    stack_size += 1u;
                }

                tlas_stack_size = stack_size;
                current_bvh_offset = blas_offsets[global_triangle_index];
                current_group = vec2<u32>(0u, 0x80000000u);
                break;
            } else {
                // BLAS leaf: triangle intersection
                let local_prim_idx = firstLeadingBit(triangle_group.y);
                triangle_group.y &= ~(1u << local_prim_idx);

                let global_prim_idx = triangle_group.x + local_prim_idx;
                let prim_id = primitive_indices[global_prim_idx];

                let t = intersect_triangle(ray, prim_id);
                if t < ray.tmax {
                    return true; // Early exit on any hit
                }
            }
        }

        // Stack management
        if (current_group.y & 0xFF000000u) == 0u {
            if stack_size == 0u {
                break;
            }

            if stack_size == tlas_stack_size {
                tlas_stack_size = INVALID_U32;
                current_bvh_offset = tlas_start_offset;
            }

            stack_size -= 1u;
            current_group = stack[stack_size];
        }
    }

    return false;
}
