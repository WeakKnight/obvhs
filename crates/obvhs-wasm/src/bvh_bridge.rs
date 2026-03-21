use std::time::Duration;

use glam::Vec3A;
use obvhs::aabb::Aabb;
use obvhs::bvh2::builder::build_bvh2_from_tris;
use obvhs::bvh2::insertion_removal::SiblingInsertionCandidate;
use obvhs::bvh2::reinsertion::ReinsertionOptimizer;
use obvhs::bvh2::Bvh2;
use obvhs::cwbvh::bvh2_to_cwbvh::bvh2_to_cwbvh;
use obvhs::cwbvh::builder::build_cwbvh_from_tris;
use obvhs::cwbvh::CwBvh;
use obvhs::faststack::HeapStack;
use obvhs::ploc::{PlocBuilder, PlocSearchDistance, SortPrecision};
use obvhs::ploc::rebuild::compute_rebuild_path_flags;
use obvhs::triangle::Triangle;
use obvhs::BvhBuildParams;
use wasm_bindgen::prelude::*;

/// JS-friendly build quality enum.
#[wasm_bindgen]
#[derive(Clone, Copy, Default)]
pub enum BuildQuality {
    Fastest,
    Fast,
    #[default]
    Medium,
    Slow,
    VerySlow,
}

impl BuildQuality {
    fn to_params(self) -> BvhBuildParams {
        match self {
            BuildQuality::Fastest => BvhBuildParams::fastest_build(),
            BuildQuality::Fast => BvhBuildParams::fast_build(),
            BuildQuality::Medium => BvhBuildParams::medium_build(),
            BuildQuality::Slow => BvhBuildParams::slow_build(),
            BuildQuality::VerySlow => BvhBuildParams::very_slow_build(),
        }
    }
}

/// Parse a flat f32 array into a Vec<Triangle>.
fn parse_triangles(flat: &[f32]) -> Vec<Triangle> {
    let tri_count = flat.len() / 9;
    let mut triangles = Vec::with_capacity(tri_count);
    for i in 0..tri_count {
        let b = i * 9;
        triangles.push(Triangle {
            v0: Vec3A::new(flat[b], flat[b + 1], flat[b + 2]),
            v1: Vec3A::new(flat[b + 3], flat[b + 4], flat[b + 5]),
            v2: Vec3A::new(flat[b + 6], flat[b + 7], flat[b + 8]),
        });
    }
    triangles
}

/// The main WASM-exported BVH handle.
/// Holds both the CwBvh (for GPU upload) and optionally a Bvh2 (for dynamic updates).
#[wasm_bindgen]
pub struct WasmCwBvh {
    cwbvh: CwBvh,
    bvh2: Option<Bvh2>,
    triangles: Vec<Triangle>,
    config: BvhBuildParams,
    build_time_ms: f64,
    // Dynamic BVH helpers (lazily initialized)
    insertion_stack: HeapStack<SiblingInsertionCandidate>,
    reinsertion_optimizer: ReinsertionOptimizer,
    ploc_builder: Option<PlocBuilder>,
    temp_flags: Vec<bool>,
    temp_indices: Vec<u32>,
}

#[wasm_bindgen]
impl WasmCwBvh {
    /// Build a CWBVH from a flat f32 array of triangle vertices.
    /// `triangles_flat` must have length divisible by 9 (3 vertices × 3 floats).
    /// `quality` selects a build quality preset.
    /// `keep_bvh2` — if true, retains the intermediate BVH2 for dynamic updates.
    #[wasm_bindgen(constructor)]
    pub fn new(
        triangles_flat: &[f32],
        quality: BuildQuality,
        keep_bvh2: bool,
    ) -> Result<WasmCwBvh, JsError> {
        if triangles_flat.len() % 9 != 0 {
            return Err(JsError::new(
                "triangles_flat length must be divisible by 9",
            ));
        }

        let triangles = parse_triangles(triangles_flat);
        let mut config = quality.to_params();
        let mut build_time = Duration::default();

        if keep_bvh2 {
            // bvh2_to_cwbvh requires each BVH2 leaf to contain exactly 1 primitive.
            // build_bvh2_from_tris applies collapse() which can merge multiple prims
            // per leaf when max_prims_per_leaf > 1, so force it to 1 here.
            config.max_prims_per_leaf = 1;

            // Build BVH2 first, then convert to CWBVH — keep BVH2 for dynamic ops
            let bvh2 = build_bvh2_from_tris(&triangles, config, &mut build_time);
            let cwbvh = bvh2_to_cwbvh(&bvh2, 1, true, false);
            Ok(WasmCwBvh {
                cwbvh,
                bvh2: Some(bvh2),
                triangles,
                config,
                build_time_ms: build_time.as_secs_f64() * 1000.0,
                insertion_stack: HeapStack::new_with_capacity(1024),
                reinsertion_optimizer: ReinsertionOptimizer::default(),
                ploc_builder: None,
                temp_flags: Vec::new(),
                temp_indices: Vec::new(),
            })
        } else {
            let cwbvh = build_cwbvh_from_tris(&triangles, config, &mut build_time);
            Ok(WasmCwBvh {
                cwbvh,
                bvh2: None,
                triangles,
                config,
                build_time_ms: build_time.as_secs_f64() * 1000.0,
                insertion_stack: HeapStack::new_with_capacity(64),
                reinsertion_optimizer: ReinsertionOptimizer::default(),
                ploc_builder: None,
                temp_flags: Vec::new(),
                temp_indices: Vec::new(),
            })
        }
    }

    /// Full rebuild of the CWBVH from the current triangle set.
    pub fn rebuild(&mut self, quality: BuildQuality) {
        let mut config = quality.to_params();
        let mut build_time = Duration::default();

        if self.bvh2.is_some() {
            // Force 1 prim per leaf so bvh2_to_cwbvh won't panic
            config.max_prims_per_leaf = 1;
            let bvh2 = build_bvh2_from_tris(&self.triangles, config, &mut build_time);
            self.cwbvh = bvh2_to_cwbvh(&bvh2, 1, true, false);
            self.bvh2 = Some(bvh2);
        } else {
            self.cwbvh = build_cwbvh_from_tris(&self.triangles, config, &mut build_time);
        }
        self.config = config;
        self.build_time_ms = build_time.as_secs_f64() * 1000.0;
    }

    /// Convert the current BVH2 to CWBVH (after dynamic updates).
    /// Only valid if keep_bvh2 was true during construction.
    pub fn convert_bvh2_to_cwbvh(&mut self) -> Result<(), JsError> {
        let bvh2 = self
            .bvh2
            .as_ref()
            .ok_or_else(|| JsError::new("No BVH2 available — was keep_bvh2 true?"))?;
        self.cwbvh = bvh2_to_cwbvh(
            bvh2,
            1,
            true,
            false,
        );
        Ok(())
    }

    // ── Accessors for zero-copy GPU upload ──

    /// Pointer to the CwBvhNode array in WASM linear memory.
    pub fn nodes_ptr(&self) -> *const u8 {
        self.cwbvh.nodes.as_ptr() as *const u8
    }

    /// Byte length of the CwBvhNode array (nodes.len() * 80).
    pub fn nodes_byte_len(&self) -> usize {
        self.cwbvh.nodes.len() * std::mem::size_of::<obvhs::cwbvh::node::CwBvhNode>()
    }

    /// Number of CwBvh nodes.
    pub fn nodes_count(&self) -> usize {
        self.cwbvh.nodes.len()
    }

    /// Pointer to the primitive_indices array.
    pub fn indices_ptr(&self) -> *const u8 {
        self.cwbvh.primitive_indices.as_ptr() as *const u8
    }

    /// Byte length of primitive_indices (len * 4).
    pub fn indices_byte_len(&self) -> usize {
        self.cwbvh.primitive_indices.len() * 4
    }

    /// Number of primitive indices.
    pub fn indices_count(&self) -> usize {
        self.cwbvh.primitive_indices.len()
    }

    /// Pointer to the Triangle array (for GPU upload).
    pub fn triangles_ptr(&self) -> *const u8 {
        self.triangles.as_ptr() as *const u8
    }

    /// Byte length of the triangles array.
    /// Each Triangle is 48 bytes (3 × Vec3A at 16 bytes each).
    pub fn triangles_byte_len(&self) -> usize {
        self.triangles.len() * std::mem::size_of::<Triangle>()
    }

    /// Number of triangles.
    pub fn triangles_count(&self) -> usize {
        self.triangles.len()
    }

    /// Total AABB of the BVH (6 floats: min_x, min_y, min_z, max_x, max_y, max_z).
    pub fn total_aabb(&self) -> Vec<f32> {
        let aabb = &self.cwbvh.total_aabb;
        vec![
            aabb.min.x, aabb.min.y, aabb.min.z, aabb.max.x, aabb.max.y, aabb.max.z,
        ]
    }

    /// Build time in milliseconds.
    pub fn build_time_ms(&self) -> f64 {
        self.build_time_ms
    }

    // ── Dynamic BVH Operations (require keep_bvh2 = true) ──

    /// Insert a new triangle into the BVH.
    /// Returns the primitive_id of the inserted triangle.
    /// After inserting, call `convert_bvh2_to_cwbvh()` to update GPU data.
    pub fn insert_triangle(
        &mut self,
        v0x: f32, v0y: f32, v0z: f32,
        v1x: f32, v1y: f32, v1z: f32,
        v2x: f32, v2y: f32, v2z: f32,
    ) -> Result<u32, JsError> {
        let bvh2 = self.bvh2.as_mut()
            .ok_or_else(|| JsError::new("No BVH2 — construct with keep_bvh2 = true"))?;

        let tri = Triangle {
            v0: Vec3A::new(v0x, v0y, v0z),
            v1: Vec3A::new(v1x, v1y, v1z),
            v2: Vec3A::new(v2x, v2y, v2z),
        };
        let prim_id = self.triangles.len() as u32;
        self.triangles.push(tri);

        let aabb = tri.aabb();
        bvh2.insert_primitive(aabb, prim_id, &mut self.insertion_stack);
        Ok(prim_id)
    }

    /// Remove a triangle from the BVH by its primitive_id.
    /// After removing, call `convert_bvh2_to_cwbvh()` to update GPU data.
    pub fn remove_triangle(&mut self, prim_id: u32) -> Result<(), JsError> {
        let bvh2 = self.bvh2.as_mut()
            .ok_or_else(|| JsError::new("No BVH2 — construct with keep_bvh2 = true"))?;
        bvh2.remove_primitive(prim_id);
        Ok(())
    }

    /// Update the AABB of a primitive and refit up the tree.
    /// Useful when an object moves — update its bounding box, then refit.
    /// `aabb_flat` must be [min_x, min_y, min_z, max_x, max_y, max_z].
    pub fn resize_primitive(
        &mut self,
        prim_id: u32,
        min_x: f32, min_y: f32, min_z: f32,
        max_x: f32, max_y: f32, max_z: f32,
    ) -> Result<(), JsError> {
        let bvh2 = self.bvh2.as_mut()
            .ok_or_else(|| JsError::new("No BVH2 — construct with keep_bvh2 = true"))?;

        bvh2.init_primitives_to_nodes_if_uninit();
        let node_id = bvh2.primitives_to_nodes[prim_id as usize] as usize;
        let aabb = Aabb {
            min: Vec3A::new(min_x, min_y, min_z),
            max: Vec3A::new(max_x, max_y, max_z),
        };
        bvh2.resize_node(node_id, aabb);
        Ok(())
    }

    /// Reinsert a primitive's node to a potentially better position in the tree.
    /// Call after `resize_primitive` for moved objects.
    pub fn reinsert_primitive(&mut self, prim_id: u32) -> Result<(), JsError> {
        let bvh2 = self.bvh2.as_mut()
            .ok_or_else(|| JsError::new("No BVH2 — construct with keep_bvh2 = true"))?;

        bvh2.init_primitives_to_nodes_if_uninit();
        let node_id = bvh2.primitives_to_nodes[prim_id as usize] as usize;
        bvh2.reinsert_node(node_id);
        Ok(())
    }

    /// Batch reinsert: resize multiple primitives, then run the reinsertion optimizer.
    /// `prim_ids` — list of primitive IDs that have moved.
    /// `aabbs_flat` — flat array of new AABBs (6 floats each: min_xyz, max_xyz).
    pub fn batch_update_and_reinsert(
        &mut self,
        prim_ids: &[u32],
        aabbs_flat: &[f32],
    ) -> Result<(), JsError> {
        if aabbs_flat.len() != prim_ids.len() * 6 {
            return Err(JsError::new("aabbs_flat length must be prim_ids.len() * 6"));
        }
        let bvh2 = self.bvh2.as_mut()
            .ok_or_else(|| JsError::new("No BVH2 — construct with keep_bvh2 = true"))?;

        bvh2.init_primitives_to_nodes_if_uninit();
        self.temp_indices.clear();

        for (i, &pid) in prim_ids.iter().enumerate() {
            let b = i * 6;
            let aabb = Aabb {
                min: Vec3A::new(aabbs_flat[b], aabbs_flat[b + 1], aabbs_flat[b + 2]),
                max: Vec3A::new(aabbs_flat[b + 3], aabbs_flat[b + 4], aabbs_flat[b + 5]),
            };
            let node_id = bvh2.primitives_to_nodes[pid as usize];
            bvh2.resize_node(node_id as usize, aabb);
            self.temp_indices.push(node_id);
        }

        self.reinsertion_optimizer
            .run_with_candidates(bvh2, &self.temp_indices, 1);
        Ok(())
    }

    /// Partial rebuild: mark moved primitives and rebuild only affected subtrees.
    /// `prim_ids` — list of primitive IDs that have moved.
    /// `aabbs_flat` — flat array of new AABBs (6 floats each).
    pub fn partial_rebuild(
        &mut self,
        prim_ids: &[u32],
        aabbs_flat: &[f32],
    ) -> Result<(), JsError> {
        if aabbs_flat.len() != prim_ids.len() * 6 {
            return Err(JsError::new("aabbs_flat length must be prim_ids.len() * 6"));
        }
        let bvh2 = self.bvh2.as_mut()
            .ok_or_else(|| JsError::new("No BVH2 — construct with keep_bvh2 = true"))?;

        bvh2.init_primitives_to_nodes_if_uninit();
        self.temp_indices.clear();

        for (i, &pid) in prim_ids.iter().enumerate() {
            let b = i * 6;
            let aabb = Aabb {
                min: Vec3A::new(aabbs_flat[b], aabbs_flat[b + 1], aabbs_flat[b + 2]),
                max: Vec3A::new(aabbs_flat[b + 3], aabbs_flat[b + 4], aabbs_flat[b + 5]),
            };
            let node_id = bvh2.primitives_to_nodes[pid as usize];
            self.temp_indices.push(node_id);
            bvh2.nodes[node_id as usize].set_aabb(aabb);
        }

        bvh2.init_parents_if_uninit();
        compute_rebuild_path_flags(bvh2, &self.temp_indices, &mut self.temp_flags);

        let ploc = self.ploc_builder.get_or_insert_with(|| {
            PlocBuilder::with_capacity(self.triangles.len())
        });
        let flags = &self.temp_flags;
        ploc.partial_rebuild(
            bvh2,
            |node_id| flags[node_id],
            PlocSearchDistance::Minimum,
            SortPrecision::U64,
            0,
        );
        Ok(())
    }

    /// Refit all nodes in the BVH2 (bottom-up AABB recalculation).
    pub fn refit_all(&mut self) -> Result<(), JsError> {
        let bvh2 = self.bvh2.as_mut()
            .ok_or_else(|| JsError::new("No BVH2 — construct with keep_bvh2 = true"))?;
        bvh2.refit_all();
        Ok(())
    }

    /// Free this handle explicitly. Also called automatically when GC collects.
    pub fn free(self) {
        drop(self);
    }
}

// ============================================================================
// TLAS Scene — Top Level Acceleration Structure for multiple BLAS instances.
//
// Data layout aligns with tray_racing (rt_gpu_software_query_tlas.hlsl):
//   bvh_nodes buffer:  [BLAS_0 nodes | BLAS_1 nodes | ... | BLAS_N nodes | TLAS nodes]
//   primitive_indices:  [BLAS_0 indices | ... | BLAS_N indices | TLAS indices (instance_ids)]
//   triangles buffer:   [BLAS_0 tris | BLAS_1 tris | ... | BLAS_N tris]
//   blas_offsets:        per instance_id → node offset (INVALID = procedural)
//   tlas_start:          node offset where TLAS nodes begin in bvh_nodes
//
// Key optimization: blas_instance_ids and blas_types are eliminated.
//   - instance_id is read from primitive_indices[] at TLAS leaf level
//   - procedural detection uses blas_offsets[instance_id] == INVALID sentinel
// ============================================================================

#[wasm_bindgen]
pub struct WasmTlasScene {
    /// Concatenated node bytes: [BLAS_0..N nodes | TLAS nodes], as raw u8
    all_nodes_bytes: Vec<u8>,
    /// Concatenated primitive indices: [BLAS_0..N indices | TLAS indices]
    all_indices: Vec<u32>,
    /// Concatenated triangles from all BLAS instances
    all_triangles: Vec<Triangle>,
    /// Per TLAS-primitive → BLAS node offset in all_nodes (in node units, not bytes).
    /// Indexed by the value in TLAS primitive_indices[i].
    /// For procedural BLAS instances, this is set to INVALID (0xFFFFFFFF) as a sentinel:
    /// the shader checks `blas_offsets[i] == 0xFFFFFFFF` to detect procedural geometry
    /// and skip BLAS traversal (analogous to DXR intersection shaders).
    /// The instance_id is implicitly `primitive_indices[i]` from the TLAS layer
    /// (entries are sorted by instance_index, so primitive_indices maps directly).
    blas_offsets: Vec<u32>,
    /// Node index where TLAS begins in all_nodes (= total BLAS node count)
    tlas_start: u32,
    build_time_ms: f64,
    total_node_count: usize,
    total_triangle_count: usize,
}

#[wasm_bindgen]
impl WasmTlasScene {
    /// Build a TLAS scene from mixed BLAS types: triangle meshes + procedural AABBs.
    ///
    /// **Triangle mesh BLAS:**
    /// - `blas_data_flat`: all BLAS triangles concatenated (f32, every 9 = 1 triangle)
    /// - `blas_tri_counts`: number of triangles per triangle-mesh BLAS instance (u32 array)
    ///
    /// **Procedural BLAS (AABB-only, like DXR PROCEDURAL_PRIMITIVE_AABBS):**
    /// - `procedural_aabbs_flat`: flat f32 array, every 6 floats = 1 AABB (min_xyz, max_xyz)
    /// - `procedural_instance_indices`: for each procedural BLAS, its position in the final
    ///   TLAS instance list. E.g., if we have 100 triangle-mesh + 100 procedural instances
    ///   interleaved, this array tells us where each procedural one sits.
    ///
    /// The total TLAS instance count = blas_tri_counts.len() + procedural count.
    /// Procedural detection uses blas_offsets sentinel (INVALID_U32).
    /// instance_id is stored in primitive_indices at TLAS level.
    ///
    /// `quality`: build quality for both BLAS and TLAS construction
    #[wasm_bindgen(constructor)]
    pub fn new(
        blas_data_flat: &[f32],
        blas_tri_counts: &[u32],
        procedural_aabbs_flat: &[f32],
        procedural_instance_indices: &[u32],
        quality: BuildQuality,
    ) -> Result<WasmTlasScene, JsError> {
        let start = js_sys::Date::now();
        let config = quality.to_params();
        let tri_blas_count = blas_tri_counts.len();
        let proc_blas_count = procedural_aabbs_flat.len() / 6;

        if procedural_aabbs_flat.len() % 6 != 0 {
            return Err(JsError::new("procedural_aabbs_flat length must be divisible by 6"));
        }
        if procedural_instance_indices.len() != proc_blas_count {
            return Err(JsError::new("procedural_instance_indices length must match procedural AABB count"));
        }

        let total_blas_count = tri_blas_count + proc_blas_count;

        // Track per-BLAS metadata: (cwbvh, is_procedural, original_instance_index)
        // We'll sort them by instance_index later to build the TLAS in the right order.

        // --- 1a. Build triangle-mesh BLAS instances ---
        struct BlasEntry {
            cwbvh: CwBvh,
            is_procedural: bool,
            instance_index: u32,  // position in final TLAS instance list
            tri_count: u32,       // 0 for procedural
        }

        // Determine triangle mesh instance indices: they fill the slots NOT taken by procedural
        let mut proc_set = std::collections::HashSet::new();
        for &idx in procedural_instance_indices.iter() {
            proc_set.insert(idx);
        }

        let mut tri_instance_indices: Vec<u32> = Vec::with_capacity(tri_blas_count);
        {
            let mut ti = 0u32;
            for i in 0..total_blas_count as u32 {
                if !proc_set.contains(&i) {
                    tri_instance_indices.push(i);
                    ti += 1;
                    if ti as usize >= tri_blas_count {
                        break;
                    }
                }
            }
        }

        if tri_instance_indices.len() != tri_blas_count {
            return Err(JsError::new("Instance index assignment mismatch: not enough slots for triangle BLAS"));
        }

        let mut entries: Vec<BlasEntry> = Vec::with_capacity(total_blas_count);
        let mut all_triangles: Vec<Triangle> = Vec::new();
        let mut offset: usize = 0;

        for (i, &tri_count) in blas_tri_counts.iter().enumerate() {
            let float_count = tri_count as usize * 9;
            if offset + float_count > blas_data_flat.len() {
                return Err(JsError::new("blas_data_flat too short for given blas_tri_counts"));
            }
            let tris = parse_triangles(&blas_data_flat[offset..offset + float_count]);
            offset += float_count;

            let mut build_time = Duration::default();
            let cwbvh = build_cwbvh_from_tris(&tris, config, &mut build_time);
            let tc = tris.len() as u32;
            all_triangles.extend_from_slice(&tris);
            entries.push(BlasEntry {
                cwbvh,
                is_procedural: false,
                instance_index: tri_instance_indices[i],
                tri_count: tc,
            });
        }

        // --- 1b. Build procedural BLAS instances (AABB-only, single-leaf CWBVH) ---
        for (i, idx) in procedural_instance_indices.iter().enumerate() {
            let b = i * 6;
            let aabb = Aabb {
                min: Vec3A::new(procedural_aabbs_flat[b], procedural_aabbs_flat[b + 1], procedural_aabbs_flat[b + 2]),
                max: Vec3A::new(procedural_aabbs_flat[b + 3], procedural_aabbs_flat[b + 4], procedural_aabbs_flat[b + 5]),
            };

            // Build a CWBVH from a single AABB primitive.
            // This creates a minimal BVH with one leaf node pointing to one primitive.
            let mut build_time = Duration::default();
            let cwbvh = obvhs::cwbvh::builder::build_cwbvh(&[aabb], config, &mut build_time);

            entries.push(BlasEntry {
                cwbvh,
                is_procedural: true,
                instance_index: *idx,
                tri_count: 0,
            });
        }

        // --- 2. Sort entries by instance_index for deterministic TLAS ordering ---
        entries.sort_by_key(|e| e.instance_index);

        // --- 3. Build TLAS from all BLAS AABBs ---
        let tlas_aabbs: Vec<Aabb> = entries.iter().map(|e| e.cwbvh.total_aabb).collect();
        let mut tlas_build_time = Duration::default();
        let tlas = obvhs::cwbvh::builder::build_cwbvh(&tlas_aabbs, config, &mut tlas_build_time);

        // --- 4. Concatenate all data (tray_racing layout) ---

        // 4a. Nodes: [BLAS_0 nodes | ... | BLAS_N nodes | TLAS nodes]
        let mut all_nodes: Vec<obvhs::cwbvh::node::CwBvhNode> = Vec::new();
        let mut blas_node_offsets: Vec<u32> = Vec::with_capacity(total_blas_count);
        let mut blas_is_procedural: Vec<bool> = Vec::with_capacity(total_blas_count);
        let mut global_prim_idx_offset: u32 = 0;
        let mut global_tri_offset: u32 = 0;
        let mut all_indices: Vec<u32> = Vec::new();

        for entry in &entries {
            let node_offset_in_units = all_nodes.len() as u32;
            blas_node_offsets.push(node_offset_in_units);
            blas_is_procedural.push(entry.is_procedural);

            // Copy BLAS nodes with patched primitive_base_idx
            for node in &entry.cwbvh.nodes {
                let mut patched = *node;
                patched.primitive_base_idx += global_prim_idx_offset;
                all_nodes.push(patched);
            }

            // Copy BLAS primitive_indices with global triangle offset
            // For procedural BLAS, primitive_indices still exist (the AABB primitive),
            // but we won't use them for triangle lookup — the shader detects procedural
            // via blas_offsets == INVALID_U32 sentinel.
            for &idx in &entry.cwbvh.primitive_indices {
                if entry.is_procedural {
                    // Procedural: push a sentinel value (won't be used for triangle lookup)
                    all_indices.push(0xFFFFFFFF);
                } else {
                    all_indices.push(idx + global_tri_offset);
                }
            }

            global_prim_idx_offset += entry.cwbvh.primitive_indices.len() as u32;
            if !entry.is_procedural {
                global_tri_offset += entry.tri_count;
            }
        }

        let tlas_start = all_nodes.len() as u32;

        // The TLAS primitive_indices will be appended to all_indices so that the
        // shader can read instance_id via `primitive_indices[global_triangle_index]`.
        // We need to patch TLAS nodes' primitive_base_idx to point into this region.
        let tlas_prim_idx_offset = all_indices.len() as u32;

        // Append TLAS nodes with patched primitive_base_idx.
        for node in &tlas.nodes {
            let mut patched = *node;
            patched.primitive_base_idx += tlas_prim_idx_offset;
            all_nodes.push(patched);
        }

        // Append TLAS primitive_indices to all_indices.
        // Each entry stores the instance_index (= sorted position = BLAS entry index),
        // so the shader can read instance_id = primitive_indices[global_triangle_index].
        // Since entries are sorted by instance_index (0..N), we have:
        //   tlas.primitive_indices[i] → entries[prim_idx].instance_index = prim_idx
        // So we just store the instance_index directly.
        for &prim_idx in &tlas.primitive_indices {
            all_indices.push(entries[prim_idx as usize].instance_index);
        }

        let all_nodes_bytes: Vec<u8> = bytemuck::cast_slice(&all_nodes).to_vec();

        // 4b. Build blas_offsets indexed by instance_id (0..total_blas_count).
        // For procedural BLAS, set to INVALID (0xFFFFFFFF) as sentinel.
        // The shader reads instance_id from primitive_indices, then indexes blas_offsets[instance_id].
        // Since entries are sorted by instance_index, blas_offsets[i] corresponds to entry i.
        let blas_offsets: Vec<u32> = entries.iter().enumerate().map(|(i, _)| {
            if blas_is_procedural[i] {
                0xFFFFFFFFu32  // sentinel: procedural geometry
            } else {
                blas_node_offsets[i]
            }
        }).collect();

        let total_node_count = all_nodes.len();
        let total_triangle_count = all_triangles.len();
        let build_time_ms = js_sys::Date::now() - start;

        Ok(WasmTlasScene {
            all_nodes_bytes,
            all_indices,
            all_triangles,
            blas_offsets,
            tlas_start,
            build_time_ms,
            total_node_count,
            total_triangle_count,
        })
    }

    // ── Data export for GPU upload (zero-copy from WASM linear memory) ──

    pub fn nodes_ptr(&self) -> *const u8 {
        self.all_nodes_bytes.as_ptr()
    }

    pub fn nodes_byte_len(&self) -> usize {
        self.all_nodes_bytes.len()
    }

    pub fn indices_ptr(&self) -> *const u8 {
        self.all_indices.as_ptr() as *const u8
    }

    pub fn indices_byte_len(&self) -> usize {
        self.all_indices.len() * 4
    }

    pub fn triangles_ptr(&self) -> *const u8 {
        self.all_triangles.as_ptr() as *const u8
    }

    pub fn triangles_byte_len(&self) -> usize {
        self.all_triangles.len() * std::mem::size_of::<Triangle>()
    }

    pub fn blas_offsets_ptr(&self) -> *const u8 {
        self.blas_offsets.as_ptr() as *const u8
    }

    pub fn blas_offsets_byte_len(&self) -> usize {
        self.blas_offsets.len() * 4
    }

    pub fn tlas_start(&self) -> u32 {
        self.tlas_start
    }

    pub fn build_time_ms(&self) -> f64 {
        self.build_time_ms
    }

    pub fn total_node_count(&self) -> usize {
        self.total_node_count
    }

    pub fn total_triangle_count(&self) -> usize {
        self.total_triangle_count
    }

    pub fn free(self) {
        drop(self);
    }
}
