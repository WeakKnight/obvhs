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
