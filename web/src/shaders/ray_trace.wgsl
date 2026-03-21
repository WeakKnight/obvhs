// ============================================================================
// ray_trace.wgsl — Ray tracing compute shader
// Generates primary rays per pixel, traverses CWBVH, computes simple shading.
// Must be concatenated after cwbvh_common.wgsl and cwbvh_traverse.wgsl
// ============================================================================

struct CameraUniforms {
    eye: vec3<f32>,
    _pad0: f32,
    look_at: vec3<f32>,
    _pad1: f32,
    up: vec3<f32>,
    fov: f32,
    width: u32,
    height: u32,
    frame: u32,
    tlas_start: u32, // Node index where TLAS begins (aligns with tray_racing ViewUniform.tlas_start)
};

@group(1) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;

// ─── Simple hash for jittered sampling ───
fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_float(seed: u32) -> f32 {
    return f32(pcg_hash(seed)) / 4294967295.0;
}

// ─── Generate primary ray for pixel (px, py) ───
fn generate_ray(px: u32, py: u32) -> Ray {
    let aspect = f32(camera.width) / f32(camera.height);
    let fov_rad = camera.fov * 3.14159265 / 180.0;
    let half_h = tan(fov_rad * 0.5);
    let half_w = half_h * aspect;

    let w = normalize(camera.eye - camera.look_at);       // camera backward
    let u = normalize(cross(camera.up, w));                // camera right
    let v = cross(w, u);                                    // camera up

    // Map pixel to [-1, 1] with sub-pixel jitter
    let seed = px + py * camera.width + camera.frame * 1000003u;
    let jx = rand_float(seed) - 0.5;
    let jy = rand_float(seed ^ 0xABCDu) - 0.5;

    let ndc_x = (f32(px) + 0.5 + jx * 0.5) / f32(camera.width) * 2.0 - 1.0;
    let ndc_y = 1.0 - (f32(py) + 0.5 + jy * 0.5) / f32(camera.height) * 2.0;

    let dir = normalize(ndc_x * half_w * u + ndc_y * half_h * v - w);

    return make_ray(camera.eye, dir, 0.0001, 1e30);
}

// ─── Compute triangle normal at hit point ───
fn compute_normal(prim_id: u32) -> vec3<f32> {
    let v0 = load_triangle_v0(prim_id);
    let v1 = load_triangle_v1(prim_id);
    let v2 = load_triangle_v2(prim_id);
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    return normalize(cross(e1, e2));
}

// ─── Shading ───
fn shade(ray: Ray, hit: RayHit) -> vec3<f32> {
    if hit.primitive_id == INVALID_U32 {
        // Sky gradient
        let t = ray.direction.y * 0.5 + 0.5;
        return mix(vec3<f32>(0.02, 0.02, 0.06), vec3<f32>(0.05, 0.1, 0.2), t);
    }

    let hit_pos = ray.origin + ray.direction * hit.t;
    var n: vec3<f32>;
    var base_color: vec3<f32>;

    if hit.is_procedural == 1u {
        // ═══ Procedural sphere shading ═══
        // Read sphere center+radius from app-layer per-instance data via instance_id.
        // This validates that instance_id is correctly propagated through TLAS traversal.
        let sphere = load_sphere_center_radius(hit.instance_id);
        let center = sphere.xyz;
        let radius = sphere.w;
        n = normalize(hit_pos - center);

        // Give procedural spheres a distinctive color based on instance_id
        // so we can visually verify each sphere has the correct instance mapping.
        let id_f = f32(hit.instance_id);
        base_color = vec3<f32>(
            fract(id_f * 0.1234) * 0.5 + 0.3,
            fract(id_f * 0.5678) * 0.5 + 0.3,
            fract(id_f * 0.9012) * 0.5 + 0.3
        );
    } else {
        // ═══ Triangle mesh shading ═══
        n = compute_normal(hit.primitive_id);
        // Base color from normal
        base_color = abs(n) * 0.5 + 0.3;
    }

    // Simple directional light + ambient
    let light_dir = normalize(vec3<f32>(0.5, 0.8, 0.3));
    let ndl = max(dot(n, light_dir), 0.0);

    // Shadow ray
    let shadow_ray = make_ray(hit_pos + n * 0.001, light_dir, 0.001, 100.0);
    let in_shadow = trace_shadow_ray(shadow_ray);
    let shadow_factor = select(1.0, 0.3, in_shadow);

    // Ambient occlusion approximation via hemisphere ray
    let ambient = 0.15;
    let diffuse = ndl * shadow_factor * 0.7;

    return base_color * (ambient + diffuse);
}

// ─── Ray Tracing Entry Points ───
// These functions are injected by the host code at shader concatenation time.
// Single-BVH mode: calls traverse_cwbvh_closest / traverse_cwbvh_any
// TLAS mode: calls traverse_tlas_closest / traverse_tlas_any
/*TRACE_IMPL*/

// ─── Compute Entry Point ───
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    if px >= camera.width || py >= camera.height { return; }

    let ray = generate_ray(px, py);
    let hit = trace_ray(ray);
    let color = shade(ray, hit);

    // Tone mapping (simple Reinhard)
    let mapped = color / (color + vec3<f32>(1.0));
    // Gamma correction
    let gamma = pow(mapped, vec3<f32>(1.0 / 2.2));

    textureStore(output_texture, vec2<i32>(i32(px), i32(py)), vec4<f32>(gamma, 1.0));
}
