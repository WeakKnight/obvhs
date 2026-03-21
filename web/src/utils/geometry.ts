/**
 * Geometry utilities — generate triangle meshes as flat Float32Array.
 * Each triangle = 9 floats (v0.xyz, v1.xyz, v2.xyz).
 */

/** Generate a UV sphere mesh. */
export function generateSphere(
  center: [number, number, number],
  radius: number,
  segments: number = 16,
  rings: number = 12
): Float32Array {
  const verts: number[] = [];

  for (let r = 0; r < rings; r++) {
    const theta0 = (r / rings) * Math.PI;
    const theta1 = ((r + 1) / rings) * Math.PI;

    for (let s = 0; s < segments; s++) {
      const phi0 = (s / segments) * 2 * Math.PI;
      const phi1 = ((s + 1) / segments) * 2 * Math.PI;

      const p00 = spherePoint(center, radius, theta0, phi0);
      const p10 = spherePoint(center, radius, theta1, phi0);
      const p01 = spherePoint(center, radius, theta0, phi1);
      const p11 = spherePoint(center, radius, theta1, phi1);

      // Two triangles per quad
      verts.push(...p00, ...p10, ...p11);
      verts.push(...p00, ...p11, ...p01);
    }
  }

  return new Float32Array(verts);
}

function spherePoint(
  c: [number, number, number],
  r: number,
  theta: number,
  phi: number
): [number, number, number] {
  return [
    c[0] + r * Math.sin(theta) * Math.cos(phi),
    c[1] + r * Math.cos(theta),
    c[2] + r * Math.sin(theta) * Math.sin(phi),
  ];
}

/** Generate an icosphere via subdivision. */
export function generateIcosphere(
  center: [number, number, number] = [0, 0, 0],
  radius: number = 1,
  subdivisions: number = 3
): Float32Array {
  const t = (1 + Math.sqrt(5)) / 2;

  // Initial icosahedron vertices
  let vertices: [number, number, number][] = [
    [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
    [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
    [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
  ];

  // Normalize to unit sphere
  vertices = vertices.map((v) => {
    const len = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    return [v[0] / len, v[1] / len, v[2] / len];
  });

  let faces: [number, number, number][] = [
    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
    [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
    [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
  ];

  // Subdivide
  for (let i = 0; i < subdivisions; i++) {
    const newFaces: [number, number, number][] = [];
    const midpointCache = new Map<string, number>();

    const getMidpoint = (i0: number, i1: number): number => {
      const key = `${Math.min(i0, i1)}_${Math.max(i0, i1)}`;
      if (midpointCache.has(key)) return midpointCache.get(key)!;
      const v0 = vertices[i0];
      const v1 = vertices[i1];
      const mid: [number, number, number] = [
        (v0[0] + v1[0]) / 2,
        (v0[1] + v1[1]) / 2,
        (v0[2] + v1[2]) / 2,
      ];
      const len = Math.sqrt(mid[0] ** 2 + mid[1] ** 2 + mid[2] ** 2);
      const idx = vertices.length;
      vertices.push([mid[0] / len, mid[1] / len, mid[2] / len]);
      midpointCache.set(key, idx);
      return idx;
    };

    for (const [a, b, c] of faces) {
      const ab = getMidpoint(a, b);
      const bc = getMidpoint(b, c);
      const ca = getMidpoint(c, a);
      newFaces.push([a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]);
    }
    faces = newFaces;
  }

  // Output as flat triangle array
  const result = new Float32Array(faces.length * 9);
  for (let i = 0; i < faces.length; i++) {
    const [a, b, c] = faces[i];
    const base = i * 9;
    result[base + 0] = center[0] + vertices[a][0] * radius;
    result[base + 1] = center[1] + vertices[a][1] * radius;
    result[base + 2] = center[2] + vertices[a][2] * radius;
    result[base + 3] = center[0] + vertices[b][0] * radius;
    result[base + 4] = center[1] + vertices[b][1] * radius;
    result[base + 5] = center[2] + vertices[b][2] * radius;
    result[base + 6] = center[0] + vertices[c][0] * radius;
    result[base + 7] = center[1] + vertices[c][1] * radius;
    result[base + 8] = center[2] + vertices[c][2] * radius;
  }
  return result;
}

/** Generate a Cornell Box scene (open-front box + 2 boxes inside). */
export function generateCornellBox(): Float32Array {
  return generateCornellBoxAt([0, 0, 0]);
}

/**
 * Generate a Cornell Box at a given center position.
 * Each Cornell Box is a 2×2×2 unit box (from center-1 to center+1 on each axis).
 * Includes an icosphere inside the box.
 */
export function generateCornellBoxAt(center: [number, number, number]): Float32Array {
  const tris: number[] = [];
  const [cx, cy, cz] = center;

  const quad = (
    a: [number, number, number], b: [number, number, number],
    c: [number, number, number], d: [number, number, number]
  ) => {
    tris.push(a[0] + cx, a[1] + cy, a[2] + cz);
    tris.push(b[0] + cx, b[1] + cy, b[2] + cz);
    tris.push(c[0] + cx, c[1] + cy, c[2] + cz);
    tris.push(a[0] + cx, a[1] + cy, a[2] + cz);
    tris.push(c[0] + cx, c[1] + cy, c[2] + cz);
    tris.push(d[0] + cx, d[1] + cy, d[2] + cz);
  };

  // Room: left=-1, right=1, bottom=-1, top=1, back=-1, front=1
  // Floor (white)
  quad([-1, -1, -1], [1, -1, -1], [1, -1, 1], [-1, -1, 1]);
  // Ceiling (white)
  quad([-1, 1, -1], [-1, 1, 1], [1, 1, 1], [1, 1, -1]);
  // Back wall (white)
  quad([-1, -1, -1], [-1, 1, -1], [1, 1, -1], [1, -1, -1]);
  // Left wall (red)
  quad([-1, -1, -1], [-1, -1, 1], [-1, 1, 1], [-1, 1, -1]);
  // Right wall (green)
  quad([1, -1, -1], [1, 1, -1], [1, 1, 1], [1, -1, 1]);

  // Tall box
  const bx = 0.3, bz = -0.3;
  const bh = 0.6, bs = 0.3;
  quad([bx - bs, -1, bz - bs], [bx + bs, -1, bz - bs], [bx + bs, -1 + bh, bz - bs], [bx - bs, -1 + bh, bz - bs]);
  quad([bx - bs, -1, bz + bs], [bx - bs, -1 + bh, bz + bs], [bx + bs, -1 + bh, bz + bs], [bx + bs, -1, bz + bs]);
  quad([bx - bs, -1, bz - bs], [bx - bs, -1 + bh, bz - bs], [bx - bs, -1 + bh, bz + bs], [bx - bs, -1, bz + bs]);
  quad([bx + bs, -1, bz - bs], [bx + bs, -1, bz + bs], [bx + bs, -1 + bh, bz + bs], [bx + bs, -1 + bh, bz - bs]);
  quad([bx - bs, -1 + bh, bz - bs], [bx + bs, -1 + bh, bz - bs], [bx + bs, -1 + bh, bz + bs], [bx - bs, -1 + bh, bz + bs]);

  // Short box
  const sx = -0.3, sz = 0.3;
  const sh = 0.3, ss = 0.3;
  quad([sx - ss, -1, sz - ss], [sx + ss, -1, sz - ss], [sx + ss, -1 + sh, sz - ss], [sx - ss, -1 + sh, sz - ss]);
  quad([sx - ss, -1, sz + ss], [sx - ss, -1 + sh, sz + ss], [sx + ss, -1 + sh, sz + ss], [sx + ss, -1, sz + ss]);
  quad([sx - ss, -1, sz - ss], [sx - ss, -1 + sh, sz - ss], [sx - ss, -1 + sh, sz + ss], [sx - ss, -1, sz + ss]);
  quad([sx + ss, -1, sz - ss], [sx + ss, -1, sz + ss], [sx + ss, -1 + sh, sz + ss], [sx + ss, -1 + sh, sz - ss]);
  quad([sx - ss, -1 + sh, sz - ss], [sx + ss, -1 + sh, sz - ss], [sx + ss, -1 + sh, sz + ss], [sx - ss, -1 + sh, sz + ss]);

  // Icosphere — sitting on the short box
  const sphereCenter: [number, number, number] = [cx + sx, cy - 1 + sh + 0.25, cz + sz];
  const sphereRadius = 0.25;
  const sphereTris = generateIcosphere(sphereCenter, sphereRadius, 2);

  // Combine room + boxes + sphere
  const roomData = new Float32Array(tris);
  const result = new Float32Array(roomData.length + sphereTris.length);
  result.set(roomData);
  result.set(sphereTris, roomData.length);
  return result;
}

/** Combine multiple triangle arrays into one. */
export function mergeTriangleArrays(...arrays: Float32Array[]): Float32Array {
  const totalLen = arrays.reduce((sum, a) => sum + a.length, 0);
  const result = new Float32Array(totalLen);
  let offset = 0;
  for (const arr of arrays) {
    result.set(arr, offset);
    offset += arr.length;
  }
  return result;
}
