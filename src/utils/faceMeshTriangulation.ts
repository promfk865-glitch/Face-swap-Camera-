export const INNER_LIP_RING = [
  78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
  308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
];

export const FACE_OVAL_RING = new Set([
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
  397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
  172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]);

// Inner eye socket aperture vertices so live eye gaze/blinking blends naturally without dead black pupils
export const LEFT_EYE_APERTURE = new Set([
  33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7,
]);

export const RIGHT_EYE_APERTURE = new Set([
  362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382,
]);

// Upper forehead / temple indices where hair or background from photos tends to bleed over the user
export const FOREHEAD_RIM_RING = new Set([
  10, 338, 297, 332, 284, 251, 389, 21, 54, 103, 67, 109,
  151, 337, 299, 333, 298, 301, 108, 69, 104, 68, 71,
]);

// Representative cheek & nose-bridge landmarks for fast skin luminance/color matching
export const SKIN_SAMPLE_INDICES = [50, 280, 101, 330, 195, 4];

const INNER_LIP_SET = new Set(INNER_LIP_RING);

export interface FaceMeshTopology {
  triangles: Uint16Array;
  vertexAlphas: Float32Array;
}

export function buildFaceMeshTopology(
  tesselationEdges: Array<{ start: number; end: number }>,
  vertexCount = 478,
): FaceMeshTopology {
  const adj = new Map<number, Set<number>>();
  for (const { start, end } of tesselationEdges) {
    if (start === end) continue;
    if (!adj.has(start)) adj.set(start, new Set());
    if (!adj.has(end)) adj.set(end, new Set());
    adj.get(start)!.add(end);
    adj.get(end)!.add(start);
  }

  // BFS inward from FACE_OVAL_RING for smooth 4-ring edge feathering
  const distFromOval = new Int16Array(vertexCount).fill(99);
  const queue: number[] = [];
  for (const idx of FACE_OVAL_RING) {
    if (idx < vertexCount) {
      distFromOval[idx] = 0;
      queue.push(idx);
    }
  }
  let head = 0;
  while (head < queue.length) {
    const curr = queue[head++];
    const nextDist = distFromOval[curr] + 1;
    const neighbors = adj.get(curr);
    if (!neighbors) continue;
    for (const nb of neighbors) {
      if (nb < vertexCount && distFromOval[nb] > nextDist) {
        distFromOval[nb] = nextDist;
        queue.push(nb);
      }
    }
  }

  const vertexAlphas = new Float32Array(vertexCount);
  for (let i = 0; i < vertexCount; i++) {
    const d = distFromOval[i];
    let alpha = d === 0 ? 0.0 : d === 1 ? 0.16 : d === 2 ? 0.52 : d === 3 ? 0.85 : 1.0;
    if (FOREHEAD_RIM_RING.has(i)) {
      alpha *= 0.42;
    }
    // Keep live eye sclera/iris cleanly visible inside the eye socket so eyes never look solid black
    if (LEFT_EYE_APERTURE.has(i) || RIGHT_EYE_APERTURE.has(i)) {
      alpha = Math.min(alpha, 0.12);
    }
    vertexAlphas[i] = alpha;
  }

  const triangles: number[] = [];
  const seen = new Set<string>();

  for (const [u, uNeighbors] of adj.entries()) {
    for (const v of uNeighbors) {
      if (v <= u) continue;
      const vNeighbors = adj.get(v);
      if (!vNeighbors) continue;
      for (const w of vNeighbors) {
        if (w <= v) continue;
        if (!uNeighbors.has(w)) continue;
        // Skip triangles strictly bridging across the inner mouth opening
        if (INNER_LIP_SET.has(u) && INNER_LIP_SET.has(v) && INNER_LIP_SET.has(w)) {
          continue;
        }
        // Skip eyeball interior triangles formed exclusively by inner eye ring vertices
        if (
          (LEFT_EYE_APERTURE.has(u) && LEFT_EYE_APERTURE.has(v) && LEFT_EYE_APERTURE.has(w)) ||
          (RIGHT_EYE_APERTURE.has(u) && RIGHT_EYE_APERTURE.has(v) && RIGHT_EYE_APERTURE.has(w))
        ) {
          continue;
        }
        const key = `${u},${v},${w}`;
        if (!seen.has(key)) {
          seen.add(key);
          triangles.push(u, v, w);
        }
      }
    }
  }

  return {
    triangles: new Uint16Array(triangles),
    vertexAlphas,
  };
}
