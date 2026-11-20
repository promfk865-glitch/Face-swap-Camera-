import {
  FaceLandmarker,
  FilesetResolver,
  type Classifications,
  type NormalizedLandmark,
} from '@mediapipe/tasks-vision';

export interface FaceBounds {
  cx: number;
  cy: number;
  w: number;
  h: number;
  seenAtMs: number;
}

export async function loadMediaPipeVision() {
  try {
    return await FilesetResolver.forVisionTasks('/mediapipe/wasm');
  } catch {
    return await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm',
    );
  }
}

export async function createResilientLandmarker(
  vision: Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>,
  runningMode: 'VIDEO' | 'IMAGE',
  numFaces = 1,
): Promise<FaceLandmarker> {
  const localModel = '/mediapipe/face_landmarker.task';
  const remoteModel =
    'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

  for (const modelAssetPath of [localModel, remoteModel]) {
    for (const delegate of ['GPU', 'CPU'] as const) {
      try {
        return await FaceLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath, delegate },
          runningMode,
          numFaces,
          minFaceDetectionConfidence: 0.2,
          minFacePresenceConfidence: 0.2,
          minTrackingConfidence: 0.15,
          outputFaceBlendshapes: true,
        });
      } catch {
        // Fallback cleanly to next delegate/model path
      }
    }
  }
  throw new Error(`Failed to initialize FaceLandmarker (${runningMode})`);
}

export function extractBounds(landmarks: NormalizedLandmark[], seenAtMs: number): FaceBounds {
  let minX = 1, minY = 1, maxX = 0, maxY = 0;
  for (const lm of landmarks) {
    if (lm.x < minX) minX = lm.x;
    if (lm.x > maxX) maxX = lm.x;
    if (lm.y < minY) minY = lm.y;
    if (lm.y > maxY) maxY = lm.y;
  }
  return {
    cx: (minX + maxX) * 0.5,
    cy: (minY + maxY) * 0.5,
    w: Math.max(0.08, maxX - minX),
    h: Math.max(0.08, maxY - minY),
    seenAtMs,
  };
}

const roiCanvasCache = typeof document !== 'undefined' ? document.createElement('canvas') : null;
if (roiCanvasCache) {
  roiCanvasCache.width = 256;
  roiCanvasCache.height = 256;
}

/**
 * Fast downscaled detection + padded shadow-lifted ROI fallback so tilted/side-profile
 * or backlit faces keep tracking reliably instead of snapping out when turning the head.
 */
export function detectSingleFaceResilient(
  videoLm: FaceLandmarker | null,
  imageLm: FaceLandmarker | null,
  fullCanvas: HTMLCanvasElement,
  detectCanvas: HTMLCanvasElement,
  lastBounds: FaceBounds | null,
  nowMs: number,
): {
  landmarks: NormalizedLandmark[] | null;
  blendshapes: Classifications | undefined;
  bounds: FaceBounds | null;
} {
  // 1) Fast downscaled full-frame pass (~384px long side)
  const scale = Math.min(1, 384 / Math.max(1, fullCanvas.width, fullCanvas.height));
  const dw = Math.max(128, Math.round(fullCanvas.width * scale));
  const dh = Math.max(128, Math.round(fullCanvas.height * scale));
  if (detectCanvas.width !== dw || detectCanvas.height !== dh) {
    detectCanvas.width = dw;
    detectCanvas.height = dh;
  }
  const dCtx = detectCanvas.getContext('2d');
  if (dCtx) {
    dCtx.drawImage(fullCanvas, 0, 0, dw, dh);
    if (videoLm) {
      const res = videoLm.detectForVideo(detectCanvas, nowMs);
      if (res.faceLandmarks.length > 0) {
        const lms = res.faceLandmarks[0];
        return {
          landmarks: lms,
          blendshapes: res.faceBlendshapes?.[0],
          bounds: extractBounds(lms, nowMs),
        };
      }
    }
  }

  // 2) ROI Fallback around last known face (zoomed + backlit contrast lift for side turns)
  if (
    imageLm &&
    roiCanvasCache &&
    lastBounds &&
    nowMs - lastBounds.seenAtMs <= 2500
  ) {
    const rCtx = roiCanvasCache.getContext('2d');
    if (rCtx) {
      const boxSize = Math.min(0.95, Math.max(lastBounds.w, lastBounds.h) * 2.15);
      const sxNorm = Math.max(0, Math.min(1 - boxSize, lastBounds.cx - boxSize * 0.5));
      const syNorm = Math.max(0, Math.min(1 - boxSize, lastBounds.cy - boxSize * 0.5));

      const sx = sxNorm * fullCanvas.width;
      const sy = syNorm * fullCanvas.height;
      const sw = Math.max(32, boxSize * fullCanvas.width);
      const sh = Math.max(32, boxSize * fullCanvas.height);

      rCtx.save();
      rCtx.clearRect(0, 0, 256, 256);
      rCtx.filter = 'brightness(1.10) contrast(1.14)';
      rCtx.drawImage(fullCanvas, sx, sy, sw, sh, 0, 0, 256, 256);
      rCtx.restore();

      const roiRes = imageLm.detect(roiCanvasCache);
      if (roiRes.faceLandmarks.length > 0) {
        const mapped: NormalizedLandmark[] = roiRes.faceLandmarks[0].map((lm) => ({
          x: Math.min(1, Math.max(0, sxNorm + lm.x * boxSize)),
          y: Math.min(1, Math.max(0, syNorm + lm.y * boxSize)),
          z: lm.z * boxSize,
          visibility: lm.visibility,
        }));
        return {
          landmarks: mapped,
          blendshapes: roiRes.faceBlendshapes?.[0],
          bounds: extractBounds(mapped, nowMs),
        };
      }
    }
  }

  return { landmarks: null, blendshapes: undefined, bounds: lastBounds };
}

/**
 * Computes per-vertex occlusion fade based on 3D head yaw so on steep 45-80 deg side turns,
 * the far occluded edge wraps smoothly behind the nose instead of stretching awkwardly.
 */
export function computeSideOcclusionWeights(
  landmarks: Array<{ x: number; y: number; z: number }>,
  baseAlphas: Float32Array,
  outAlphas: Float32Array,
) {
  // Nose bridge center x (landmark 4) vs left (234) and right (454) cheek/ear edges
  const noseX = landmarks[4]?.x ?? 0.5;
  const leftEdgeX = landmarks[234]?.x ?? 0.3;
  const rightEdgeX = landmarks[454]?.x ?? 0.7;
  const span = Math.max(0.04, Math.abs(rightEdgeX - leftEdgeX));
  // yaw in [-1, 1]: negative = turned left, positive = turned right
  const midX = (leftEdgeX + rightEdgeX) * 0.5;
  const yaw = Math.max(-1, Math.min(1, ((noseX - midX) / (span * 0.5)) * 1.4));
  const absYaw = Math.abs(yaw);

  if (absYaw < 0.35) {
    outAlphas.set(baseAlphas);
    return;
  }

  const sideFadeStrength = Math.min(1, (absYaw - 0.35) / 0.55);
  for (let i = 0; i < landmarks.length && i < baseAlphas.length; i++) {
    let a = baseAlphas[i];
    // On turned heads, vertices on the occluded side relative to nose have positive relX * sign(yaw) < 0
    const dxNorm = (landmarks[i].x - noseX) / (span * 0.5);
    const onHiddenSide = yaw < 0 ? dxNorm > 0.12 : dxNorm < -0.12;
    if (onHiddenSide) {
      const edgeFrac = Math.min(1, (Math.abs(dxNorm) - 0.12) / 0.55);
      a *= 1.0 - sideFadeStrength * edgeFrac * 0.85;
    }
    outAlphas[i] = a;
  }
}
