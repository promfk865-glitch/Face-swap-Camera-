import type { FaceLandmarker, NormalizedLandmark, Classifications } from '@mediapipe/tasks-vision';
import { LandmarkOneEuroSmoother } from './oneEuroFilter';

export interface VideoTrajectoryKeyframe {
  timeMs: number;
  cx: number;
  cy: number;
  faceWidth: number;
  yaw: number;
  jawOpen: number;
  landmarks: Array<{ x: number; y: number; z: number }>;
}

export interface DetectedVideoPerson {
  id: string;
  index: number;
  labelEn: string;
  labelBn: string;
  thumbnailUrl: string;
  anchorX: number;
  anchorY: number;
  faceWidth: number;
  assignedFaceId: string | null;
  trajectory: VideoTrajectoryKeyframe[];
}

const BENGALI_DIGITS = ['০', '১', '২', '৩', '৪', '৫', '৬', '৭', '৮', '৯'];
function toBengaliNumber(n: number): string {
  return String(n)
    .split('')
    .map((ch) => BENGALI_DIGITS[Number(ch)] ?? ch)
    .join('');
}

function getFaceBounds(landmarks: Array<{ x: number; y: number; z?: number }>) {
  let minX = 1, minY = 1, maxX = 0, maxY = 0;
  for (const lm of landmarks) {
    if (lm.x < minX) minX = lm.x;
    if (lm.x > maxX) maxX = lm.x;
    if (lm.y < minY) minY = lm.y;
    if (lm.y > maxY) maxY = lm.y;
  }
  const cx = (minX + maxX) * 0.5;
  const cy = (minY + maxY) * 0.5;
  const w = Math.max(0.06, maxX - minX);
  const h = Math.max(0.06, maxY - minY);
  return { cx, cy, w, h, minX, minY, maxX, maxY };
}

function estimateYaw(landmarks: Array<{ x: number; y: number }>): number {
  const noseX = landmarks[4]?.x ?? 0.5;
  const leftEdgeX = landmarks[234]?.x ?? 0.3;
  const rightEdgeX = landmarks[454]?.x ?? 0.7;
  const span = Math.max(0.04, Math.abs(rightEdgeX - leftEdgeX));
  const midX = (leftEdgeX + rightEdgeX) * 0.5;
  return Math.max(-1, Math.min(1, ((noseX - midX) / (span * 0.5)) * 1.4));
}

function seekVideoTo(video: HTMLVideoElement, timeSec: number): Promise<void> {
  return new Promise((resolve) => {
    const clamped = Math.max(0, Math.min(video.duration || 0, timeSec));
    if (Math.abs(video.currentTime - clamped) < 0.02) {
      resolve();
      return;
    }
    const onSeeked = () => {
      video.removeEventListener('seeked', onSeeked);
      resolve();
    };
    video.addEventListener('seeked', onSeeked, { once: true });
    video.currentTime = clamped;
  });
}

function cropFaceThumbnail(
  sourceCanvas: HTMLCanvasElement,
  bounds: ReturnType<typeof getFaceBounds>,
): string {
  const thumbCanvas = document.createElement('canvas');
  thumbCanvas.width = 128;
  thumbCanvas.height = 128;
  const ctx = thumbCanvas.getContext('2d');
  if (!ctx) return '';

  const pad = Math.max(bounds.w, bounds.h) * 0.35;
  const sx = Math.max(0, (bounds.cx - bounds.w * 0.5 - pad) * sourceCanvas.width);
  const sy = Math.max(0, (bounds.cy - bounds.h * 0.5 - pad) * sourceCanvas.height);
  const sw = Math.min(sourceCanvas.width - sx, (bounds.w + pad * 2) * sourceCanvas.width);
  const sh = Math.min(sourceCanvas.height - sy, (bounds.h + pad * 2) * sourceCanvas.height);
  const side = Math.max(24, Math.min(sw, sh));

  ctx.drawImage(sourceCanvas, sx, sy, side, side, 0, 0, 128, 128);
  return thumbCanvas.toDataURL('image/jpeg', 0.85);
}

const videoRoiCanvas = typeof document !== 'undefined' ? document.createElement('canvas') : null;
if (videoRoiCanvas) {
  videoRoiCanvas.width = 256;
  videoRoiCanvas.height = 256;
}

function detectClustersInRoi(
  sourceCanvas: HTMLCanvasElement,
  imageLm: FaceLandmarker,
  anchors: Array<{ cx: number; cy: number; faceWidth: number }>,
): { faceLandmarks: NormalizedLandmark[][]; faceBlendshapes?: Classifications[] } {
  if (!videoRoiCanvas || anchors.length === 0) return { faceLandmarks: [] };
  const rCtx = videoRoiCanvas.getContext('2d');
  if (!rCtx) return { faceLandmarks: [] };

  const recoveredFaces: NormalizedLandmark[][] = [];
  const recoveredShapes: Classifications[] = [];

  for (const a of anchors) {
    const boxSize = Math.min(0.95, Math.max(0.30, a.faceWidth * 2.4));
    const sxNorm = Math.max(0, Math.min(1 - boxSize, a.cx - boxSize * 0.5));
    const syNorm = Math.max(0, Math.min(1 - boxSize, a.cy - boxSize * 0.5));
    const sx = sxNorm * sourceCanvas.width;
    const sy = syNorm * sourceCanvas.height;
    const sw = Math.max(32, boxSize * sourceCanvas.width);
    const sh = Math.max(32, boxSize * sourceCanvas.height);

    rCtx.save();
    rCtx.clearRect(0, 0, 256, 256);
    rCtx.filter = 'brightness(1.12) contrast(1.16)';
    rCtx.drawImage(sourceCanvas, sx, sy, sw, sh, 0, 0, 256, 256);
    rCtx.restore();

    const roiRes = imageLm.detect(videoRoiCanvas);
    if (roiRes.faceLandmarks.length > 0) {
      recoveredFaces.push(
        roiRes.faceLandmarks[0].map((lm) => ({
          x: Math.min(1, Math.max(0, sxNorm + lm.x * boxSize)),
          y: Math.min(1, Math.max(0, syNorm + lm.y * boxSize)),
          z: lm.z * boxSize,
          visibility: lm.visibility,
        })),
      );
      if (roiRes.faceBlendshapes?.[0]) {
        recoveredShapes.push(roiRes.faceBlendshapes[0]);
      }
    }
  }
  return { faceLandmarks: recoveredFaces, faceBlendshapes: recoveredShapes };
}

export async function scanVideoForPeople(
  video: HTMLVideoElement,
  imageLandmarker: FaceLandmarker,
  onProgress?: (pct: number) => void,
): Promise<DetectedVideoPerson[]> {
  const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 1;
  // Dense enough (~4 fps, capped at 56 keyframes) to capture trajectory before & after side head turns
  const sampleCount = Math.min(56, Math.max(14, Math.ceil(duration * 4)));
  const scanCanvas = document.createElement('canvas');
  const scale = Math.min(1, 512 / Math.max(1, video.videoWidth, video.videoHeight));
  scanCanvas.width = Math.max(160, Math.round(video.videoWidth * scale));
  scanCanvas.height = Math.max(160, Math.round(video.videoHeight * scale));
  const scanCtx = scanCanvas.getContext('2d');
  if (!scanCtx) return [];

  const clusters: Array<DetectedVideoPerson & { bestArea: number; bestCenteredScore: number }> = [];

  for (let i = 0; i < sampleCount; i++) {
    const frac = sampleCount === 1 ? 0 : (i / (sampleCount - 1)) * 0.96 + 0.02;
    const t = duration * frac;
    const timeMs = Math.round(t * 1000);
    await seekVideoTo(video, t);
    scanCtx.drawImage(video, 0, 0, scanCanvas.width, scanCanvas.height);

    let res: { faceLandmarks: NormalizedLandmark[][]; faceBlendshapes?: Classifications[] } =
      imageLandmarker.detect(scanCanvas);

    if (res.faceLandmarks.length === 0 && clusters.length > 0) {
      res = detectClustersInRoi(
        scanCanvas,
        imageLandmarker,
        clusters.map((c) => {
          const lastKf = c.trajectory[c.trajectory.length - 1];
          return {
            cx: lastKf?.cx ?? c.anchorX,
            cy: lastKf?.cy ?? c.anchorY,
            faceWidth: lastKf?.faceWidth ?? c.faceWidth,
          };
        }),
      );
    }

    for (let fIdx = 0; fIdx < res.faceLandmarks.length; fIdx++) {
      const faceLms = res.faceLandmarks[fIdx];
      const b = getFaceBounds(faceLms);
      const area = b.w * b.h;
      const yaw = estimateYaw(faceLms);
      const centeredScore = area * (1.15 - Math.min(0.8, Math.abs(yaw)));
      const jawOpen =
        res.faceBlendshapes?.[fIdx]?.categories?.find((c) => c.categoryName === 'jawOpen')?.score ?? 0;

      const keyframe: VideoTrajectoryKeyframe = {
        timeMs,
        cx: b.cx,
        cy: b.cy,
        faceWidth: b.w,
        yaw,
        jawOpen,
        landmarks: faceLms.map((lm) => ({ x: lm.x, y: lm.y, z: lm.z })),
      };

      let matchedCluster: (typeof clusters)[number] | null = null;
      let bestDist = Infinity;

      for (const cluster of clusters) {
        const lastKf = cluster.trajectory[cluster.trajectory.length - 1];
        const refX = lastKf?.cx ?? cluster.anchorX;
        const refY = lastKf?.cy ?? cluster.anchorY;
        const dist = Math.hypot(b.cx - refX, b.cy - refY);
        const threshold = Math.max(0.18, Math.max(cluster.faceWidth, b.w) * 1.25);
        if (dist < threshold && dist < bestDist) {
          bestDist = dist;
          matchedCluster = cluster;
        }
      }

      if (matchedCluster) {
        matchedCluster.anchorX = matchedCluster.anchorX * 0.7 + b.cx * 0.3;
        matchedCluster.anchorY = matchedCluster.anchorY * 0.7 + b.cy * 0.3;
        matchedCluster.faceWidth = Math.max(matchedCluster.faceWidth, b.w);
        matchedCluster.trajectory.push(keyframe);
        if (centeredScore > matchedCluster.bestCenteredScore) {
          matchedCluster.bestCenteredScore = centeredScore;
          matchedCluster.bestArea = area;
          matchedCluster.thumbnailUrl = cropFaceThumbnail(scanCanvas, b);
        }
      } else if (clusters.length < 5) {
        const idx = clusters.length + 1;
        clusters.push({
          id: `person_${idx}`,
          index: idx,
          labelEn: `Person ${idx}`,
          labelBn: `ব্যক্তি ${toBengaliNumber(idx)}`,
          thumbnailUrl: cropFaceThumbnail(scanCanvas, b),
          anchorX: b.cx,
          anchorY: b.cy,
          faceWidth: b.w,
          assignedFaceId: null,
          trajectory: [keyframe],
          bestArea: area,
          bestCenteredScore: centeredScore,
        });
      }
    }
    onProgress?.(Math.round(((i + 1) / sampleCount) * 100));
  }

  await seekVideoTo(video, 0);
  return clusters.map(({ bestArea: _area, bestCenteredScore: _cs, ...rest }) => {
    rest.trajectory.sort((a, b) => a.timeMs - b.timeMs);
    return rest;
  });
}

/**
 * Lightweight 8x8 zero-mean luminance head-patch tracker that locks onto the turned head
 * silhouette (hair, jaw, ear, cheek) even when frontal facial landmarks are occluded.
 */
export class ProfileHeadTracker {
  private lumCanvas: HTMLCanvasElement | null = null;
  private lumCtx: CanvasRenderingContext2D | null = null;
  private readonly gridW = 112;
  private readonly gridH = 112;
  private lumBuf = new Float32Array(112 * 112);
  private templates = new Map<string, { patch: Float32Array; mean: number; std: number; boxNorm: number }>();

  constructor() {
    if (typeof document !== 'undefined') {
      this.lumCanvas = document.createElement('canvas');
      this.lumCanvas.width = this.gridW;
      this.lumCanvas.height = this.gridH;
      this.lumCtx = this.lumCanvas.getContext('2d', { willReadFrequently: true });
    }
  }

  prepareFrame(sourceCanvas: HTMLCanvasElement): boolean {
    if (!this.lumCtx) return false;
    this.lumCtx.drawImage(sourceCanvas, 0, 0, this.gridW, this.gridH);
    const data = this.lumCtx.getImageData(0, 0, this.gridW, this.gridH).data;
    for (let i = 0, p = 0; i < data.length; i += 4, p++) {
      this.lumBuf[p] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    }
    return true;
  }

  private samplePatch(cxNorm: number, cyNorm: number, boxNorm: number, out: Float32Array): { mean: number; std: number } {
    const dim = 8;
    const half = boxNorm * 0.5;
    let sum = 0;
    for (let gy = 0; gy < dim; gy++) {
      const ny = cyNorm - half + ((gy + 0.5) / dim) * boxNorm;
      const py = Math.max(0, Math.min(this.gridH - 1, Math.round(ny * (this.gridH - 1))));
      for (let gx = 0; gx < dim; gx++) {
        const nx = cxNorm - half + ((gx + 0.5) / dim) * boxNorm;
        const px = Math.max(0, Math.min(this.gridW - 1, Math.round(nx * (this.gridW - 1))));
        const v = this.lumBuf[py * this.gridW + px];
        out[gy * dim + gx] = v;
        sum += v;
      }
    }
    const mean = sum / 64;
    let varSum = 0;
    for (let i = 0; i < 64; i++) {
      const d = out[i] - mean;
      varSum += d * d;
    }
    return { mean, std: Math.max(4, Math.sqrt(varSum / 64)) };
  }

  captureTemplate(personId: string, cxNorm: number, cyNorm: number, faceWidthNorm: number) {
    const boxNorm = Math.min(0.65, Math.max(0.14, faceWidthNorm * 1.45));
    const patch = new Float32Array(64);
    const stats = this.samplePatch(cxNorm, cyNorm, boxNorm, patch);
    this.templates.set(personId, { patch, mean: stats.mean, std: stats.std, boxNorm });
  }

  locateHead(
    personId: string,
    predCx: number,
    predCy: number,
  ): { cx: number; cy: number; confidence: number } {
    const tpl = this.templates.get(personId);
    if (!tpl) return { cx: predCx, cy: predCy, confidence: 0 };

    const candidate = new Float32Array(64);
    let bestCx = predCx;
    let bestCy = predCy;
    let bestNcc = -1;

    const evalAt = (tx: number, ty: number) => {
      const cx = Math.max(0.05, Math.min(0.95, tx));
      const cy = Math.max(0.05, Math.min(0.95, ty));
      const stats = this.samplePatch(cx, cy, tpl.boxNorm, candidate);
      let cov = 0;
      for (let i = 0; i < 64; i++) {
        cov += (candidate[i] - stats.mean) * (tpl.patch[i] - tpl.mean);
      }
      const ncc = cov / (64 * stats.std * tpl.std);
      if (ncc > bestNcc) {
        bestNcc = ncc;
        bestCx = cx;
        bestCy = cy;
      }
    };

    // Coarse search around trajectory prediction (±0.088 normalized)
    const radius = 0.088;
    const coarseStep = 0.022;
    for (let dy = -radius; dy <= radius + 0.001; dy += coarseStep) {
      for (let dx = -radius; dx <= radius + 0.001; dx += coarseStep) {
        evalAt(predCx + dx, predCy + dy);
      }
    }

    // Fine refinement around best coarse match
    const refineOriginX = bestCx;
    const refineOriginY = bestCy;
    for (let dy = -0.016; dy <= 0.017; dy += 0.008) {
      for (let dx = -0.016; dx <= 0.017; dx += 0.008) {
        evalAt(refineOriginX + dx, refineOriginY + dy);
      }
    }

    return { cx: bestCx, cy: bestCy, confidence: Math.max(0, bestNcc) };
  }

  clear() {
    this.templates.clear();
  }
}

export interface ResolvedVideoFaceTarget {
  personId: string;
  assignedFaceId: string | null;
  smoothedLandmarks: Array<{ x: number; y: number; z: number }>;
  jawOpen: number;
  opacityScale: number;
}

function findBracketingKeyframes(
  trajectory: VideoTrajectoryKeyframe[],
  liveSample: VideoTrajectoryKeyframe | undefined,
  timeMs: number,
): { prev: VideoTrajectoryKeyframe | null; next: VideoTrajectoryKeyframe | null } {
  let prev: VideoTrajectoryKeyframe | null = null;
  let next: VideoTrajectoryKeyframe | null = null;

  for (const kf of trajectory) {
    if (kf.timeMs <= timeMs) {
      prev = kf;
    } else if (!next) {
      next = kf;
      break;
    }
  }

  if (liveSample && liveSample.timeMs <= timeMs) {
    if (!prev || liveSample.timeMs >= prev.timeMs) {
      prev = liveSample;
    }
  }
  return { prev, next };
}

function reconstructFromTrajectory(
  person: DetectedVideoPerson,
  prev: VideoTrajectoryKeyframe | null,
  next: VideoTrajectoryKeyframe | null,
  timeMs: number,
  headTracker: ProfileHeadTracker,
  lastAnchor: { x: number; y: number } | undefined,
): { landmarks: Array<{ x: number; y: number; z: number }>; jawOpen: number; cx: number; cy: number; opacityScale: number } | null {
  if (!prev && !next) return null;

  // Case A: Bracketed between previous and next keyframes (3D trajectory interpolation)
  if (prev && next && next.timeMs > prev.timeMs) {
    const gapMs = next.timeMs - prev.timeMs;
    const rawAlpha = Math.max(0, Math.min(1, (timeMs - prev.timeMs) / gapMs));
    const alpha = rawAlpha * rawAlpha * (3 - 2 * rawAlpha); // smoothstep for natural head ease

    const predCx = prev.cx + (next.cx - prev.cx) * alpha;
    const predCy = prev.cy + (next.cy - prev.cy) * alpha;

    const tracked = headTracker.locateHead(person.id, predCx, predCy);
    const trackBlend = tracked.confidence > 0.55 ? 0.65 : tracked.confidence > 0.35 ? 0.35 : 0;
    const finalCx = predCx + (tracked.cx - predCx) * trackBlend;
    const finalCy = predCy + (tracked.cy - predCy) * trackBlend;

    // Slight profile compression when turning deeply sideways mid-gap
    const turnArc = Math.sin(Math.PI * rawAlpha);
    const turningSideways =
      (prev.yaw * next.yaw >= 0 && (Math.abs(prev.yaw) > 0.22 || Math.abs(next.yaw) > 0.22)) ||
      Math.abs(next.yaw - prev.yaw) > 0.45;
    const xProfileScale = turningSideways ? 1.0 - 0.16 * turnArc : 1.0;

    const count = Math.min(prev.landmarks.length, next.landmarks.length);
    const landmarks: Array<{ x: number; y: number; z: number }> = new Array(count);
    for (let i = 0; i < count; i++) {
      const pRelX = prev.landmarks[i].x - prev.cx;
      const pRelY = prev.landmarks[i].y - prev.cy;
      const nRelX = next.landmarks[i].x - next.cx;
      const nRelY = next.landmarks[i].y - next.cy;
      const rx = (pRelX + (nRelX - pRelX) * alpha) * xProfileScale;
      const ry = pRelY + (nRelY - pRelY) * alpha;
      const rz = prev.landmarks[i].z + (next.landmarks[i].z - prev.landmarks[i].z) * alpha;
      landmarks[i] = {
        x: Math.max(0, Math.min(1, finalCx + rx)),
        y: Math.max(0, Math.min(1, finalCy + ry)),
        z: rz,
      };
    }

    const jawOpen = prev.jawOpen + (next.jawOpen - prev.jawOpen) * alpha;
    // Maintain 100% swap opacity across normal side-turn gaps (up to 6s); only soften extreme missing stretches
    const opacityScale =
      gapMs <= 6000 ? 1.0 : Math.max(0.25, 1.0 - (Math.min(prev ? timeMs - prev.timeMs : 0, next.timeMs - timeMs) - 3000) / 3000);

    return { landmarks, jawOpen, cx: finalCx, cy: finalCy, opacityScale };
  }

  // Case B: One-sided boundary (very beginning or very end of video)
  const ref = prev ?? next!;
  const distMs = Math.abs(timeMs - ref.timeMs);
  if (distMs > 3000) return null;

  const baseCx = lastAnchor?.x ?? ref.cx;
  const baseCy = lastAnchor?.y ?? ref.cy;
  const tracked = headTracker.locateHead(person.id, baseCx, baseCy);
  const useTracked = tracked.confidence > 0.45;
  const finalCx = useTracked ? baseCx * 0.25 + tracked.cx * 0.75 : ref.cx;
  const finalCy = useTracked ? baseCy * 0.25 + tracked.cy * 0.75 : ref.cy;

  const landmarks = ref.landmarks.map((lm) => ({
    x: Math.max(0, Math.min(1, finalCx + (lm.x - ref.cx))),
    y: Math.max(0, Math.min(1, finalCy + (lm.y - ref.cy))),
    z: lm.z,
  }));

  const holdLimit = useTracked ? 2600 : 800;
  const opacityScale = distMs <= 900 ? 1.0 : Math.max(0, 1.0 - (distMs - 900) / Math.max(400, holdLimit - 900));
  if (opacityScale <= 0.02) return null;

  return { landmarks, jawOpen: ref.jawOpen, cx: finalCx, cy: finalCy, opacityScale };
}

export function resolveVideoFacesWithTrajectory(
  videoLm: FaceLandmarker | null,
  imageLm: FaceLandmarker | null,
  sourceCanvas: HTMLCanvasElement,
  detectCanvas: HTMLCanvasElement,
  people: DetectedVideoPerson[],
  dynamicAnchors: Map<string, { x: number; y: number }>,
  smoothers: Map<string, LandmarkOneEuroSmoother>,
  liveSnapshots: Map<string, VideoTrajectoryKeyframe>,
  headTracker: ProfileHeadTracker,
  videoTimeMs: number,
  monotonicMs: number,
  smoothingStrength: number,
): ResolvedVideoFaceTarget[] {
  if (people.length === 0) return [];

  headTracker.prepareFrame(sourceCanvas);

  // Build trajectory-aware anchors for ROI fallback
  const predictedAnchors = people.map((p) => {
    const { prev, next } = findBracketingKeyframes(p.trajectory, liveSnapshots.get(p.id), videoTimeMs);
    if (prev && next && next.timeMs > prev.timeMs) {
      const a = Math.max(0, Math.min(1, (videoTimeMs - prev.timeMs) / (next.timeMs - prev.timeMs)));
      return {
        cx: prev.cx + (next.cx - prev.cx) * a,
        cy: prev.cy + (next.cy - prev.cy) * a,
        faceWidth: p.faceWidth,
      };
    }
    const anchor = dynamicAnchors.get(p.id);
    const ref = prev ?? next;
    return {
      cx: anchor?.x ?? ref?.cx ?? p.anchorX,
      cy: anchor?.y ?? ref?.cy ?? p.anchorY,
      faceWidth: p.faceWidth,
    };
  });

  // 1) Fast downscaled full-frame pass -> IMAGE -> shadow-lifted trajectory ROI
  let res: { faceLandmarks: NormalizedLandmark[][]; faceBlendshapes?: Classifications[] } = {
    faceLandmarks: [],
  };
  const scale = Math.min(1, 480 / Math.max(1, sourceCanvas.width, sourceCanvas.height));
  const dw = Math.max(128, Math.round(sourceCanvas.width * scale));
  const dh = Math.max(128, Math.round(sourceCanvas.height * scale));
  if (detectCanvas.width !== dw || detectCanvas.height !== dh) {
    detectCanvas.width = dw;
    detectCanvas.height = dh;
  }
  const dCtx = detectCanvas.getContext('2d');
  if (dCtx) {
    dCtx.drawImage(sourceCanvas, 0, 0, dw, dh);
    if (videoLm) {
      const vRes = videoLm.detectForVideo(detectCanvas, monotonicMs);
      if (vRes.faceLandmarks.length > 0) res = vRes;
    }
    if (res.faceLandmarks.length === 0 && imageLm) {
      const iRes = imageLm.detect(detectCanvas);
      if (iRes.faceLandmarks.length > 0) res = iRes;
    }
  }
  if (res.faceLandmarks.length === 0 && imageLm) {
    res = detectClustersInRoi(sourceCanvas, imageLm, predictedAnchors);
  }

  // 2) Match detected faces to tracked people
  const usedPeople = new Set<string>();
  const outputs: ResolvedVideoFaceTarget[] = [];

  for (let idx = 0; idx < res.faceLandmarks.length; idx++) {
    const lms = res.faceLandmarks[idx];
    const bounds = getFaceBounds(lms);
    const jawOpen =
      res.faceBlendshapes?.[idx]?.categories?.find((c) => c.categoryName === 'jawOpen')?.score ?? 0;

    let bestPerson: DetectedVideoPerson | null = null;
    let bestDist = Infinity;

    for (let pIdx = 0; pIdx < people.length; pIdx++) {
      const p = people[pIdx];
      if (usedPeople.has(p.id)) continue;
      const pred = predictedAnchors[pIdx];
      const dist = Math.hypot(bounds.cx - pred.cx, bounds.cy - pred.cy);
      if (dist < 0.36 && dist < bestDist) {
        bestDist = dist;
        bestPerson = p;
      }
    }

    if (!bestPerson) {
      for (const p of people) {
        if (!usedPeople.has(p.id)) {
          bestPerson = p;
          break;
        }
      }
    }
    if (!bestPerson) continue;

    usedPeople.add(bestPerson.id);
    dynamicAnchors.set(bestPerson.id, { x: bounds.cx, y: bounds.cy });
    headTracker.captureTemplate(bestPerson.id, bounds.cx, bounds.cy, bounds.w);

    const kf: VideoTrajectoryKeyframe = {
      timeMs: videoTimeMs,
      cx: bounds.cx,
      cy: bounds.cy,
      faceWidth: bounds.w,
      yaw: estimateYaw(lms),
      jawOpen,
      landmarks: lms.map((pt) => ({ x: pt.x, y: pt.y, z: pt.z })),
    };
    liveSnapshots.set(bestPerson.id, kf);

    let smoother = smoothers.get(bestPerson.id);
    if (!smoother) {
      smoother = new LandmarkOneEuroSmoother();
      smoothers.set(bestPerson.id, smoother);
    }
    const smoothedLandmarks = smoother.smooth(lms, monotonicMs, smoothingStrength);

    outputs.push({
      personId: bestPerson.id,
      assignedFaceId: bestPerson.assignedFaceId,
      smoothedLandmarks,
      jawOpen,
      opacityScale: 1.0,
    });
  }

  // 3) For ANY person missed on this frame (steep side turn / backlit), reconstruct seamlessly via
  //    bidirectional 3D trajectory (prev <-> next) + Profile Head Tracker
  for (const person of people) {
    if (usedPeople.has(person.id) || !person.assignedFaceId) continue;

    const { prev, next } = findBracketingKeyframes(
      person.trajectory,
      liveSnapshots.get(person.id),
      videoTimeMs,
    );
    const rec = reconstructFromTrajectory(
      person,
      prev,
      next,
      videoTimeMs,
      headTracker,
      dynamicAnchors.get(person.id),
    );
    if (!rec) continue;

    dynamicAnchors.set(person.id, { x: rec.cx, y: rec.cy });
    let smoother = smoothers.get(person.id);
    if (!smoother) {
      smoother = new LandmarkOneEuroSmoother();
      smoothers.set(person.id, smoother);
    }
    const smoothedLandmarks = smoother.smooth(rec.landmarks, monotonicMs, smoothingStrength);

    outputs.push({
      personId: person.id,
      assignedFaceId: person.assignedFaceId,
      smoothedLandmarks,
      jawOpen: rec.jawOpen,
      opacityScale: rec.opacityScale,
    });
  }

  return outputs;
}

