import type { FaceLandmarker, NormalizedLandmark, Classifications } from '@mediapipe/tasks-vision';
import { LandmarkOneEuroSmoother } from './oneEuroFilter';

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
}

const BENGALI_DIGITS = ['০', '১', '২', '৩', '৪', '৫', '৬', '৭', '৮', '৯'];
function toBengaliNumber(n: number): string {
  return String(n)
    .split('')
    .map((ch) => BENGALI_DIGITS[Number(ch)] ?? ch)
    .join('');
}

function getFaceBounds(landmarks: NormalizedLandmark[]) {
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

export async function scanVideoForPeople(
  video: HTMLVideoElement,
  imageLandmarker: FaceLandmarker,
  onProgress?: (pct: number) => void,
): Promise<DetectedVideoPerson[]> {
  const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 1;
  const sampleCount = 9;
  const scanCanvas = document.createElement('canvas');
  const scale = Math.min(1, 640 / Math.max(1, video.videoWidth, video.videoHeight));
  scanCanvas.width = Math.max(160, Math.round(video.videoWidth * scale));
  scanCanvas.height = Math.max(160, Math.round(video.videoHeight * scale));
  const scanCtx = scanCanvas.getContext('2d');
  if (!scanCtx) return [];

  const clusters: Array<DetectedVideoPerson & { bestArea: number }> = [];

  for (let i = 0; i < sampleCount; i++) {
    const t = duration * (0.04 + (i / (sampleCount - 1)) * 0.90);
    await seekVideoTo(video, t);
    scanCtx.drawImage(video, 0, 0, scanCanvas.width, scanCanvas.height);

    const res = imageLandmarker.detect(scanCanvas);
    for (const faceLms of res.faceLandmarks) {
      const b = getFaceBounds(faceLms);
      const area = b.w * b.h;
      let matchedCluster: (typeof clusters)[number] | null = null;
      let bestDist = Infinity;

      for (const cluster of clusters) {
        const dist = Math.hypot(b.cx - cluster.anchorX, b.cy - cluster.anchorY);
        const threshold = Math.max(0.16, Math.max(cluster.faceWidth, b.w) * 1.1);
        if (dist < threshold && dist < bestDist) {
          bestDist = dist;
          matchedCluster = cluster;
        }
      }

      if (matchedCluster) {
        matchedCluster.anchorX = matchedCluster.anchorX * 0.6 + b.cx * 0.4;
        matchedCluster.anchorY = matchedCluster.anchorY * 0.6 + b.cy * 0.4;
        matchedCluster.faceWidth = Math.max(matchedCluster.faceWidth, b.w);
        if (area > matchedCluster.bestArea) {
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
          bestArea: area,
        });
      }
    }
    onProgress?.(Math.round(((i + 1) / sampleCount) * 100));
  }

  await seekVideoTo(video, 0);
  return clusters.map(({ bestArea: _area, ...rest }) => rest);
}

export interface MatchedVideoFace {
  personId: string;
  assignedFaceId: string | null;
  smoothedLandmarks: Array<{ x: number; y: number; z: number }>;
  jawOpen: number;
}

export function matchFrameFacesToPeople(
  frameFaces: NormalizedLandmark[][],
  blendshapesList: Classifications[] | undefined,
  people: DetectedVideoPerson[],
  dynamicAnchors: Map<string, { x: number; y: number }>,
  smoothers: Map<string, LandmarkOneEuroSmoother>,
  timestampMs: number,
  smoothingStrength: number,
): MatchedVideoFace[] {
  if (frameFaces.length === 0 || people.length === 0) return [];

  const detectedWithBounds = frameFaces.map((lms, idx) => ({
    lms,
    bounds: getFaceBounds(lms),
    jawOpen: blendshapesList?.[idx]?.categories?.find((c) => c.categoryName === 'jawOpen')?.score ?? 0,
  }));

  const usedPeople = new Set<string>();
  const matches: MatchedVideoFace[] = [];

  for (const det of detectedWithBounds) {
    let bestPerson: DetectedVideoPerson | null = null;
    let bestDist = Infinity;

    for (const p of people) {
      if (usedPeople.has(p.id)) continue;
      const anchor = dynamicAnchors.get(p.id) ?? { x: p.anchorX, y: p.anchorY };
      const dist = Math.hypot(det.bounds.cx - anchor.x, det.bounds.cy - anchor.y);
      if (dist < 0.34 && dist < bestDist) {
        bestDist = dist;
        bestPerson = p;
      }
    }

    if (!bestPerson) {
      // Fallback: assign first unused person if only 1 person or closest unassigned
      for (const p of people) {
        if (!usedPeople.has(p.id)) {
          bestPerson = p;
          break;
        }
      }
    }

    if (!bestPerson) continue;
    usedPeople.add(bestPerson.id);
    dynamicAnchors.set(bestPerson.id, { x: det.bounds.cx, y: det.bounds.cy });

    let smoother = smoothers.get(bestPerson.id);
    if (!smoother) {
      smoother = new LandmarkOneEuroSmoother();
      smoothers.set(bestPerson.id, smoother);
    }
    const smoothedLandmarks = smoother.smooth(det.lms, timestampMs, smoothingStrength);

    matches.push({
      personId: bestPerson.id,
      assignedFaceId: bestPerson.assignedFaceId,
      smoothedLandmarks,
      jawOpen: det.jawOpen,
    });
  }

  return matches;
}
