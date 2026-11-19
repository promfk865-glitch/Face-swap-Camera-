import React, { useEffect, useRef, useState } from 'react';
import {
  FaceLandmarker,
  FilesetResolver,
  type NormalizedLandmark,
} from '@mediapipe/tasks-vision';
import {
  ArrowLeft,
  Camera,
  Check,
  Download,
  ImagePlus,
  Loader2,
  Pause,
  Play,
  Users,
} from 'lucide-react';
import type { DemoFaceMeta } from '../data/demoFaces';
import { buildFaceMeshTopology } from '../utils/faceMeshTriangulation';
import { FaceWarpGlRenderer, type FaceSwapDrawTarget } from '../utils/faceWarpGl';
import { LandmarkOneEuroSmoother } from '../utils/oneEuroFilter';
import {
  matchFrameFacesToPeople,
  scanVideoForPeople,
  type DetectedVideoPerson,
} from '../utils/videoFaceTracker';

export interface CharacterAsset extends DemoFaceMeta {
  image: HTMLImageElement;
  landmarks: NormalizedLandmark[];
}

interface VideoSwapStudioProps {
  videoUrl: string;
  faces: DemoFaceMeta[];
  loadedFacesMap: Map<string, CharacterAsset>;
  imageLandmarker: FaceLandmarker | null;
  opacity: number;
  smoothing: number;
  onUploadCharacterFace: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onSavedVideo: (msg: string) => void;
  onError: (msg: string) => void;
  onClose: () => void;
}

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = reader.result as string;
      const commaIdx = result.indexOf(',');
      resolve(commaIdx >= 0 ? result.slice(commaIdx + 1) : result);
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

async function createMultiFaceVideoLandmarker(): Promise<FaceLandmarker> {
  let vision;
  try {
    vision = await FilesetResolver.forVisionTasks('/mediapipe/wasm');
  } catch {
    vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm',
    );
  }
  const localModel = '/mediapipe/face_landmarker.task';
  const remoteModel =
    'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

  for (const modelUrl of [localModel, remoteModel]) {
    for (const delegate of ['GPU', 'CPU'] as const) {
      try {
        return await FaceLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: modelUrl, delegate },
          runningMode: 'VIDEO',
          numFaces: 5,
          outputFaceBlendshapes: true,
        });
      } catch {
        // Try next delegate/model
      }
    }
  }
  throw new Error('Unable to initialize multi-face video tracker');
}

async function createMultiFaceImageLandmarker(): Promise<FaceLandmarker> {
  let vision;
  try {
    vision = await FilesetResolver.forVisionTasks('/mediapipe/wasm');
  } catch {
    vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm',
    );
  }
  const localModel = '/mediapipe/face_landmarker.task';
  const remoteModel =
    'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

  for (const modelUrl of [localModel, remoteModel]) {
    for (const delegate of ['GPU', 'CPU'] as const) {
      try {
        return await FaceLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: modelUrl, delegate },
          runningMode: 'IMAGE',
          numFaces: 5,
        });
      } catch {
        // Try next
      }
    }
  }
  throw new Error('Unable to initialize multi-face image detector');
}

export const VideoSwapStudio: React.FC<VideoSwapStudioProps> = ({
  videoUrl,
  faces,
  loadedFacesMap,
  opacity,
  smoothing,
  onUploadCharacterFace,
  onSavedVideo,
  onError,
  onClose,
}) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const frameCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const uploadCharRef = useRef<HTMLInputElement | null>(null);
  const rendererRef = useRef<FaceWarpGlRenderer | null>(null);
  const videoLmRef = useRef<FaceLandmarker | null>(null);
  const imageLmRef = useRef<FaceLandmarker | null>(null);

  const smoothersRef = useRef<Map<string, LandmarkOneEuroSmoother>>(new Map());
  const anchorsRef = useRef<Map<string, { x: number; y: number }>>(new Map());
  const peopleRef = useRef<DetectedVideoPerson[]>([]);

  const [people, setPeople] = useState<DetectedVideoPerson[]>([]);
  const [selectedPersonId, setSelectedPersonId] = useState<string | null>(null);
  const [isScanning, setIsScanning] = useState(true);
  const [scanProgress, setScanProgress] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTimeSec, setCurrentTimeSec] = useState(0);
  const [durationSec, setDurationSec] = useState(0);
  const [isExporting, setIsExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState(0);

  peopleRef.current = people;

  const opacityRef = useRef(opacity);
  opacityRef.current = opacity;
  const smoothingRef = useRef(smoothing);
  smoothingRef.current = smoothing;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const targetCanvas = canvas;
    const renderer = new FaceWarpGlRenderer(targetCanvas);
    const topology = buildFaceMeshTopology(FaceLandmarker.FACE_LANDMARKS_TESSELATION);
    renderer.setTriangulation(topology.triangles, topology.vertexAlphas);
    rendererRef.current = renderer;

    let cancelled = false;
    async function setupAndScan() {
      const video = videoRef.current;
      if (!video) return;

      setIsScanning(true);
      try {
        await new Promise<void>((resolve, reject) => {
          const onReady = () => {
            cleanup();
            resolve();
          };
          const onErr = () => {
            cleanup();
            reject(new Error('Could not load video'));
          };
          const cleanup = () => {
            video.removeEventListener('loadeddata', onReady);
            video.removeEventListener('error', onErr);
          };
          if (video.readyState >= 2) {
            resolve();
            return;
          }
          video.addEventListener('loadeddata', onReady);
          video.addEventListener('error', onErr);
        });

        if (cancelled) return;
        setDurationSec(video.duration || 0);

        const maxSide = 960;
        const scale = Math.min(1, maxSide / Math.max(1, video.videoWidth, video.videoHeight));
        const renderW = Math.max(160, Math.round(video.videoWidth * scale));
        const renderH = Math.max(160, Math.round(video.videoHeight * scale));
        targetCanvas.width = renderW;
        targetCanvas.height = renderH;

        const fc = document.createElement('canvas');
        fc.width = renderW;
        fc.height = renderH;
        frameCanvasRef.current = fc;

        const [vLm, iLm] = await Promise.all([
          createMultiFaceVideoLandmarker(),
          createMultiFaceImageLandmarker(),
        ]);
        if (cancelled) {
          vLm.close();
          iLm.close();
          return;
        }
        videoLmRef.current = vLm;
        imageLmRef.current = iLm;

        const detected = await scanVideoForPeople(video, iLm, setScanProgress);
        if (cancelled) return;

        setPeople(detected);
        if (detected.length > 0) {
          setSelectedPersonId(detected[0].id);
        } else {
          onError('No faces detected in this video • এই ভিডিওতে কোনো মুখ পাওয়া যায়নি');
        }
      } catch {
        if (!cancelled) {
          onError('Could not process selected video • ভিডিওটি প্রসেস করা যায়নি');
        }
      } finally {
        if (!cancelled) {
          setIsScanning(false);
        }
      }
    }

    setupAndScan();

    return () => {
      cancelled = true;
      videoLmRef.current?.close();
      imageLmRef.current?.close();
    };
  }, [videoUrl]);

  const renderCurrentVideoFrame = (monotonicMs: number) => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const fc = frameCanvasRef.current;
    const renderer = rendererRef.current;
    const vLm = videoLmRef.current;
    if (!video || !canvas || !fc || !renderer || video.readyState < 2) return;

    const ctx = fc.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, fc.width, fc.height);

    const targets: FaceSwapDrawTarget[] = [];
    if (vLm && peopleRef.current.length > 0) {
      const res = vLm.detectForVideo(fc, monotonicMs);
      const matched = matchFrameFacesToPeople(
        res.faceLandmarks,
        res.faceBlendshapes,
        peopleRef.current,
        anchorsRef.current,
        smoothersRef.current,
        monotonicMs,
        smoothingRef.current,
      );

      for (const m of matched) {
        if (!m.assignedFaceId) continue;
        const asset = loadedFacesMap.get(m.assignedFaceId);
        if (!asset) continue;
        targets.push({
          personKey: m.personId,
          faceKey: asset.id,
          faceImage: asset.image,
          srcLandmarks: asset.landmarks,
          liveLandmarks: m.smoothedLandmarks,
          jawOpen: m.jawOpen,
        });
      }
    }

    renderer.renderMultiFaceFrame(fc, canvas.width, canvas.height, false, targets, opacityRef.current);
  };

  useEffect(() => {
    if (isScanning) return;
    let rafId = 0;
    let lastMonotonic = performance.now();
    let lastVideoTime = -1;

    const loop = () => {
      const video = videoRef.current;
      if (video && video.readyState >= 2) {
        if (video.currentTime !== lastVideoTime) {
          lastVideoTime = video.currentTime;
          setCurrentTimeSec(video.currentTime);
          const now = Math.max(performance.now(), lastMonotonic + 1);
          lastMonotonic = now;
          renderCurrentVideoFrame(now);
        }
        if (!video.paused && video.ended) {
          setIsPlaying(false);
        }
      }
      rafId = requestAnimationFrame(loop);
    };

    // Draw initial frame
    const initialTimer = window.setTimeout(() => {
      const now = Math.max(performance.now(), lastMonotonic + 1);
      lastMonotonic = now;
      renderCurrentVideoFrame(now);
    }, 60);

    rafId = requestAnimationFrame(loop);
    return () => {
      window.clearTimeout(initialTimer);
      cancelAnimationFrame(rafId);
    };
  }, [isScanning, people, opacity]);

  const handleAssignFace = (faceId: string | null) => {
    if (!selectedPersonId) return;
    setPeople((prev) =>
      prev.map((p) => (p.id === selectedPersonId ? { ...p, assignedFaceId: faceId } : p)),
    );
  };

  const togglePlay = async () => {
    const video = videoRef.current;
    if (!video || isScanning || isExporting) return;
    if (video.paused || video.ended) {
      if (video.ended) video.currentTime = 0;
      smoothersRef.current.clear();
      anchorsRef.current.clear();
      await video.play();
      setIsPlaying(true);
    } else {
      video.pause();
      setIsPlaying(false);
    }
  };

  const handleSeek = (val: number) => {
    const video = videoRef.current;
    if (!video || isExporting) return;
    smoothersRef.current.clear();
    anchorsRef.current.clear();
    video.currentTime = val;
    setCurrentTimeSec(val);
  };

  const handleExportVideo = async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || isScanning || isExporting) return;

    setIsExporting(true);
    setExportProgress(0);
    setIsPlaying(false);
    video.pause();
    video.currentTime = 0;
    smoothersRef.current.clear();
    anchorsRef.current.clear();

    await new Promise<void>((r) => {
      const onSeeked = () => {
        video.removeEventListener('seeked', onSeeked);
        r();
      };
      if (video.currentTime < 0.05) {
        r();
      } else {
        video.addEventListener('seeked', onSeeked, { once: true });
      }
    });

    const canvasStream = canvas.captureStream(30);
    const tracks = [...canvasStream.getVideoTracks()];

    // Preserve original video audio track when supported
    const videoAny = video as HTMLVideoElement & {
      captureStream?: () => MediaStream;
      mozCaptureStream?: () => MediaStream;
    };
    try {
      const origStream = videoAny.captureStream?.() ?? videoAny.mozCaptureStream?.();
      const audioTracks = origStream?.getAudioTracks() ?? [];
      if (audioTracks.length > 0) {
        tracks.push(audioTracks[0]);
      }
    } catch {
      // Proceed cleanly with video stream if browser restricts audio capture
    }

    const combined = new MediaStream(tracks);
    const mimeType = MediaRecorder.isTypeSupported('video/mp4')
      ? 'video/mp4'
      : MediaRecorder.isTypeSupported('video/webm;codecs=vp9,opus')
        ? 'video/webm;codecs=vp9,opus'
        : 'video/webm';

    const ext = mimeType.includes('mp4') ? 'mp4' : 'webm';
    const fileName = `FaceSwapVideo_${Date.now()}.${ext}`;
    const chunks: Blob[] = [];
    const recorder = new MediaRecorder(combined, { mimeType });
    recorder.ondataavailable = (ev) => {
      if (ev.data.size > 0) chunks.push(ev.data);
    };

    const progressTimer = window.setInterval(() => {
      if (video.duration > 0) {
        setExportProgress(Math.min(99, Math.round((video.currentTime / video.duration) * 100)));
      }
    }, 120);

    recorder.onstop = async () => {
      window.clearInterval(progressTimer);
      setExportProgress(100);
      const blob = new Blob(chunks, { type: mimeType });

      if (window.AndroidBridge?.saveVideoBase64) {
        try {
          const b64 = await blobToBase64(blob);
          const ok = window.AndroidBridge.saveVideoBase64(b64, fileName, mimeType);
          if (ok) {
            onSavedVideo('Swapped video saved to Movies/FaceSwapCam • ভিডিও গ্যালারিতে সেভ হয়েছে');
            setIsExporting(false);
            return;
          }
        } catch {
          // Fall through to browser download
        }
      }

      const file = new File([blob], fileName, { type: mimeType });
      if (navigator.canShare && navigator.canShare({ files: [file] })) {
        navigator.share({ files: [file], title: 'FaceSwap Cam' }).catch(() => {});
      }
      const dlUrl = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = dlUrl;
      a.download = fileName;
      a.click();
      setTimeout(() => URL.revokeObjectURL(dlUrl), 5000);
      onSavedVideo('Swapped video saved • ভিডিও সেভ হয়েছে');
      setIsExporting(false);
    };

    recorder.start(250);
    video.loop = false;

    const onEnded = () => {
      video.removeEventListener('ended', onEnded);
      setTimeout(() => {
        if (recorder.state !== 'inactive') recorder.stop();
      }, 150);
    };
    video.addEventListener('ended', onEnded, { once: true });

    try {
      await video.play();
    } catch {
      window.clearInterval(progressTimer);
      if (recorder.state !== 'inactive') recorder.stop();
      setIsExporting(false);
      onError('Could not play video for export • ভিডিও এক্সপোর্ট করা যায়নি');
    }
  };

  const selectedPerson = people.find((p) => p.id === selectedPersonId) ?? null;

  return (
    <div
      style={{
        position: 'absolute',
        inset: 0,
        zIndex: 35,
        background: '#090B10',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <video
        ref={videoRef}
        src={videoUrl}
        playsInline
        crossOrigin="anonymous"
        style={{
          position: 'fixed',
          top: -9999,
          left: -9999,
          width: 1,
          height: 1,
          opacity: 0,
          pointerEvents: 'none',
        }}
      />

      {/* Top Studio Header */}
      <header
        style={{
          padding: 'calc(env(safe-area-inset-top, 12px) + 8px) 14px 10px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          borderBottom: '1px solid rgba(255,255,255,0.08)',
          background: 'rgba(13, 17, 26, 0.94)',
        }}
      >
        <button
          onClick={onClose}
          disabled={isExporting}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            padding: '8px 12px',
            borderRadius: 12,
            background: 'rgba(255,255,255,0.06)',
            border: '1px solid rgba(255,255,255,0.12)',
            color: '#F8FAFC',
            fontSize: 12,
            fontWeight: 600,
            cursor: 'pointer',
          }}
        >
          <ArrowLeft size={16} />
          <span>Camera • ক্যামেরা</span>
        </button>

        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <Users size={16} color="#00E5C3" />
          <span style={{ fontSize: 13, fontWeight: 700 }}>Video Studio • ভিডিও স্টুডিও</span>
        </div>

        <button
          onClick={handleExportVideo}
          disabled={isScanning || isExporting || people.length === 0}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            padding: '8px 12px',
            borderRadius: 12,
            background: isExporting || isScanning ? 'rgba(0,229,195,0.2)' : '#00E5C3',
            color: isExporting || isScanning ? '#00E5C3' : '#090B10',
            border: 'none',
            fontSize: 12,
            fontWeight: 700,
            cursor: 'pointer',
          }}
        >
          <Download size={15} />
          <span>{isExporting ? `${exportProgress}%` : 'Save • সেভ'}</span>
        </button>
      </header>

      {/* Main Canvas Stage (Full Natural Aspect, Zero Zoom) */}
      <div style={{ position: 'relative', flex: 1, minHeight: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#05070B' }}>
        <canvas
          ref={canvasRef}
          style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block' }}
        />

        {(isScanning || isExporting) && (
          <div
            style={{
              position: 'absolute',
              inset: 0,
              background: 'rgba(9,11,16,0.78)',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 10,
            }}
          >
            <Loader2 size={32} color="#00E5C3" className="animate-spin" />
            <div style={{ fontSize: 14, fontWeight: 700, color: '#F8FAFC' }}>
              {isScanning
                ? `Scanning faces (${scanProgress}%) • মুখ খোঁজা হচ্ছে (${scanProgress}%)`
                : `Saving swapped video (${exportProgress}%) • ভিডিও সেভ হচ্ছে (${exportProgress}%)`}
            </div>
          </div>
        )}
      </div>

      {/* Playback Controls + Multi-Person Selector + Face Assignment Deck */}
      <div
        style={{
          background: '#111521',
          borderTop: '1px solid rgba(255,255,255,0.1)',
          padding: '10px 12px calc(env(safe-area-inset-bottom, 12px) + 12px)',
          display: 'flex',
          flexDirection: 'column',
          gap: 10,
        }}
      >
        {/* Scrubber */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <button
            onClick={togglePlay}
            disabled={isScanning || isExporting}
            style={{
              width: 40,
              height: 40,
              borderRadius: 12,
              border: '1px solid rgba(255,255,255,0.14)',
              background: 'rgba(255,255,255,0.06)',
              color: '#00E5C3',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
              flexShrink: 0,
            }}
          >
            {isPlaying ? <Pause size={18} /> : <Play size={18} />}
          </button>
          <input
            type="range"
            min={0}
            max={Math.max(0.1, durationSec)}
            step={0.05}
            value={currentTimeSec}
            onChange={(e) => handleSeek(Number(e.target.value))}
            style={{ flex: 1, accentColor: '#00E5C3' }}
          />
          <span style={{ fontSize: 11, color: '#9CA3B8', minWidth: 44, textAlign: 'right' }}>
            {currentTimeSec.toFixed(1)}s
          </span>
        </div>

        {/* Detected Faces Row */}
        {people.length > 0 && (
          <div>
            <div style={{ fontSize: 11, color: '#9CA3B8', marginBottom: 6 }}>
              1. Tap a face in video • ভিডিওর মুখ বেছে নিন ({people.length} found):
            </div>
            <div className="no-scrollbar" style={{ display: 'flex', gap: 8, overflowX: 'auto' }}>
              {people.map((person) => {
                const isSelected = person.id === selectedPersonId;
                const assignedMeta = person.assignedFaceId
                  ? faces.find((f) => f.id === person.assignedFaceId)
                  : null;
                return (
                  <button
                    key={person.id}
                    onClick={() => setSelectedPersonId(person.id)}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 8,
                      padding: '6px 10px',
                      borderRadius: 14,
                      border: isSelected ? '2px solid #00E5C3' : '1px solid rgba(255,255,255,0.12)',
                      background: isSelected ? 'rgba(0,229,195,0.14)' : 'rgba(255,255,255,0.04)',
                      color: '#F8FAFC',
                      cursor: 'pointer',
                      flexShrink: 0,
                    }}
                  >
                    {person.thumbnailUrl && (
                      <img
                        src={person.thumbnailUrl}
                        alt={person.labelEn}
                        style={{ width: 38, height: 38, borderRadius: 10, objectFit: 'cover' }}
                      />
                    )}
                    <div style={{ textAlign: 'left' }}>
                      <div style={{ fontSize: 11, fontWeight: 700 }}>{person.labelEn}</div>
                      <div style={{ fontSize: 10, color: '#9CA3B8' }}>{person.labelBn}</div>
                      <div style={{ fontSize: 10, color: '#00E5C3', fontWeight: 600 }}>
                        {assignedMeta ? `→ ${assignedMeta.nameEn}` : 'Original • আসল'}
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {/* Swap Target Face Carousel */}
        {selectedPerson && (
          <div>
            <div style={{ fontSize: 11, color: '#9CA3B8', marginBottom: 6 }}>
              2. Replace {selectedPerson.labelEn} ({selectedPerson.labelBn}) with • নতুন মুখ দিন:
            </div>
            <div className="no-scrollbar" style={{ display: 'flex', gap: 8, overflowX: 'auto' }}>
              <button
                onClick={() => handleAssignFace(null)}
                style={{
                  minWidth: 64,
                  height: 74,
                  borderRadius: 14,
                  background: 'rgba(20,24,36,0.9)',
                  border: selectedPerson.assignedFaceId === null ? '2px solid #00E5C3' : '1px solid rgba(255,255,255,0.12)',
                  color: '#F8FAFC',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 3,
                  cursor: 'pointer',
                  flexShrink: 0,
                }}
              >
                <Camera size={18} color={selectedPerson.assignedFaceId === null ? '#00E5C3' : '#9CA3B8'} />
                <span style={{ fontSize: 10, fontWeight: 600 }}>Original</span>
                <span style={{ fontSize: 9, color: '#9CA3B8' }}>আসল</span>
              </button>

              {faces.map((face) => {
                const active = selectedPerson.assignedFaceId === face.id;
                return (
                  <button
                    key={face.id}
                    onClick={() => handleAssignFace(face.id)}
                    style={{
                      position: 'relative',
                      minWidth: 66,
                      height: 74,
                      borderRadius: 14,
                      padding: 4,
                      background: active ? 'rgba(0,229,195,0.16)' : 'rgba(20,24,36,0.9)',
                      border: active ? '2px solid #00E5C3' : '1px solid rgba(255,255,255,0.12)',
                      color: '#F8FAFC',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      cursor: 'pointer',
                      flexShrink: 0,
                    }}
                  >
                    <img
                      src={face.src}
                      alt={face.nameEn}
                      style={{ width: 40, height: 40, borderRadius: 10, objectFit: 'cover' }}
                    />
                    <span style={{ fontSize: 9, fontWeight: 600 }}>{face.nameEn}</span>
                    <span style={{ fontSize: 8, color: '#9CA3B8' }}>{face.nameBn}</span>
                    {active && (
                      <span
                        style={{
                          position: 'absolute',
                          top: 3,
                          right: 3,
                          background: '#00E5C3',
                          color: '#090B10',
                          borderRadius: '50%',
                          width: 16,
                          height: 16,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                        }}
                      >
                        <Check size={11} />
                      </span>
                    )}
                  </button>
                );
              })}

              <button
                onClick={() => uploadCharRef.current?.click()}
                style={{
                  minWidth: 68,
                  height: 74,
                  borderRadius: 14,
                  background: 'rgba(20,24,36,0.9)',
                  border: '1px dashed rgba(0,229,195,0.55)',
                  color: '#00E5C3',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 3,
                  cursor: 'pointer',
                  flexShrink: 0,
                }}
              >
                <ImagePlus size={18} />
                <span style={{ fontSize: 10, fontWeight: 600 }}>Upload</span>
                <span style={{ fontSize: 9, color: '#9CA3B8' }}>আপলোড</span>
              </button>
              <input
                ref={uploadCharRef}
                type="file"
                accept="image/*"
                style={{ display: 'none' }}
                onChange={onUploadCharacterFace}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
