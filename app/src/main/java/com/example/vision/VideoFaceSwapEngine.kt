package com.example.vision

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapShader
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.graphics.Shader
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.media.MediaMuxer
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import com.example.model.CharacterFace
import com.example.model.DetectedVideoPerson
import com.example.model.VideoTrajectoryKeyframe
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

class VideoFaceSwapEngine(private val context: Context) {

    private var multiFaceLandmarker: FaceLandmarker? = null
    private val previewHeadTracker = ProfileHeadTracker()
    private val previewLiveSnapshots = HashMap<String, VideoTrajectoryKeyframe>()
    private var previewLastUri: Uri? = null

    private class ProfileHeadTracker {
        private val gridW = 112
        private val gridH = 112
        private val pixels = IntArray(gridW * gridH)
        private val lum = FloatArray(gridW * gridH)

        private data class Template(
            val patch: FloatArray,
            val mean: Float,
            val std: Float,
            val boxNorm: Float,
        )

        private val templates = HashMap<String, Template>()

        fun prepareFrame(source: Bitmap) {
            val scaled = Bitmap.createScaledBitmap(source, gridW, gridH, true)
            scaled.getPixels(pixels, 0, gridW, 0, 0, gridW, gridH)
            if (scaled !== source) scaled.recycle()
            for (i in pixels.indices) {
                val c = pixels[i]
                val r = (c shr 16) and 0xFF
                val g = (c shr 8) and 0xFF
                val b = c and 0xFF
                lum[i] = 0.299f * r + 0.587f * g + 0.114f * b
            }
        }

        private fun samplePatch(cxNorm: Float, cyNorm: Float, boxNorm: Float, out: FloatArray): Pair<Float, Float> {
            val dim = 8
            val half = boxNorm * 0.5f
            var sum = 0f
            for (gy in 0 until dim) {
                val ny = cyNorm - half + ((gy + 0.5f) / dim) * boxNorm
                val py = (ny * (gridH - 1)).toInt().coerceIn(0, gridH - 1)
                for (gx in 0 until dim) {
                    val nx = cxNorm - half + ((gx + 0.5f) / dim) * boxNorm
                    val px = (nx * (gridW - 1)).toInt().coerceIn(0, gridW - 1)
                    val v = lum[py * gridW + px]
                    out[gy * dim + gx] = v
                    sum += v
                }
            }
            val mean = sum / 64f
            var varSum = 0f
            for (i in 0 until 64) {
                val d = out[i] - mean
                varSum += d * d
            }
            return mean to max(4f, sqrt(varSum / 64f))
        }

        fun captureTemplate(personId: String, cxNorm: Float, cyNorm: Float, faceWidthNorm: Float) {
            val boxNorm = (faceWidthNorm * 1.45f).coerceIn(0.14f, 0.65f)
            val patch = FloatArray(64)
            val (mean, std) = samplePatch(cxNorm, cyNorm, boxNorm, patch)
            templates[personId] = Template(patch, mean, std, boxNorm)
        }

        fun locateHead(personId: String, predCx: Float, predCy: Float): Triple<Float, Float, Float> {
            val tpl = templates[personId] ?: return Triple(predCx, predCy, 0f)
            val candidate = FloatArray(64)
            var bestCx = predCx
            var bestCy = predCy
            var bestNcc = -1f

            fun evalAt(tx: Float, ty: Float) {
                val cx = tx.coerceIn(0.05f, 0.95f)
                val cy = ty.coerceIn(0.05f, 0.95f)
                val (cMean, cStd) = samplePatch(cx, cy, tpl.boxNorm, candidate)
                var cov = 0f
                for (i in 0 until 64) {
                    cov += (candidate[i] - cMean) * (tpl.patch[i] - tpl.mean)
                }
                val ncc = cov / (64f * cStd * tpl.std)
                if (ncc > bestNcc) {
                    bestNcc = ncc
                    bestCx = cx
                    bestCy = cy
                }
            }

            var dy = -0.088f
            while (dy <= 0.089f) {
                var dx = -0.088f
                while (dx <= 0.089f) {
                    evalAt(predCx + dx, predCy + dy)
                    dx += 0.022f
                }
                dy += 0.022f
            }

            val rOriginX = bestCx
            val rOriginY = bestCy
            var ry = -0.016f
            while (ry <= 0.017f) {
                var rx = -0.016f
                while (rx <= 0.017f) {
                    evalAt(rOriginX + rx, rOriginY + ry)
                    rx += 0.008f
                }
                ry += 0.008f
            }

            return Triple(bestCx, bestCy, bestNcc.coerceAtLeast(0f))
        }

        fun clear() {
            templates.clear()
        }
    }

    private data class DetectedFaceSample(
        val rawXY: FloatArray,
        val count: Int,
        val jawOpen: Float,
        val yaw: Float,
        val faceWidth: Float,
        val cx: Float,
        val cy: Float,
    )

    private fun ensureMultiFaceLandmarker(): FaceLandmarker? {
        multiFaceLandmarker?.let { return it }
        for (delegate in listOf(Delegate.GPU, Delegate.CPU)) {
            val ok = runCatching {
                val base = BaseOptions.builder()
                    .setModelAssetPath("face_landmarker.task")
                    .setDelegate(delegate)
                    .build()
                val opts = FaceLandmarker.FaceLandmarkerOptions.builder()
                    .setBaseOptions(base)
                    .setRunningMode(RunningMode.IMAGE)
                    .setNumFaces(5)
                    .setMinFaceDetectionConfidence(0.2f)
                    .setMinFacePresenceConfidence(0.2f)
                    .setMinTrackingConfidence(0.15f)
                    .setOutputFaceBlendshapes(true)
                    .build()
                FaceLandmarker.createFromOptions(context, opts)
            }.getOrNull()
            if (ok != null) {
                multiFaceLandmarker = ok
                return ok
            }
        }
        return null
    }

    fun scanVideoForPeople(
        uri: Uri,
        onProgress: (Int) -> Unit,
    ): Pair<Long, List<DetectedVideoPerson>> {
        val detector = ensureMultiFaceLandmarker() ?: return 0L to emptyList()
        val retriever = MediaMetadataRetriever()
        try {
            retriever.setDataSource(context, uri)
            val durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLongOrNull()
                ?.coerceAtLeast(1L)
                ?: 1000L

            data class Cluster(
                val id: String,
                val index: Int,
                var anchorX: Float,
                var anchorY: Float,
                var faceWidth: Float,
                var bestCenteredScore: Float,
                var thumbnail: Bitmap,
                val trajectory: MutableList<VideoTrajectoryKeyframe>,
            )

            val clusters = mutableListOf<Cluster>()
            val durationSec = durationMs / 1000.0
            val sampleCount = min(56, max(14, ceil(durationSec * 4.0).toInt()))
            for (i in 0 until sampleCount) {
                val frac = if (sampleCount == 1) 0f else (i.toFloat() / (sampleCount - 1)) * 0.96f + 0.02f
                val frameMs = (durationMs * frac).toLong()
                val timeUs = frameMs * 1000L
                val frame = extractScaledFrame(retriever, timeUs, 512)
                if (frame != null) {
                    val detectedSamples = ArrayList<DetectedFaceSample>()
                    val result = detector.detect(BitmapImageBuilder(frame).build())
                    val fList = result.faceLandmarks()
                    val bList = result.faceBlendshapes().orElse(emptyList())
                    for (fIdx in fList.indices) {
                        val lms = fList[fIdx]
                        if (lms.isEmpty()) continue
                        var minX = 1f
                        var minY = 1f
                        var maxX = 0f
                        var maxY = 0f
                        val rawXY = FloatArray(lms.size * 2)
                        for (ptIdx in lms.indices) {
                            val px = lms[ptIdx].x()
                            val py = lms[ptIdx].y()
                            rawXY[ptIdx * 2] = px
                            rawXY[ptIdx * 2 + 1] = py
                            if (px < minX) minX = px
                            if (px > maxX) maxX = px
                            if (py < minY) minY = py
                            if (py > maxY) maxY = py
                        }
                        val jaw = bList.getOrNull(fIdx)?.firstOrNull { it.categoryName() == "jawOpen" }?.score() ?: 0f
                        detectedSamples.add(
                            DetectedFaceSample(
                                rawXY = rawXY,
                                count = lms.size,
                                jawOpen = jaw,
                                yaw = estimateYaw(rawXY, lms.size),
                                faceWidth = (maxX - minX).coerceAtLeast(0.06f),
                                cx = (minX + maxX) * 0.5f,
                                cy = (minY + maxY) * 0.5f,
                            ),
                        )
                    }

                    if (detectedSamples.isEmpty()) {
                        val lifted = createShadowLiftedBitmap(frame)
                        val liftedRes = detector.detect(BitmapImageBuilder(lifted).build())
                        val lList = liftedRes.faceLandmarks()
                        val lbList = liftedRes.faceBlendshapes().orElse(emptyList())
                        for (fIdx in lList.indices) {
                            val lms = lList[fIdx]
                            if (lms.isEmpty()) continue
                            var minX = 1f; var minY = 1f; var maxX = 0f; var maxY = 0f
                            val rawXY = FloatArray(lms.size * 2)
                            for (ptIdx in lms.indices) {
                                val px = lms[ptIdx].x(); val py = lms[ptIdx].y()
                                rawXY[ptIdx * 2] = px; rawXY[ptIdx * 2 + 1] = py
                                if (px < minX) minX = px; if (px > maxX) maxX = px
                                if (py < minY) minY = py; if (py > maxY) maxY = py
                            }
                            val jaw = lbList.getOrNull(fIdx)?.firstOrNull { it.categoryName() == "jawOpen" }?.score() ?: 0f
                            detectedSamples.add(
                                DetectedFaceSample(
                                    rawXY = rawXY,
                                    count = lms.size,
                                    jawOpen = jaw,
                                    yaw = estimateYaw(rawXY, lms.size),
                                    faceWidth = (maxX - minX).coerceAtLeast(0.06f),
                                    cx = (minX + maxX) * 0.5f,
                                    cy = (minY + maxY) * 0.5f,
                                ),
                            )
                        }
                        if (lifted !== frame) lifted.recycle()
                    }

                    if (detectedSamples.isEmpty() && clusters.isNotEmpty()) {
                        for (c in clusters) {
                            val lastKf = c.trajectory.lastOrNull()
                            val rx = lastKf?.cx ?: c.anchorX
                            val ry = lastKf?.cy ?: c.anchorY
                            val boxSize = min(0.95f, max(c.faceWidth, 0.18f) * 2.4f)
                            val roiSample = detectFaceInRoi(frame, detector, rx, ry, boxSize)
                            if (roiSample != null) detectedSamples.add(roiSample)
                        }
                    }

                    for (sample in detectedSamples) {
                        val fw = sample.faceWidth
                        val fh = fw * 1.15f
                        val area = fw * fh
                        val centeredScore = area * (1.15f - min(0.8f, abs(sample.yaw)))
                        val keyframe = VideoTrajectoryKeyframe(
                            timeMs = frameMs,
                            cx = sample.cx,
                            cy = sample.cy,
                            faceWidth = fw,
                            yaw = sample.yaw,
                            jawOpen = sample.jawOpen,
                            xy = sample.rawXY,
                        )

                        var matched: Cluster? = null
                        var bestDist = Float.MAX_VALUE
                        for (c in clusters) {
                            val lastKf = c.trajectory.lastOrNull()
                            val refX = lastKf?.cx ?: c.anchorX
                            val refY = lastKf?.cy ?: c.anchorY
                            val dist = hypot(sample.cx - refX, sample.cy - refY)
                            val threshold = max(0.18f, max(c.faceWidth, fw) * 1.25f)
                            if (dist < threshold && dist < bestDist) {
                                bestDist = dist
                                matched = c
                            }
                        }

                        if (matched != null) {
                            matched.anchorX = matched.anchorX * 0.7f + sample.cx * 0.3f
                            matched.anchorY = matched.anchorY * 0.7f + sample.cy * 0.3f
                            matched.faceWidth = max(matched.faceWidth, fw)
                            matched.trajectory.add(keyframe)
                            if (centeredScore > matched.bestCenteredScore) {
                                matched.bestCenteredScore = centeredScore
                                matched.thumbnail = cropFaceThumbnail(frame, sample.cx, sample.cy, fw, fh)
                            }
                        } else if (clusters.size < 5) {
                            val idx = clusters.size + 1
                            clusters.add(
                                Cluster(
                                    id = "person_$idx",
                                    index = idx,
                                    anchorX = sample.cx,
                                    anchorY = sample.cy,
                                    faceWidth = fw,
                                    bestCenteredScore = centeredScore,
                                    thumbnail = cropFaceThumbnail(frame, sample.cx, sample.cy, fw, fh),
                                    trajectory = mutableListOf(keyframe),
                                ),
                            )
                        }
                    }
                    frame.recycle()
                }
                onProgress(((i + 1) * 100) / sampleCount)
            }

            previewHeadTracker.clear()
            previewLiveSnapshots.clear()
            previewLastUri = uri

            val people = clusters.map { c ->
                c.trajectory.sortBy { it.timeMs }
                DetectedVideoPerson(
                    id = c.id,
                    index = c.index,
                    labelEn = "Person ${c.index}",
                    labelBn = "ব্যক্তি ${toBengaliDigits(c.index)}",
                    thumbnail = c.thumbnail,
                    anchorX = c.anchorX,
                    anchorY = c.anchorY,
                    faceWidth = c.faceWidth,
                    assignedFaceId = null,
                    trajectory = c.trajectory.toList(),
                )
            }
            return durationMs to people
        } catch (t: Throwable) {
            Log.e(TAG, "scanVideoForPeople failed", t)
            return 0L to emptyList()
        } finally {
            runCatching { retriever.release() }
        }
    }

    fun renderPreviewBitmap(
        uri: Uri,
        timeMs: Long,
        people: List<DetectedVideoPerson>,
        facesById: Map<String, CharacterFace>,
        opacity: Float,
    ): Bitmap? {
        val detector = ensureMultiFaceLandmarker() ?: return null
        if (previewLastUri != uri) {
            previewHeadTracker.clear()
            previewLiveSnapshots.clear()
            previewLastUri = uri
        }
        val retriever = MediaMetadataRetriever()
        return try {
            retriever.setDataSource(context, uri)
            val frame = extractScaledFrame(retriever, timeMs * 1000L, 720) ?: return null
            val out = frame.copy(Bitmap.Config.ARGB_8888, true)
            frame.recycle()
            val anchors = people.associate { it.id to floatArrayOf(it.anchorX, it.anchorY) }.toMutableMap()
            drawMultiFaceSwapsOnBitmap(
                targetBitmap = out,
                detector = detector,
                people = people,
                facesById = facesById,
                dynamicAnchors = anchors,
                smoothers = null,
                liveSnapshots = previewLiveSnapshots,
                headTracker = previewHeadTracker,
                timestampMs = timeMs,
                smoothingStrength = 0f,
                opacity = opacity,
            )
            out
        } catch (t: Throwable) {
            Log.e(TAG, "renderPreviewBitmap failed", t)
            null
        } finally {
            runCatching { retriever.release() }
        }
    }

    fun exportSwappedVideo(
        uri: Uri,
        people: List<DetectedVideoPerson>,
        facesById: Map<String, CharacterFace>,
        opacity: Float,
        smoothingStrength: Float,
        onProgress: (Int) -> Unit,
    ): Uri? {
        val detector = ensureMultiFaceLandmarker() ?: return null
        val retriever = MediaMetadataRetriever()
        val tempOut = File(context.cacheDir, "faceswap_export_${System.currentTimeMillis()}.mp4")
        var encoder: MediaCodec? = null
        var muxer: MediaMuxer? = null
        var audioExtractor: MediaExtractor? = null

        try {
            retriever.setDataSource(context, uri)
            val durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLongOrNull()
                ?.coerceAtLeast(500L)
                ?: 1000L

            val firstFrame = extractScaledFrame(retriever, 0L, 720) ?: return null
            val outW = (firstFrame.width and 1.inv()).coerceAtLeast(160)
            val outH = (firstFrame.height and 1.inv()).coerceAtLeast(160)
            firstFrame.recycle()

            val fps = 20
            val frameDurationUs = 1_000_000L / fps
            val totalFrames = ((durationMs * fps) / 1000L).toInt().coerceAtLeast(1)

            val videoFormat = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, outW, outH).apply {
                setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
                setInteger(MediaFormat.KEY_BIT_RATE, 3_000_000)
                setInteger(MediaFormat.KEY_FRAME_RATE, fps)
                setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
            }

            encoder = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
            encoder.configure(videoFormat, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            val inputSurface = encoder.createInputSurface()
            encoder.start()

            muxer = MediaMuxer(tempOut.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)

            // Check for an original audio track to copy cleanly
            var audioTrackIn = -1
            var audioFormat: MediaFormat? = null
            audioExtractor = MediaExtractor().apply {
                setDataSource(context, uri, null)
                for (t in 0 until trackCount) {
                    val fmt = getTrackFormat(t)
                    val mime = fmt.getString(MediaFormat.KEY_MIME) ?: ""
                    if (mime.startsWith("audio/")) {
                        audioTrackIn = t
                        audioFormat = fmt
                        selectTrack(t)
                        break
                    }
                }
            }

            var videoTrackMuxer = -1
            var audioTrackMuxer = -1
            var muxerStarted = false
            var encodedFrameIndex = 0L

            val bufferInfo = MediaCodec.BufferInfo()
            val dynamicAnchors = people.associate { it.id to floatArrayOf(it.anchorX, it.anchorY) }.toMutableMap()
            val smoothers = HashMap<String, LandmarkOneEuroSmoother>()
            val liveSnapshots = HashMap<String, VideoTrajectoryKeyframe>()
            val exportHeadTracker = ProfileHeadTracker()

            val codec = encoder
            val mp4Muxer = muxer

            fun drainEncoder(endOfStream: Boolean) {
                if (endOfStream) {
                    codec.signalEndOfInputStream()
                }
                while (true) {
                    val outIndex = codec.dequeueOutputBuffer(bufferInfo, if (endOfStream) 10_000L else 0L)
                    when {
                        outIndex == MediaCodec.INFO_TRY_AGAIN_LATER -> {
                            if (!endOfStream) break
                        }
                        outIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                            videoTrackMuxer = mp4Muxer.addTrack(codec.outputFormat)
                            val af = audioFormat
                            if (audioTrackIn >= 0 && af != null) {
                                audioTrackMuxer = runCatching { mp4Muxer.addTrack(af) }.getOrDefault(-1)
                            }
                            mp4Muxer.start()
                            muxerStarted = true
                        }
                        outIndex >= 0 -> {
                            val encoded = codec.getOutputBuffer(outIndex)
                            if (encoded != null && bufferInfo.size > 0 && muxerStarted && (bufferInfo.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG) == 0) {
                                encoded.position(bufferInfo.offset)
                                encoded.limit(bufferInfo.offset + bufferInfo.size)
                                bufferInfo.presentationTimeUs = encodedFrameIndex * frameDurationUs
                                encodedFrameIndex++
                                mp4Muxer.writeSampleData(videoTrackMuxer, encoded, bufferInfo)
                            }
                            codec.releaseOutputBuffer(outIndex, false)
                            if ((bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                                break
                            }
                        }
                    }
                }
            }

            for (frameIdx in 0 until totalFrames) {
                val ptsUs = frameIdx * frameDurationUs
                val raw = extractScaledFrame(retriever, ptsUs, max(outW, outH)) ?: continue
                val canvasBitmap = if (raw.width == outW && raw.height == outH) {
                    raw.copy(Bitmap.Config.ARGB_8888, true).also { raw.recycle() }
                } else {
                    Bitmap.createScaledBitmap(raw, outW, outH, true).also { raw.recycle() }
                }

                drawMultiFaceSwapsOnBitmap(
                    targetBitmap = canvasBitmap,
                    detector = detector,
                    people = people,
                    facesById = facesById,
                    dynamicAnchors = dynamicAnchors,
                    smoothers = smoothers,
                    liveSnapshots = liveSnapshots,
                    headTracker = exportHeadTracker,
                    timestampMs = ptsUs / 1000L,
                    smoothingStrength = smoothingStrength,
                    opacity = opacity,
                )

                val surfCanvas = inputSurface.lockCanvas(null)
                surfCanvas.drawBitmap(canvasBitmap, 0f, 0f, null)
                inputSurface.unlockCanvasAndPost(surfCanvas)
                canvasBitmap.recycle()

                drainEncoder(false)
                onProgress(((frameIdx + 1) * 90) / totalFrames)
            }

            drainEncoder(true)

            // Copy original audio track samples if present
            if (muxerStarted && audioTrackIn >= 0 && audioTrackMuxer >= 0) {
                val audioBuf = ByteBuffer.allocate(256 * 1024)
                val audioInfo = MediaCodec.BufferInfo()
                while (true) {
                    val sampleSize = audioExtractor.readSampleData(audioBuf, 0)
                    if (sampleSize < 0) break
                    audioInfo.offset = 0
                    audioInfo.size = sampleSize
                    audioInfo.presentationTimeUs = audioExtractor.sampleTime
                    audioInfo.flags = audioExtractor.sampleFlags
                    if (audioInfo.presentationTimeUs <= durationMs * 1000L) {
                        muxer.writeSampleData(audioTrackMuxer, audioBuf, audioInfo)
                    }
                    audioExtractor.advance()
                }
            }
            onProgress(100)

            muxer.stop()
            muxer.release()
            muxer = null
            encoder.stop()
            encoder.release()
            encoder = null
            inputSurface.release()

            return saveFileToMoviesMediaStore(tempOut)
        } catch (t: Throwable) {
            Log.e(TAG, "exportSwappedVideo failed", t)
            return null
        } finally {
            runCatching { retriever.release() }
            runCatching { audioExtractor?.release() }
            runCatching { encoder?.release() }
            runCatching { muxer?.release() }
            runCatching { if (tempOut.exists()) tempOut.delete() }
        }
    }

    private fun findBracketingKeyframes(
        trajectory: List<VideoTrajectoryKeyframe>,
        liveSample: VideoTrajectoryKeyframe?,
        timeMs: Long,
    ): Pair<VideoTrajectoryKeyframe?, VideoTrajectoryKeyframe?> {
        var prev: VideoTrajectoryKeyframe? = null
        var next: VideoTrajectoryKeyframe? = null
        for (kf in trajectory) {
            if (kf.timeMs <= timeMs) {
                prev = kf
            } else if (next == null) {
                next = kf
                break
            }
        }
        if (liveSample != null && liveSample.timeMs <= timeMs) {
            if (prev == null || liveSample.timeMs >= prev.timeMs) {
                prev = liveSample
            }
        }
        return prev to next
    }

    private fun drawMultiFaceSwapsOnBitmap(
        targetBitmap: Bitmap,
        detector: FaceLandmarker,
        people: List<DetectedVideoPerson>,
        facesById: Map<String, CharacterFace>,
        dynamicAnchors: MutableMap<String, FloatArray>,
        smoothers: MutableMap<String, LandmarkOneEuroSmoother>?,
        liveSnapshots: MutableMap<String, VideoTrajectoryKeyframe>,
        headTracker: ProfileHeadTracker,
        timestampMs: Long,
        smoothingStrength: Float,
        opacity: Float,
    ) {
        if (people.isEmpty()) return
        headTracker.prepareFrame(targetBitmap)

        // Trajectory-predicted anchors for ROI fallback and clustering
        val predictedAnchors = HashMap<String, FloatArray>(people.size)
        for (p in people) {
            val (prev, next) = findBracketingKeyframes(p.trajectory, liveSnapshots[p.id], timestampMs)
            val pred = if (prev != null && next != null && next.timeMs > prev.timeMs) {
                val a = ((timestampMs - prev.timeMs).toFloat() / (next.timeMs - prev.timeMs)).coerceIn(0f, 1f)
                floatArrayOf(prev.cx + (next.cx - prev.cx) * a, prev.cy + (next.cy - prev.cy) * a)
            } else {
                dynamicAnchors[p.id]
                    ?: (prev ?: next)?.let { floatArrayOf(it.cx, it.cy) }
                    ?: floatArrayOf(p.anchorX, p.anchorY)
            }
            predictedAnchors[p.id] = pred
        }

        val result = detector.detect(BitmapImageBuilder(targetBitmap).build())
        val faceList = result.faceLandmarks()
        val blendshapesList = result.faceBlendshapes().orElse(emptyList())

        val detectedSamples = ArrayList<DetectedFaceSample>(faceList.size.coerceAtLeast(1))
        for (faceIdx in faceList.indices) {
            val lms = faceList[faceIdx]
            if (lms.isEmpty()) continue
            var minX = 1f
            var minY = 1f
            var maxX = 0f
            var maxY = 0f
            val rawXY = FloatArray(lms.size * 2)
            for (i in lms.indices) {
                val x = lms[i].x()
                val y = lms[i].y()
                rawXY[i * 2] = x
                rawXY[i * 2 + 1] = y
                if (x < minX) minX = x
                if (x > maxX) maxX = x
                if (y < minY) minY = y
                if (y > maxY) maxY = y
            }
            val jawOpen = blendshapesList.getOrNull(faceIdx)
                ?.firstOrNull { it.categoryName() == "jawOpen" }
                ?.score() ?: 0f
            detectedSamples.add(
                DetectedFaceSample(
                    rawXY = rawXY,
                    count = lms.size,
                    jawOpen = jawOpen,
                    yaw = estimateYaw(rawXY, lms.size),
                    faceWidth = (maxX - minX).coerceAtLeast(0.06f),
                    cx = (minX + maxX) * 0.5f,
                    cy = (minY + maxY) * 0.5f,
                ),
            )
        }

        // Backlit / Steep Side-Turn ROI fallback per tracked person around predicted trajectory center
        if (detectedSamples.isEmpty()) {
            for (p in people) {
                val anchor = predictedAnchors[p.id] ?: floatArrayOf(p.anchorX, p.anchorY)
                val boxSize = min(0.95f, max(p.faceWidth, 0.18f) * 2.4f)
                val roiSample = detectFaceInRoi(targetBitmap, detector, anchor[0], anchor[1], boxSize)
                if (roiSample != null) {
                    detectedSamples.add(roiSample)
                }
            }
        }

        val canvas = Canvas(targetBitmap)
        val usedPeople = HashSet<String>()

        for (sample in detectedSamples) {
            var bestPerson: DetectedVideoPerson? = null
            var bestDist = Float.MAX_VALUE
            for (p in people) {
                if (usedPeople.contains(p.id)) continue
                val anchor = predictedAnchors[p.id] ?: floatArrayOf(p.anchorX, p.anchorY)
                val dist = hypot(sample.cx - anchor[0], sample.cy - anchor[1])
                if (dist < 0.36f && dist < bestDist) {
                    bestDist = dist
                    bestPerson = p
                }
            }
            if (bestPerson == null) {
                bestPerson = people.firstOrNull { !usedPeople.contains(it.id) }
            }
            if (bestPerson == null) continue
            usedPeople.add(bestPerson.id)
            dynamicAnchors[bestPerson.id] = floatArrayOf(sample.cx, sample.cy)
            headTracker.captureTemplate(bestPerson.id, sample.cx, sample.cy, sample.faceWidth)
            liveSnapshots[bestPerson.id] = VideoTrajectoryKeyframe(
                timeMs = timestampMs,
                cx = sample.cx,
                cy = sample.cy,
                faceWidth = sample.faceWidth,
                yaw = sample.yaw,
                jawOpen = sample.jawOpen,
                xy = sample.rawXY.copyOf(),
            )

            val assignedId = bestPerson.assignedFaceId ?: continue
            val charFace = facesById[assignedId] ?: continue
            val charUv = charFace.uvLandmarks ?: continue
            val count = min(sample.count, charUv.size / 2)

            val finalXY = if (smoothers != null && smoothingStrength > 0.01f) {
                val smoother = smoothers.getOrPut(bestPerson.id) { LandmarkOneEuroSmoother() }
                smoother.smooth(sample.rawXY, count, timestampMs, smoothingStrength)
            } else {
                sample.rawXY
            }

            renderSingleFaceSwap(
                canvas = canvas,
                targetBitmap = targetBitmap,
                finalXY = finalXY,
                count = count,
                jawOpen = sample.jawOpen,
                charFace = charFace,
                opacity = opacity,
            )
        }

        // Bidirectional 3D Trajectory + Profile Head Tracker reconstruction for any person whose face
        // is turned sideways / occluded on this frame
        for (p in people) {
            if (usedPeople.contains(p.id)) continue
            val assignedId = p.assignedFaceId ?: continue
            val charFace = facesById[assignedId] ?: continue
            val charUv = charFace.uvLandmarks ?: continue

            val (prev, next) = findBracketingKeyframes(p.trajectory, liveSnapshots[p.id], timestampMs)
            if (prev == null && next == null) continue

            val finalRawXY: FloatArray
            val count: Int
            val jawOpen: Float
            val opacityScale: Float
            val trackCx: Float
            val trackCy: Float

            if (prev != null && next != null && next.timeMs > prev.timeMs) {
                val gapMs = (next.timeMs - prev.timeMs).toFloat()
                val rawAlpha = ((timestampMs - prev.timeMs) / gapMs).coerceIn(0f, 1f)
                val alpha = rawAlpha * rawAlpha * (3f - 2f * rawAlpha)

                val predCx = prev.cx + (next.cx - prev.cx) * alpha
                val predCy = prev.cy + (next.cy - prev.cy) * alpha

                val (tCx, tCy, tConf) = headTracker.locateHead(p.id, predCx, predCy)
                val blend = when {
                    tConf > 0.55f -> 0.65f
                    tConf > 0.35f -> 0.35f
                    else -> 0f
                }
                trackCx = predCx + (tCx - predCx) * blend
                trackCy = predCy + (tCy - predCy) * blend

                val turnArc = sin(PI * rawAlpha).toFloat()
                val turningSideways = (prev.yaw * next.yaw >= 0f && (abs(prev.yaw) > 0.22f || abs(next.yaw) > 0.22f)) ||
                    abs(next.yaw - prev.yaw) > 0.45f
                val xProfileScale = if (turningSideways) 1f - 0.16f * turnArc else 1f

                count = min(min(prev.xy.size / 2, next.xy.size / 2), charUv.size / 2)
                finalRawXY = FloatArray(count * 2)
                for (i in 0 until count) {
                    val pRelX = prev.xy[i * 2] - prev.cx
                    val pRelY = prev.xy[i * 2 + 1] - prev.cy
                    val nRelX = next.xy[i * 2] - next.cx
                    val nRelY = next.xy[i * 2 + 1] - next.cy
                    val rx = (pRelX + (nRelX - pRelX) * alpha) * xProfileScale
                    val ry = pRelY + (nRelY - pRelY) * alpha
                    finalRawXY[i * 2] = (trackCx + rx).coerceIn(0f, 1f)
                    finalRawXY[i * 2 + 1] = (trackCy + ry).coerceIn(0f, 1f)
                }
                jawOpen = prev.jawOpen + (next.jawOpen - prev.jawOpen) * alpha
                opacityScale = if (gapMs <= 6000f) {
                    1f
                } else {
                    val edgeDist = min((timestampMs - prev.timeMs).toFloat(), (next.timeMs - timestampMs).toFloat())
                    (1f - (edgeDist - 3000f) / 3000f).coerceIn(0.25f, 1f)
                }
            } else {
                val ref = prev ?: next!!
                val distMs = abs(timestampMs - ref.timeMs)
                if (distMs > 3000L) continue

                val lastAnchor = dynamicAnchors[p.id]
                val baseCx = lastAnchor?.get(0) ?: ref.cx
                val baseCy = lastAnchor?.get(1) ?: ref.cy
                val (tCx, tCy, tConf) = headTracker.locateHead(p.id, baseCx, baseCy)
                val useTracked = tConf > 0.45f
                trackCx = if (useTracked) baseCx * 0.25f + tCx * 0.75f else ref.cx
                trackCy = if (useTracked) baseCy * 0.25f + tCy * 0.75f else ref.cy

                count = min(ref.xy.size / 2, charUv.size / 2)
                finalRawXY = FloatArray(count * 2)
                for (i in 0 until count) {
                    finalRawXY[i * 2] = (trackCx + (ref.xy[i * 2] - ref.cx)).coerceIn(0f, 1f)
                    finalRawXY[i * 2 + 1] = (trackCy + (ref.xy[i * 2 + 1] - ref.cy)).coerceIn(0f, 1f)
                }
                jawOpen = ref.jawOpen
                val holdLimit = if (useTracked) 2600f else 800f
                opacityScale = if (distMs <= 900L) {
                    1f
                } else {
                    (1f - (distMs - 900f) / max(400f, holdLimit - 900f)).coerceIn(0f, 1f)
                }
                if (opacityScale <= 0.02f) continue
            }

            dynamicAnchors[p.id] = floatArrayOf(trackCx, trackCy)
            val finalXY = if (smoothers != null && smoothingStrength > 0.01f) {
                val smoother = smoothers.getOrPut(p.id) { LandmarkOneEuroSmoother() }
                smoother.smooth(finalRawXY, count, timestampMs, smoothingStrength)
            } else {
                finalRawXY
            }

            renderSingleFaceSwap(
                canvas = canvas,
                targetBitmap = targetBitmap,
                finalXY = finalXY,
                count = count,
                jawOpen = jawOpen,
                charFace = charFace,
                opacity = opacity * opacityScale,
            )
        }
    }

    private fun renderSingleFaceSwap(
        canvas: Canvas,
        targetBitmap: Bitmap,
        finalXY: FloatArray,
        count: Int,
        jawOpen: Float,
        charFace: CharacterFace,
        opacity: Float,
    ) {
        val charBmp = charFace.bitmap ?: return
        val charUv = charFace.uvLandmarks ?: return
        val w = targetBitmap.width.toFloat()
        val h = targetBitmap.height.toFloat()
        val indices = FaceMeshTriangulation.triangleIndices
        val preAlphas = FaceMeshTriangulation.vertexAlphas

        if (jawOpen > 0.06f) {
            val mouthPath = Path()
            val ring = FaceMeshTriangulation.innerLipRing
            for (rIdx in ring.indices) {
                val idx = ring[rIdx]
                if (idx >= count) continue
                val px = finalXY[idx * 2] * w
                val py = finalXY[idx * 2 + 1] * h
                if (rIdx == 0) mouthPath.moveTo(px, py) else mouthPath.lineTo(px, py)
            }
            mouthPath.close()
            val mouthPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                style = Paint.Style.FILL
                color = Color.argb(((opacity * jawOpen * 4f).coerceIn(0f, 0.94f) * 255).toInt(), 20, 6, 10)
            }
            canvas.drawPath(mouthPath, mouthPaint)
        }

        // 3D yaw side-occlusion alpha modulation
        val noseX = if (count > 4) finalXY[4 * 2] else 0.5f
        val leftEdgeX = if (count > 234) finalXY[234 * 2] else 0.3f
        val rightEdgeX = if (count > 454) finalXY[454 * 2] else 0.7f
        val span = max(0.04f, kotlin.math.abs(rightEdgeX - leftEdgeX))
        val midX = (leftEdgeX + rightEdgeX) * 0.5f
        val yaw = (((noseX - midX) / (span * 0.5f)) * 1.4f).coerceIn(-1f, 1f)
        val absYaw = kotlin.math.abs(yaw)
        val sideFadeStrength = if (absYaw >= 0.35f) ((absYaw - 0.35f) / 0.55f).coerceIn(0f, 1f) else 0f

        // Skin RGB matching between live video frame & target character face
        val liveSkin = FaceLandmarkerHelper.sampleSkinRgb(targetBitmap, finalXY)
        val srcSkin = FaceLandmarkerHelper.sampleSkinRgb(charBmp, charUv)
        val rGain = ((liveSkin[0] / srcSkin[0]).coerceIn(0.6f, 1.45f) - 1f) * 0.68f + 1f
        val gGain = ((liveSkin[1] / srcSkin[1]).coerceIn(0.6f, 1.45f) - 1f) * 0.68f + 1f
        val bGain = ((liveSkin[2] / srcSkin[2]).coerceIn(0.6f, 1.45f) - 1f) * 0.68f + 1f

        val verts = FloatArray(count * 2)
        val texs = FloatArray(count * 2)
        val colors = IntArray(count)
        val cw = charBmp.width.toFloat()
        val ch = charBmp.height.toFloat()

        for (i in 0 until count) {
            verts[i * 2] = finalXY[i * 2] * w
            verts[i * 2 + 1] = finalXY[i * 2 + 1] * h
            texs[i * 2] = charUv[i * 2] * cw
            texs[i * 2 + 1] = charUv[i * 2 + 1] * ch

            var vAlpha = if (i < preAlphas.size) preAlphas[i] else 1f
            if (sideFadeStrength > 0f) {
                val dxNorm = (finalXY[i * 2] - noseX) / (span * 0.5f)
                val onHiddenSide = if (yaw < 0f) dxNorm > 0.12f else dxNorm < -0.12f
                if (onHiddenSide) {
                    val edgeFrac = ((kotlin.math.abs(dxNorm) - 0.12f) / 0.55f).coerceIn(0f, 1f)
                    vAlpha *= 1f - sideFadeStrength * edgeFrac * 0.85f
                }
            }
            val aInt = (vAlpha * opacity * 255f).toInt().coerceIn(0, 255)
            val rInt = (rGain.coerceIn(0f, 1f) * 255f).toInt().coerceIn(0, 255)
            val gInt = (gGain.coerceIn(0f, 1f) * 255f).toInt().coerceIn(0, 255)
            val bInt = (bGain.coerceIn(0f, 1f) * 255f).toInt().coerceIn(0, 255)
            colors[i] = Color.argb(aInt, rInt, gInt, bInt)
        }

        val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG).apply {
            shader = BitmapShader(charBmp, Shader.TileMode.CLAMP, Shader.TileMode.CLAMP)
        }
        canvas.drawVertices(
            Canvas.VertexMode.TRIANGLES,
            count * 2,
            verts,
            0,
            texs,
            0,
            colors,
            0,
            indices,
            0,
            indices.size,
            paint,
        )
    }

    private fun detectFaceInRoi(
        frame: Bitmap,
        detector: FaceLandmarker,
        anchorX: Float,
        anchorY: Float,
        boxSize: Float,
    ): DetectedFaceSample? {
        val sxNorm = (anchorX - boxSize * 0.5f).coerceIn(0f, 1f - boxSize)
        val syNorm = (anchorY - boxSize * 0.5f).coerceIn(0f, 1f - boxSize)
        val sx = (sxNorm * frame.width).toInt().coerceIn(0, frame.width - 1)
        val sy = (syNorm * frame.height).toInt().coerceIn(0, frame.height - 1)
        val sw = (boxSize * frame.width).toInt().coerceIn(32, frame.width - sx)
        val sh = (boxSize * frame.height).toInt().coerceIn(32, frame.height - sy)

        val roiBitmap = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888)
        return try {
            val canvas = Canvas(roiBitmap)
            val paint = Paint(Paint.FILTER_BITMAP_FLAG).apply {
                colorFilter = android.graphics.ColorMatrixColorFilter(
                    android.graphics.ColorMatrix(
                        floatArrayOf(
                            1.18f, 0f, 0f, 0f, 8f,
                            0f, 1.18f, 0f, 0f, 8f,
                            0f, 0f, 1.18f, 0f, 8f,
                            0f, 0f, 0f, 1f, 0f,
                        ),
                    ),
                )
            }
            canvas.drawBitmap(
                frame,
                android.graphics.Rect(sx, sy, sx + sw, sy + sh),
                android.graphics.Rect(0, 0, 256, 256),
                paint,
            )

            val res = detector.detect(BitmapImageBuilder(roiBitmap).build())
            val lms = res.faceLandmarks().firstOrNull() ?: return null
            val rawXY = FloatArray(lms.size * 2)
            var minX = 1f
            var minY = 1f
            var maxX = 0f
            var maxY = 0f
            for (i in lms.indices) {
                val mx = (sxNorm + lms[i].x() * boxSize).coerceIn(0f, 1f)
                val my = (syNorm + lms[i].y() * boxSize).coerceIn(0f, 1f)
                rawXY[i * 2] = mx
                rawXY[i * 2 + 1] = my
                if (mx < minX) minX = mx
                if (mx > maxX) maxX = mx
                if (my < minY) minY = my
                if (my > maxY) maxY = my
            }
            val jawOpen = res.faceBlendshapes()
                .orElse(emptyList())
                .firstOrNull()
                ?.firstOrNull { it.categoryName() == "jawOpen" }
                ?.score() ?: 0f
            val count = lms.size
            DetectedFaceSample(
                rawXY = rawXY,
                count = count,
                jawOpen = jawOpen,
                yaw = estimateYaw(rawXY, count),
                faceWidth = (maxX - minX).coerceAtLeast(0.06f),
                cx = (minX + maxX) * 0.5f,
                cy = (minY + maxY) * 0.5f,
            )
        } catch (_: Throwable) {
            null
        } finally {
            roiBitmap.recycle()
        }
    }

    private fun createShadowLiftedBitmap(source: Bitmap): Bitmap {
        val out = Bitmap.createBitmap(source.width, source.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(out)
        val paint = Paint(Paint.FILTER_BITMAP_FLAG).apply {
            colorFilter = android.graphics.ColorMatrixColorFilter(
                android.graphics.ColorMatrix(
                    floatArrayOf(
                        1.18f, 0f, 0f, 0f, 8f,
                        0f, 1.18f, 0f, 0f, 8f,
                        0f, 0f, 1.18f, 0f, 8f,
                        0f, 0f, 0f, 1f, 0f,
                    ),
                ),
            )
        }
        canvas.drawBitmap(source, 0f, 0f, paint)
        return out
    }

    private fun extractScaledFrame(
        retriever: MediaMetadataRetriever,
        timeUs: Long,
        maxSide: Int,
    ): Bitmap? {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
            val wMeta = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toIntOrNull() ?: 0
            val hMeta = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toIntOrNull() ?: 0
            if (wMeta > 0 && hMeta > 0) {
                val scale = min(1f, maxSide.toFloat() / max(wMeta, hMeta))
                val targetW = (wMeta * scale).toInt().coerceAtLeast(64)
                val targetH = (hMeta * scale).toInt().coerceAtLeast(64)
                val scaled = runCatching {
                    retriever.getScaledFrameAtTime(
                        timeUs,
                        MediaMetadataRetriever.OPTION_CLOSEST_SYNC,
                        max(targetW, targetH),
                        max(targetW, targetH),
                    )
                }.getOrNull()
                if (scaled != null) return scaled
            }
        }
        val raw = retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST) ?: return null
        val longest = max(raw.width, raw.height)
        if (longest <= maxSide) return raw
        val scale = maxSide.toFloat() / longest
        val out = Bitmap.createScaledBitmap(
            raw,
            (raw.width * scale).toInt().coerceAtLeast(64),
            (raw.height * scale).toInt().coerceAtLeast(64),
            true,
        )
        if (out !== raw) raw.recycle()
        return out
    }

    private fun cropFaceThumbnail(
        frame: Bitmap,
        cx: Float,
        cy: Float,
        fw: Float,
        fh: Float,
    ): Bitmap {
        val pad = max(fw, fh) * 0.35f
        val left = ((cx - fw * 0.5f - pad) * frame.width).toInt().coerceIn(0, frame.width - 1)
        val top = ((cy - fh * 0.5f - pad) * frame.height).toInt().coerceIn(0, frame.height - 1)
        val width = ((fw + pad * 2f) * frame.width).toInt().coerceIn(16, frame.width - left)
        val height = ((fh + pad * 2f) * frame.height).toInt().coerceIn(16, frame.height - top)
        val side = min(width, height)
        val cropped = Bitmap.createBitmap(frame, left, top, side, side)
        val thumb = Bitmap.createScaledBitmap(cropped, 128, 128, true)
        if (thumb !== cropped) cropped.recycle()
        return thumb
    }

    private fun saveFileToMoviesMediaStore(tempFile: File): Uri? {
        val fileName = "FaceSwapVideo_${System.currentTimeMillis()}.mp4"
        val resolver = context.contentResolver
        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, fileName)
            put(MediaStore.MediaColumns.MIME_TYPE, "video/mp4")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.Video.Media.RELATIVE_PATH, "${Environment.DIRECTORY_MOVIES}/FaceSwapCam")
            }
        }
        val uri = resolver.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, values) ?: return null
        resolver.openOutputStream(uri)?.use { out ->
            FileInputStream(tempOutOrFallback(tempFile)).use { input ->
                input.copyTo(out)
            }
        }
        return uri
    }

    private fun tempOutOrFallback(file: File): File = file

    fun close() {
        runCatching { multiFaceLandmarker?.close() }
        multiFaceLandmarker = null
    }

    companion object {
        private const val TAG = "VideoFaceSwapEngine"
        private val BN_DIGITS = charArrayOf('০', '১', '২', '৩', '৪', '৫', '৬', '৭', '৮', '৯')

        private fun estimateYaw(xy: FloatArray, count: Int): Float {
            val noseX = if (count > 4) xy[4 * 2] else 0.5f
            val leftEdgeX = if (count > 234) xy[234 * 2] else 0.3f
            val rightEdgeX = if (count > 454) xy[454 * 2] else 0.7f
            val span = max(0.04f, abs(rightEdgeX - leftEdgeX))
            val midX = (leftEdgeX + rightEdgeX) * 0.5f
            return (((noseX - midX) / (span * 0.5f)) * 1.4f).coerceIn(-1f, 1f)
        }

        private fun toBengaliDigits(value: Int): String {
            return value.toString().map { ch ->
                if (ch in '0'..'9') BN_DIGITS[ch - '0'] else ch
            }.joinToString("")
        }
    }
}
