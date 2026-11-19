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
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

class VideoFaceSwapEngine(private val context: Context) {

    private var multiFaceLandmarker: FaceLandmarker? = null

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
                    .setMinFaceDetectionConfidence(0.45f)
                    .setMinFacePresenceConfidence(0.45f)
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
                var bestArea: Float,
                var thumbnail: Bitmap,
            )

            val clusters = mutableListOf<Cluster>()
            val sampleCount = 9
            for (i in 0 until sampleCount) {
                val frac = 0.04f + (i.toFloat() / (sampleCount - 1)) * 0.90f
                val timeUs = (durationMs * 1000L * frac).toLong()
                val frame = extractScaledFrame(retriever, timeUs, 640)
                if (frame != null) {
                    val result = detector.detect(BitmapImageBuilder(frame).build())
                    for (lms in result.faceLandmarks()) {
                        var minX = 1f
                        var minY = 1f
                        var maxX = 0f
                        var maxY = 0f
                        for (pt in lms) {
                            if (pt.x() < minX) minX = pt.x()
                            if (pt.x() > maxX) maxX = pt.x()
                            if (pt.y() < minY) minY = pt.y()
                            if (pt.y() > maxY) maxY = pt.y()
                        }
                        val cx = (minX + maxX) * 0.5f
                        val cy = (minY + maxY) * 0.5f
                        val fw = (maxX - minX).coerceAtLeast(0.06f)
                        val fh = (maxY - minY).coerceAtLeast(0.06f)
                        val area = fw * fh

                        var matched: Cluster? = null
                        var bestDist = Float.MAX_VALUE
                        for (c in clusters) {
                            val dist = hypot(cx - c.anchorX, cy - c.anchorY)
                            val threshold = max(0.16f, max(c.faceWidth, fw) * 1.1f)
                            if (dist < threshold && dist < bestDist) {
                                bestDist = dist
                                matched = c
                            }
                        }

                        if (matched != null) {
                            matched.anchorX = matched.anchorX * 0.6f + cx * 0.4f
                            matched.anchorY = matched.anchorY * 0.6f + cy * 0.4f
                            matched.faceWidth = max(matched.faceWidth, fw)
                            if (area > matched.bestArea) {
                                matched.bestArea = area
                                matched.thumbnail = cropFaceThumbnail(frame, cx, cy, fw, fh)
                            }
                        } else if (clusters.size < 5) {
                            val idx = clusters.size + 1
                            clusters.add(
                                Cluster(
                                    id = "person_$idx",
                                    index = idx,
                                    anchorX = cx,
                                    anchorY = cy,
                                    faceWidth = fw,
                                    bestArea = area,
                                    thumbnail = cropFaceThumbnail(frame, cx, cy, fw, fh),
                                ),
                            )
                        }
                    }
                    frame.recycle()
                }
                onProgress(((i + 1) * 100) / sampleCount)
            }

            val people = clusters.map { c ->
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
                            if (audioTrackIn >= 0 && audioFormat != null) {
                                audioTrackMuxer = runCatching { mp4Muxer.addTrack(audioFormat!!) }.getOrDefault(-1)
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

    private fun drawMultiFaceSwapsOnBitmap(
        targetBitmap: Bitmap,
        detector: FaceLandmarker,
        people: List<DetectedVideoPerson>,
        facesById: Map<String, CharacterFace>,
        dynamicAnchors: MutableMap<String, FloatArray>,
        smoothers: MutableMap<String, LandmarkOneEuroSmoother>?,
        timestampMs: Long,
        smoothingStrength: Float,
        opacity: Float,
    ) {
        if (people.isEmpty()) return
        val result = detector.detect(BitmapImageBuilder(targetBitmap).build())
        val faceList = result.faceLandmarks()
        if (faceList.isEmpty()) return

        val blendshapesList = result.faceBlendshapes().orElse(emptyList())
        val canvas = Canvas(targetBitmap)
        val w = targetBitmap.width.toFloat()
        val h = targetBitmap.height.toFloat()
        val usedPeople = HashSet<String>()
        val indices = FaceMeshTriangulation.triangleIndices
        val preAlphas = FaceMeshTriangulation.vertexAlphas

        for (faceIdx in faceList.indices) {
            val lms = faceList[faceIdx]
            var minX = 1f
            var minY = 1f
            var maxX = 0f
            var maxY = 0f
            for (pt in lms) {
                if (pt.x() < minX) minX = pt.x()
                if (pt.x() > maxX) maxX = pt.x()
                if (pt.y() < minY) minY = pt.y()
                if (pt.y() > maxY) maxY = pt.y()
            }
            val cx = (minX + maxX) * 0.5f
            val cy = (minY + maxY) * 0.5f

            var bestPerson: DetectedVideoPerson? = null
            var bestDist = Float.MAX_VALUE
            for (p in people) {
                if (usedPeople.contains(p.id)) continue
                val anchor = dynamicAnchors[p.id] ?: floatArrayOf(p.anchorX, p.anchorY)
                val dist = hypot(cx - anchor[0], cy - anchor[1])
                if (dist < 0.34f && dist < bestDist) {
                    bestDist = dist
                    bestPerson = p
                }
            }
            if (bestPerson == null) {
                bestPerson = people.firstOrNull { !usedPeople.contains(it.id) }
            }
            if (bestPerson == null) continue
            usedPeople.add(bestPerson.id)
            dynamicAnchors[bestPerson.id] = floatArrayOf(cx, cy)

            val assignedId = bestPerson.assignedFaceId ?: continue
            val charFace = facesById[assignedId] ?: continue
            val charBmp = charFace.bitmap ?: continue
            val charUv = charFace.uvLandmarks ?: continue

            val count = min(lms.size, charUv.size / 2)
            val rawXY = FloatArray(count * 2)
            for (i in 0 until count) {
                rawXY[i * 2] = lms[i].x()
                rawXY[i * 2 + 1] = lms[i].y()
            }

            val finalXY = if (smoothers != null && smoothingStrength > 0.01f) {
                val smoother = smoothers.getOrPut(bestPerson.id) { LandmarkOneEuroSmoother() }
                smoother.smooth(rawXY, count, timestampMs, smoothingStrength)
            } else {
                rawXY
            }

            val jawOpen = blendshapesList.getOrNull(faceIdx)
                ?.firstOrNull { it.categoryName() == "jawOpen" }
                ?.score() ?: 0f

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

                val vAlpha = if (i < preAlphas.size) preAlphas[i] else 1f
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

        private fun toBengaliDigits(value: Int): String {
            return value.toString().map { ch ->
                if (ch in '0'..'9') BN_DIGITS[ch - '0'] else ch
            }.joinToString("")
        }
    }
}
