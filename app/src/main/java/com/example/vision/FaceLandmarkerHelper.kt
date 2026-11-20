package com.example.vision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult

import android.graphics.Color

data class LiveFaceTrackingResult(
    val smoothedXY: FloatArray,
    val landmarkCount: Int,
    val jawOpen: Float,
    val hasFace: Boolean,
    val liveSkinRgb: FloatArray? = null,
)

class FaceLandmarkerHelper(
    private val context: Context,
    private val onLiveResult: (LiveFaceTrackingResult) -> Unit,
) {
    private var liveLandmarker: FaceLandmarker? = null
    private var imageLandmarker: FaceLandmarker? = null
    private val smoother = LandmarkOneEuroSmoother()

    @Volatile
    var smoothingStrength: Float = 0.65f

    @Volatile
    private var lastFrameTimestampMs: Long = 0L

    @Volatile
    private var latestRawXY: FloatArray? = null

    @Volatile
    private var lastSkinSampleMs: Long = 0L

    @Volatile
    private var cachedLiveSkinRgb: FloatArray? = null

    private val frameLock = Any()
    private var latestFrameForRoi: Bitmap? = null
    private var lastBounds: FloatArray? = null // [cx, cy, w, h]
    private var lastSeenMs: Long = 0L

    fun initialize(): Boolean {
        val liveOk = createLiveLandmarker(Delegate.GPU) || createLiveLandmarker(Delegate.CPU)
        val imageOk = createImageLandmarker(Delegate.GPU) || createImageLandmarker(Delegate.CPU)
        return liveOk && imageOk
    }

    private fun createLiveLandmarker(delegate: Delegate): Boolean {
        return try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath("face_landmarker.task")
                .setDelegate(delegate)
                .build()

            val options = FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setNumFaces(1)
                .setMinFaceDetectionConfidence(0.2f)
                .setMinFacePresenceConfidence(0.2f)
                .setMinTrackingConfidence(0.15f)
                .setOutputFaceBlendshapes(true)
                .setResultListener { result, _ -> handleLiveStreamResult(result) }
                .setErrorListener { err ->
                    Log.e(TAG, "Live FaceLandmarker error (${delegate.name})", err)
                }
                .build()

            liveLandmarker = FaceLandmarker.createFromOptions(context, options)
            true
        } catch (t: Throwable) {
            Log.w(TAG, "Failed to init LIVE_STREAM with ${delegate.name}", t)
            false
        }
    }

    private fun createImageLandmarker(delegate: Delegate): Boolean {
        return try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath("face_landmarker.task")
                .setDelegate(delegate)
                .build()

            val options = FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.IMAGE)
                .setNumFaces(1)
                .setMinFaceDetectionConfidence(0.2f)
                .setMinFacePresenceConfidence(0.2f)
                .setOutputFaceBlendshapes(true)
                .build()

            imageLandmarker = FaceLandmarker.createFromOptions(context, options)
            true
        } catch (t: Throwable) {
            Log.w(TAG, "Failed to init IMAGE with ${delegate.name}", t)
            false
        }
    }

    fun detectStaticImage(bitmap: Bitmap): FloatArray? {
        val detector = imageLandmarker ?: return null
        return try {
            val mpImage = BitmapImageBuilder(bitmap).build()
            val result = detector.detect(mpImage)
            val landmarks = result.faceLandmarks().firstOrNull() ?: return null
            val out = FloatArray(landmarks.size * 2)
            for (i in landmarks.indices) {
                out[i * 2] = landmarks[i].x()
                out[i * 2 + 1] = landmarks[i].y()
            }
            out
        } catch (t: Throwable) {
            Log.e(TAG, "Static face detection failed", t)
            null
        }
    }

    fun detectLiveFrame(imageProxy: ImageProxy) {
        val detector = liveLandmarker
        if (detector == null) {
            imageProxy.close()
            return
        }

        try {
            val rawBitmap = imageProxy.toBitmap()
            val rotation = imageProxy.imageInfo.rotationDegrees
            val uprightBitmap = if (rotation != 0) {
                val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
                val rotated = Bitmap.createBitmap(
                    rawBitmap,
                    0,
                    0,
                    rawBitmap.width,
                    rawBitmap.height,
                    matrix,
                    true,
                )
                if (rotated !== rawBitmap) rawBitmap.recycle()
                rotated
            } else {
                rawBitmap
            }

            synchronized(frameLock) {
                latestFrameForRoi = uprightBitmap
            }

            val ts = maxOf(SystemClock.uptimeMillis(), lastFrameTimestampMs + 1L)
            lastFrameTimestampMs = ts

            val rawLandmarks = latestRawXY
            if (rawLandmarks != null && ts - lastSkinSampleMs >= 160L) {
                lastSkinSampleMs = ts
                cachedLiveSkinRgb = sampleSkinRgb(uprightBitmap, rawLandmarks)
            }

            val mpImage = BitmapImageBuilder(uprightBitmap).build()
            detector.detectAsync(mpImage, ts)
        } catch (t: Throwable) {
            Log.e(TAG, "Live frame analysis error", t)
        } finally {
            imageProxy.close()
        }
    }

    private fun handleLiveStreamResult(result: FaceLandmarkerResult) {
        val nowMs = SystemClock.uptimeMillis()
        val landmarks = result.faceLandmarks().firstOrNull()
        if (landmarks != null && landmarks.isNotEmpty()) {
            val count = landmarks.size
            val raw = FloatArray(count * 2)
            for (i in 0 until count) {
                raw[i * 2] = landmarks[i].x()
                raw[i * 2 + 1] = landmarks[i].y()
            }
            val jawOpen = result.faceBlendshapes()
                .orElse(emptyList())
                .firstOrNull()
                ?.firstOrNull { it.categoryName() == "jawOpen" }
                ?.score()
                ?: 0f
            emitTrackedLandmarks(raw, count, jawOpen, nowMs)
            return
        }

        // ROI Fallback around last known face for steep side turns & backlit frames
        val roiMatch = tryDetectRoiFallback(nowMs)
        if (roiMatch != null) {
            val (raw, jawOpen) = roiMatch
            emitTrackedLandmarks(raw, raw.size / 2, jawOpen, nowMs)
            return
        }

        latestRawXY = null
        cachedLiveSkinRgb = null
        if (nowMs - lastSeenMs > 300L) {
            smoother.reset()
        }
        onLiveResult(
            LiveFaceTrackingResult(
                smoothedXY = FloatArray(0),
                landmarkCount = 0,
                jawOpen = 0f,
                hasFace = false,
                liveSkinRgb = null,
            ),
        )
    }

    private fun emitTrackedLandmarks(raw: FloatArray, count: Int, rawJaw: Float, nowMs: Long) {
        latestRawXY = raw
        var minX = 1f
        var minY = 1f
        var maxX = 0f
        var maxY = 0f
        for (i in 0 until count) {
            val x = raw[i * 2]
            val y = raw[i * 2 + 1]
            if (x < minX) minX = x
            if (x > maxX) maxX = x
            if (y < minY) minY = y
            if (y > maxY) maxY = y
        }
        synchronized(frameLock) {
            lastBounds = floatArrayOf(
                (minX + maxX) * 0.5f,
                (minY + maxY) * 0.5f,
                (maxX - minX).coerceAtLeast(0.08f),
                (maxY - minY).coerceAtLeast(0.08f),
            )
            lastSeenMs = nowMs
        }

        val smoothed = smoother.smooth(raw, count, nowMs, smoothingStrength)
        val smoothedJaw = smoother.smoothJawOpen(rawJaw, nowMs)

        onLiveResult(
            LiveFaceTrackingResult(
                smoothedXY = smoothed,
                landmarkCount = count,
                jawOpen = smoothedJaw,
                hasFace = true,
                liveSkinRgb = cachedLiveSkinRgb,
            ),
        )
    }

    private fun tryDetectRoiFallback(nowMs: Long): Pair<FloatArray, Float>? {
        val detector = imageLandmarker ?: return null
        val frame: Bitmap
        val bounds: FloatArray
        val seenAt: Long
        synchronized(frameLock) {
            frame = latestFrameForRoi ?: return null
            bounds = lastBounds ?: return null
            seenAt = lastSeenMs
        }
        if (frame.isRecycled || nowMs - seenAt > 2500L) return null

        val boxSize = minOf(0.95f, maxOf(bounds[2], bounds[3]) * 2.15f)
        val sxNorm = (bounds[0] - boxSize * 0.5f).coerceIn(0f, 1f - boxSize)
        val syNorm = (bounds[1] - boxSize * 0.5f).coerceIn(0f, 1f - boxSize)

        val sx = (sxNorm * frame.width).toInt().coerceIn(0, frame.width - 1)
        val sy = (syNorm * frame.height).toInt().coerceIn(0, frame.height - 1)
        val sw = (boxSize * frame.width).toInt().coerceIn(32, frame.width - sx)
        val sh = (boxSize * frame.height).toInt().coerceIn(32, frame.height - sy)

        val roiBitmap = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888)
        return try {
            val canvas = android.graphics.Canvas(roiBitmap)
            val paint = android.graphics.Paint(android.graphics.Paint.FILTER_BITMAP_FLAG).apply {
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
            val out = FloatArray(lms.size * 2)
            for (i in lms.indices) {
                out[i * 2] = (sxNorm + lms[i].x() * boxSize).coerceIn(0f, 1f)
                out[i * 2 + 1] = (syNorm + lms[i].y() * boxSize).coerceIn(0f, 1f)
            }
            val rawJaw = res.faceBlendshapes()
                .orElse(emptyList())
                .firstOrNull()
                ?.firstOrNull { it.categoryName() == "jawOpen" }
                ?.score()
                ?: 0f
            out to rawJaw
        } catch (_: Throwable) {
            null
        } finally {
            roiBitmap.recycle()
        }
    }

    fun close() {
        runCatching { liveLandmarker?.close() }
        runCatching { imageLandmarker?.close() }
        liveLandmarker = null
        imageLandmarker = null
    }

    companion object {
        private const val TAG = "FaceLandmarkerHelper"

        fun sampleSkinRgb(bitmap: Bitmap, uvXY: FloatArray): FloatArray {
            if (bitmap.isRecycled || bitmap.width <= 0 || bitmap.height <= 0) {
                return floatArrayOf(0.5f, 0.45f, 0.42f)
            }
            var r = 0f
            var g = 0f
            var b = 0f
            var count = 0
            val maxW = bitmap.width - 1
            val maxH = bitmap.height - 1
            for (idx in FaceMeshTriangulation.skinSampleIndices) {
                if (idx * 2 + 1 >= uvXY.size) continue
                val px = (uvXY[idx * 2] * maxW).toInt().coerceIn(0, maxW)
                val py = (uvXY[idx * 2 + 1] * maxH).toInt().coerceIn(0, maxH)
                val pixel = bitmap.getPixel(px, py)
                r += Color.red(pixel) / 255f
                g += Color.green(pixel) / 255f
                b += Color.blue(pixel) / 255f
                count++
            }
            if (count == 0) return floatArrayOf(0.5f, 0.45f, 0.42f)
            return floatArrayOf(
                (r / count).coerceAtLeast(0.08f),
                (g / count).coerceAtLeast(0.08f),
                (b / count).coerceAtLeast(0.08f),
            )
        }
    }
}
