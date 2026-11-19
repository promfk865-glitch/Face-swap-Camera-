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
                .setMinFaceDetectionConfidence(0.5f)
                .setMinFacePresenceConfidence(0.5f)
                .setMinTrackingConfidence(0.5f)
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
                .setMinFaceDetectionConfidence(0.45f)
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
        val landmarks = result.faceLandmarks().firstOrNull()
        if (landmarks == null || landmarks.isEmpty()) {
            latestRawXY = null
            cachedLiveSkinRgb = null
            smoother.reset()
            onLiveResult(
                LiveFaceTrackingResult(
                    smoothedXY = FloatArray(0),
                    landmarkCount = 0,
                    jawOpen = 0f,
                    hasFace = false,
                    liveSkinRgb = null,
                ),
            )
            return
        }

        val count = landmarks.size
        val raw = FloatArray(count * 2)
        for (i in 0 until count) {
            raw[i * 2] = landmarks[i].x()
            raw[i * 2 + 1] = landmarks[i].y()
        }
        latestRawXY = raw

        val nowMs = SystemClock.uptimeMillis()
        val smoothed = smoother.smooth(raw, count, nowMs, smoothingStrength)

        val rawJaw = result.faceBlendshapes()
            .orElse(emptyList())
            .firstOrNull()
            ?.firstOrNull { it.categoryName() == "jawOpen" }
            ?.score()
            ?: 0f
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
