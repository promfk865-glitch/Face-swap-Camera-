package com.example.camera

import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import android.util.Log
import android.util.Rational
import android.view.Surface
import androidx.camera.core.CameraEffect
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.UseCaseGroup
import androidx.camera.core.ViewPort
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.FallbackStrategy
import androidx.camera.video.MediaStoreOutputOptions
import androidx.camera.video.QualitySelector
import androidx.camera.video.Recorder
import androidx.camera.video.Recording
import androidx.camera.video.VideoCapture
import androidx.camera.video.VideoRecordEvent
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.util.Consumer
import androidx.lifecycle.LifecycleOwner
import com.example.gl.FaceSwapSurfaceProcessor
import com.example.model.VideoResolution
import com.example.vision.FaceLandmarkerHelper
import java.text.SimpleDateFormat
import java.util.Locale
import java.util.concurrent.Executor
import java.util.concurrent.Executors

private class FaceSwapCameraEffect(
    targets: Int,
    executor: Executor,
    processor: FaceSwapSurfaceProcessor,
    errorListener: Consumer<Throwable>,
) : CameraEffect(targets, executor, processor, errorListener)

class CameraEngine(
    private val context: Context,
    val surfaceProcessor: FaceSwapSurfaceProcessor,
    private val faceLandmarkerHelper: FaceLandmarkerHelper,
) {
    private val analysisExecutor = Executors.newSingleThreadExecutor()
    private var videoCapture: VideoCapture<Recorder>? = null
    private var activeRecording: Recording? = null
    private var cameraProvider: ProcessCameraProvider? = null

    fun bindCamera(
        lifecycleOwner: LifecycleOwner,
        previewView: PreviewView,
        isFrontCamera: Boolean,
        resolution: VideoResolution,
        onError: (String) -> Unit,
    ) {
        val providerFuture = ProcessCameraProvider.getInstance(context)
        providerFuture.addListener({
            try {
                val provider = providerFuture.get()
                cameraProvider = provider

                val resolutionSelector = ResolutionSelector.Builder()
                    .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                    .build()

                val preview = Preview.Builder()
                    .setResolutionSelector(resolutionSelector)
                    .build()
                    .apply {
                        setSurfaceProvider(previewView.surfaceProvider)
                    }

                val imageAnalysis = ImageAnalysis.Builder()
                    .setResolutionSelector(resolutionSelector)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .build()
                    .apply {
                        setAnalyzer(analysisExecutor) { imageProxy ->
                            faceLandmarkerHelper.detectLiveFrame(imageProxy)
                        }
                    }

                surfaceProcessor.isFrontCamera = isFrontCamera

                val recorder = Recorder.Builder()
                    .setQualitySelector(
                        QualitySelector.from(
                            resolution.cameraQuality,
                            FallbackStrategy.higherQualityOrLowerThan(resolution.cameraQuality),
                        ),
                    )
                    .build()
                val video = VideoCapture.Builder(recorder)
                    .setMirrorMode(androidx.camera.core.MirrorMode.MIRROR_MODE_OFF)
                    .build()
                videoCapture = video

                val effect = FaceSwapCameraEffect(
                    targets = CameraEffect.PREVIEW or CameraEffect.VIDEO_CAPTURE,
                    executor = surfaceProcessor.glExecutor,
                    processor = surfaceProcessor,
                    errorListener = Consumer { err ->
                        Log.e(TAG, "CameraEffect error", err)
                    },
                )

                val rotation = previewView.display?.rotation ?: Surface.ROTATION_0
                val viewPort = previewView.viewPort
                    ?: ViewPort.Builder(Rational(3, 4), rotation)
                        .setScaleType(ViewPort.FIT)
                        .build()

                val useCaseGroup = UseCaseGroup.Builder()
                    .setViewPort(viewPort)
                    .addUseCase(preview)
                    .addUseCase(imageAnalysis)
                    .addUseCase(video)
                    .addEffect(effect)
                    .build()

                val selector = if (isFrontCamera) {
                    CameraSelector.DEFAULT_FRONT_CAMERA
                } else {
                    CameraSelector.DEFAULT_BACK_CAMERA
                }

                provider.unbindAll()
                provider.bindToLifecycle(lifecycleOwner, selector, useCaseGroup)
            } catch (t: Throwable) {
                Log.e(TAG, "Camera bind failed", t)
                onError(t.localizedMessage ?: "Camera initialization failed")
            }
        }, ContextCompat.getMainExecutor(context))
    }

    @SuppressLint("MissingPermission")
    fun startRecording(
        hasAudioPermission: Boolean,
        onStarted: () -> Unit,
        onFinalized: (Uri?, String?) -> Unit,
    ) {
        val capture = videoCapture ?: return
        if (activeRecording != null) return

        val fileName = "FaceSwap_${
            SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(System.currentTimeMillis())
        }.mp4"

        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, fileName)
            put(MediaStore.MediaColumns.MIME_TYPE, "video/mp4")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.Video.Media.RELATIVE_PATH, "Movies/FaceSwapCam")
            }
        }

        val mediaStoreOutput = MediaStoreOutputOptions.Builder(
            context.contentResolver,
            MediaStore.Video.Media.EXTERNAL_CONTENT_URI,
        ).setContentValues(contentValues).build()

        var pending = capture.output.prepareRecording(context, mediaStoreOutput)
        if (hasAudioPermission) {
            pending = pending.withAudioEnabled()
        }

        try {
            activeRecording = pending.start(ContextCompat.getMainExecutor(context)) { event ->
                when (event) {
                    is VideoRecordEvent.Start -> onStarted()
                    is VideoRecordEvent.Finalize -> {
                        activeRecording = null
                        if (event.hasError()) {
                            onFinalized(null, "Recording failed (code ${event.error})")
                        } else {
                            onFinalized(event.outputResults.outputUri, null)
                        }
                    }
                }
            }
        } catch (t: Throwable) {
            activeRecording = null
            Log.e(TAG, "Failed to start recording", t)
            onFinalized(null, t.localizedMessage ?: "Unable to start recording")
        }
    }

    fun stopRecording() {
        activeRecording?.stop()
    }

    fun release() {
        runCatching { activeRecording?.stop() }
        runCatching { cameraProvider?.unbindAll() }
        analysisExecutor.shutdown()
    }

    companion object {
        private const val TAG = "CameraEngine"
    }
}
