package com.example.viewmodel

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.R
import com.example.camera.CameraEngine
import com.example.data.SettingsRepository
import com.example.gl.FaceSwapSurfaceProcessor
import com.example.model.BilingualMessage
import com.example.model.CameraSettings
import com.example.model.CharacterFace
import com.example.model.FaceSource
import com.example.model.FaceSwapUiState
import com.example.model.VideoResolution
import com.example.model.VideoStudioState
import com.example.vision.FaceLandmarkerHelper
import com.example.vision.VideoFaceSwapEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.math.max

class FaceSwapViewModel(application: Application) : AndroidViewModel(application) {

    private val settingsRepo = SettingsRepository(application)
    val surfaceProcessor = FaceSwapSurfaceProcessor()

    val faceLandmarkerHelper = FaceLandmarkerHelper(
        context = application,
        onLiveResult = { result ->
            surfaceProcessor.updateLiveTracking(result)
            if (_uiState.value.isFaceTracking != result.hasFace) {
                _uiState.update { it.copy(isFaceTracking = result.hasFace) }
            }
        },
    )

    val cameraEngine = CameraEngine(
        context = application,
        surfaceProcessor = surfaceProcessor,
        faceLandmarkerHelper = faceLandmarkerHelper,
    )

    val videoSwapEngine = VideoFaceSwapEngine(application)

    private val _uiState = MutableStateFlow(FaceSwapUiState())
    val uiState: StateFlow<FaceSwapUiState> = _uiState.asStateFlow()

    private var timerJob: Job? = null
    private var videoScanJob: Job? = null
    private var videoPlayJob: Job? = null
    private var videoPreviewJob: Job? = null

    init {
        observeSettings()
        initializeVisionAndDemoFaces()
    }

    private fun observeSettings() {
        viewModelScope.launch {
            settingsRepo.settingsFlow.collect { settings ->
                faceLandmarkerHelper.smoothingStrength = settings.smoothingStrength
                surfaceProcessor.blendOpacity = settings.blendOpacity
                surfaceProcessor.userMirrored = settings.isMirrored
                surfaceProcessor.isFrontCamera = settings.isFrontCamera
                _uiState.update { it.copy(settings = settings) }
            }
        }
    }

    private fun initializeVisionAndDemoFaces() {
        val seedFaces = listOf(
            CharacterFace("cyber_aria", "Cyber Aria", "সাইবার আরিয়া", "Realistic", "বাস্তব", FaceSource.BundledDrawable(R.drawable.img_char_cyber_aria)),
            CharacterFace("studio_dev", "Studio Dev", "স্টুডিও দেব", "Realistic", "বাস্তব", FaceSource.BundledDrawable(R.drawable.img_char_studio_dev)),
            CharacterFace("neon_pixar", "Neon 3D", "নিয়ন ৩ডি", "3D Style", "৩ডি", FaceSource.BundledDrawable(R.drawable.img_char_neon_pixar)),
            CharacterFace("royal_knight", "Royal Hero", "রয়্যাল হিরো", "Fantasy", "ফ্যান্টাসি", FaceSource.BundledDrawable(R.drawable.img_char_royal_knight)),
            CharacterFace("retro_pop", "Cel Pop", "সেল পপ", "Stylized", "স্টাইল", FaceSource.BundledDrawable(R.drawable.img_char_retro_pop)),
            CharacterFace("cosmic_sage", "Cosmic Sage", "কসমিক সেজ", "Editorial", "এডিটোরিয়াল", FaceSource.BundledDrawable(R.drawable.img_char_cosmic_sage)),
        )
        _uiState.update { it.copy(faces = seedFaces) }

        viewModelScope.launch(Dispatchers.Default) {
            val initialized = faceLandmarkerHelper.initialize()
            if (!initialized) {
                _uiState.update {
                    it.copy(
                        isModelReady = false,
                        errorMessage = BilingualMessage(
                            "Failed to initialize MediaPipe Face Landmarker.",
                            "মিডিয়াপাইপ ফেস ল্যান্ডমার্কার চালু করা যায়নি।",
                        ),
                    )
                }
                return@launch
            }

            _uiState.update { it.copy(isModelReady = true) }

            val ctx = getApplication<Application>()
            val loadedList = seedFaces.map { face ->
                val res = (face.source as FaceSource.BundledDrawable).resId
                val bmp = decodeScaledDrawable(ctx.resources, res, 768)
                val uvs = bmp?.let { faceLandmarkerHelper.detectStaticImage(it) }
                face.copy(bitmap = bmp, uvLandmarks = uvs)
            }

            _uiState.update { it.copy(faces = loadedList) }

            val activeId = _uiState.value.selectedFaceId
            val activeFace = loadedList.firstOrNull { it.id == activeId }
            if (activeFace != null && activeFace.isReady) {
                surfaceProcessor.setCharacterFace(activeFace.bitmap, activeFace.uvLandmarks)
            }
        }
    }

    fun selectFace(faceId: String?) {
        _uiState.update { it.copy(selectedFaceId = faceId) }
        if (faceId == null) {
            surfaceProcessor.setCharacterFace(null, null)
            return
        }
        val selected = _uiState.value.faces.firstOrNull { it.id == faceId }
        if (selected != null && selected.isReady) {
            surfaceProcessor.setCharacterFace(selected.bitmap, selected.uvLandmarks)
        }
    }

    fun uploadCustomFace(uri: Uri) {
        viewModelScope.launch {
            _uiState.update { it.copy(isValidatingUpload = true, errorMessage = null) }
            val result = withContext(Dispatchers.IO) {
                val bmp = decodeScaledUri(uri, 768) ?: return@withContext null
                val uvs = faceLandmarkerHelper.detectStaticImage(bmp)
                if (uvs == null) {
                    bmp.recycle()
                    Pair<Bitmap?, FloatArray?>(null, null)
                } else {
                    Pair<Bitmap?, FloatArray?>(bmp, uvs)
                }
            }

            if (result == null) {
                _uiState.update {
                    it.copy(
                        isValidatingUpload = false,
                        errorMessage = BilingualMessage(
                            "Unable to read the selected photo.",
                            "নির্বাচিত ছবিটি পড়া যায়নি।",
                        ),
                    )
                }
                return@launch
            }

            val (bmp, uvs) = result
            if (bmp == null || uvs == null) {
                _uiState.update {
                    it.copy(
                        isValidatingUpload = false,
                        errorMessage = BilingualMessage(
                            "No face detected in photo. Choose a clear front-facing portrait.",
                            "ছবিতে কোনো মুখ শনাক্ত হয়নি। পরিষ্কার সামনের দিকের ছবি দিন।",
                        ),
                    )
                }
                return@launch
            }

            val newId = "upload_${System.currentTimeMillis()}"
            val newFace = CharacterFace(
                id = newId,
                nameEn = "Custom Face",
                nameBn = "কাস্টম মুখ",
                styleEn = "Gallery",
                styleBn = "গ্যালারি",
                source = FaceSource.UploadedUri(uri),
                bitmap = bmp,
                uvLandmarks = uvs,
            )

            _uiState.update { state ->
                state.copy(
                    faces = state.faces + newFace,
                    selectedFaceId = newId,
                    isValidatingUpload = false,
                )
            }
            surfaceProcessor.setCharacterFace(bmp, uvs)
        }
    }

    fun toggleRecording(hasAudioPermission: Boolean) {
        if (_uiState.value.isRecording) {
            cameraEngine.stopRecording()
            timerJob?.cancel()
            _uiState.update { it.copy(isRecording = false) }
            return
        }

        _uiState.update { it.copy(lastSavedVideoUri = null, errorMessage = null) }
        cameraEngine.startRecording(
            hasAudioPermission = hasAudioPermission,
            onStarted = {
                _uiState.update { it.copy(isRecording = true, recordingDurationMs = 0L) }
                timerJob?.cancel()
                timerJob = viewModelScope.launch {
                    val startMs = System.currentTimeMillis()
                    while (isActive) {
                        _uiState.update { it.copy(recordingDurationMs = System.currentTimeMillis() - startMs) }
                        delay(200L)
                    }
                }
            },
            onFinalized = { uri, err ->
                timerJob?.cancel()
                _uiState.update { state ->
                    state.copy(
                        isRecording = false,
                        recordingDurationMs = 0L,
                        lastSavedVideoUri = uri,
                        errorMessage = err?.let {
                            BilingualMessage(it, "ভিডিও সংরক্ষণে সমস্যা হয়েছে।")
                        },
                    )
                }
            },
        )
    }

    fun flipCamera() {
        val next = !_uiState.value.settings.isFrontCamera
        viewModelScope.launch { settingsRepo.setFrontCamera(next) }
    }

    fun setMirrored(mirrored: Boolean) {
        viewModelScope.launch { settingsRepo.setMirrored(mirrored) }
    }

    fun setSmoothingStrength(value: Float) {
        viewModelScope.launch { settingsRepo.setSmoothingStrength(value) }
    }

    fun setBlendOpacity(value: Float) {
        viewModelScope.launch { settingsRepo.setBlendOpacity(value) }
    }

    fun setResolution(resolution: VideoResolution) {
        viewModelScope.launch { settingsRepo.setResolution(resolution) }
    }

    fun setShowSettings(show: Boolean) {
        _uiState.update { it.copy(showSettings = show) }
    }

    fun setShowGuide(show: Boolean) {
        _uiState.update { it.copy(showGuide = show) }
    }

    fun dismissError() {
        _uiState.update { it.copy(errorMessage = null) }
    }

    fun dismissSavedVideoBanner() {
        _uiState.update { it.copy(lastSavedVideoUri = null) }
    }

    fun reportError(en: String, bn: String) {
        _uiState.update { it.copy(errorMessage = BilingualMessage(en, bn)) }
    }

    fun openVideoForSwap(uri: Uri) {
        videoScanJob?.cancel()
        videoPlayJob?.cancel()
        _uiState.update {
            it.copy(
                errorMessage = null,
                videoStudio = VideoStudioState(
                    videoUri = uri,
                    isScanning = true,
                    scanProgress = 0,
                ),
            )
        }
        videoScanJob = viewModelScope.launch(Dispatchers.IO) {
            val (durationMs, people) = videoSwapEngine.scanVideoForPeople(uri) { pct ->
                _uiState.update { s ->
                    s.copy(videoStudio = s.videoStudio.copy(scanProgress = pct))
                }
            }
            val firstBmp = videoSwapEngine.renderPreviewBitmap(
                uri = uri,
                timeMs = 0L,
                people = people,
                facesById = _uiState.value.faces.associateBy { it.id },
                opacity = _uiState.value.settings.blendOpacity,
            )
            _uiState.update { state ->
                state.copy(
                    errorMessage = if (people.isEmpty()) {
                        BilingualMessage(
                            "No faces detected in this video.",
                            "এই ভিডিওতে কোনো মুখ পাওয়া যায়নি।",
                        )
                    } else {
                        state.errorMessage
                    },
                    videoStudio = state.videoStudio.copy(
                        durationMs = durationMs,
                        detectedPeople = people,
                        selectedPersonId = people.firstOrNull()?.id,
                        previewBitmap = firstBmp,
                        isScanning = false,
                        scanProgress = 100,
                    ),
                )
            }
        }
    }

    fun selectVideoPerson(personId: String) {
        _uiState.update { s ->
            s.copy(videoStudio = s.videoStudio.copy(selectedPersonId = personId))
        }
    }

    fun assignFaceToSelectedVideoPerson(faceId: String?) {
        val targetPersonId = _uiState.value.videoStudio.selectedPersonId ?: return
        _uiState.update { s ->
            val updated = s.videoStudio.detectedPeople.map { p ->
                if (p.id == targetPersonId) p.copy(assignedFaceId = faceId) else p
            }
            s.copy(videoStudio = s.videoStudio.copy(detectedPeople = updated))
        }
        refreshVideoPreviewFrame()
    }

    fun seekVideoPreview(positionMs: Long) {
        _uiState.update { s ->
            s.copy(videoStudio = s.videoStudio.copy(previewPositionMs = positionMs))
        }
        refreshVideoPreviewFrame()
    }

    fun toggleVideoPlayPreview() {
        val studio = _uiState.value.videoStudio
        if (studio.isPlayingPreview) {
            videoPlayJob?.cancel()
            _uiState.update { s -> s.copy(videoStudio = s.videoStudio.copy(isPlayingPreview = false)) }
            return
        }
        val uri = studio.videoUri ?: return
        _uiState.update { s -> s.copy(videoStudio = s.videoStudio.copy(isPlayingPreview = true)) }
        videoPlayJob?.cancel()
        videoPlayJob = viewModelScope.launch(Dispatchers.IO) {
            var pos = _uiState.value.videoStudio.previewPositionMs
            val dur = _uiState.value.videoStudio.durationMs.coerceAtLeast(100L)
            if (pos >= dur - 120L) pos = 0L
            while (isActive && _uiState.value.videoStudio.isPlayingPreview) {
                val curState = _uiState.value
                val bmp = videoSwapEngine.renderPreviewBitmap(
                    uri = uri,
                    timeMs = pos,
                    people = curState.videoStudio.detectedPeople,
                    facesById = curState.faces.associateBy { it.id },
                    opacity = curState.settings.blendOpacity,
                )
                _uiState.update { s ->
                    s.copy(
                        videoStudio = s.videoStudio.copy(
                            previewPositionMs = pos,
                            previewBitmap = bmp ?: s.videoStudio.previewBitmap,
                        ),
                    )
                }
                pos += 140L
                if (pos >= dur) {
                    _uiState.update { s -> s.copy(videoStudio = s.videoStudio.copy(isPlayingPreview = false, previewPositionMs = 0L)) }
                    break
                }
                delay(120L)
            }
        }
    }

    private fun refreshVideoPreviewFrame() {
        val studio = _uiState.value.videoStudio
        val uri = studio.videoUri ?: return
        videoPreviewJob?.cancel()
        videoPreviewJob = viewModelScope.launch(Dispatchers.IO) {
            val state = _uiState.value
            val bmp = videoSwapEngine.renderPreviewBitmap(
                uri = uri,
                timeMs = state.videoStudio.previewPositionMs,
                people = state.videoStudio.detectedPeople,
                facesById = state.faces.associateBy { it.id },
                opacity = state.settings.blendOpacity,
            )
            if (bmp != null) {
                _uiState.update { s ->
                    s.copy(videoStudio = s.videoStudio.copy(previewBitmap = bmp))
                }
            }
        }
    }

    fun exportSwappedVideo() {
        val studio = _uiState.value.videoStudio
        val uri = studio.videoUri ?: return
        videoPlayJob?.cancel()
        _uiState.update { s ->
            s.copy(
                videoStudio = s.videoStudio.copy(
                    isPlayingPreview = false,
                    isExporting = true,
                    exportProgress = 0,
                ),
            )
        }
        viewModelScope.launch(Dispatchers.IO) {
            val state = _uiState.value
            val savedUri = videoSwapEngine.exportSwappedVideo(
                uri = uri,
                people = state.videoStudio.detectedPeople,
                facesById = state.faces.associateBy { it.id },
                opacity = state.settings.blendOpacity,
                smoothingStrength = state.settings.smoothingStrength,
                onProgress = { pct ->
                    _uiState.update { s ->
                        s.copy(videoStudio = s.videoStudio.copy(exportProgress = pct))
                    }
                },
            )
            _uiState.update { s ->
                s.copy(
                    lastSavedVideoUri = savedUri ?: s.lastSavedVideoUri,
                    errorMessage = if (savedUri == null) {
                        BilingualMessage("Failed to save swapped video.", "ভিডিও সেভ করতে সমস্যা হয়েছে।")
                    } else {
                        s.errorMessage
                    },
                    videoStudio = s.videoStudio.copy(
                        isExporting = false,
                        exportProgress = 100,
                    ),
                )
            }
        }
    }

    fun closeVideoStudio() {
        videoScanJob?.cancel()
        videoPlayJob?.cancel()
        videoPreviewJob?.cancel()
        _uiState.update { s -> s.copy(videoStudio = VideoStudioState()) }
    }

    private fun decodeScaledDrawable(
        resources: android.content.res.Resources,
        resId: Int,
        maxDim: Int,
    ): Bitmap? {
        return runCatching {
            val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            BitmapFactory.decodeResource(resources, resId, opts)
            val sample = calculateSampleSize(opts.outWidth, opts.outHeight, maxDim)
            val decodeOpts = BitmapFactory.Options().apply {
                inSampleSize = sample
                inPreferredConfig = Bitmap.Config.ARGB_8888
            }
            BitmapFactory.decodeResource(resources, resId, decodeOpts)
        }.getOrNull()
    }

    private fun decodeScaledUri(uri: Uri, maxDim: Int): Bitmap? {
        val resolver = getApplication<Application>().contentResolver
        return runCatching {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                val src = ImageDecoder.createSource(resolver, uri)
                ImageDecoder.decodeBitmap(src) { decoder, info, _ ->
                    decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                    val longest = max(info.size.width, info.size.height)
                    if (longest > maxDim) {
                        val scale = maxDim.toFloat() / longest
                        decoder.setTargetSize(
                            (info.size.width * scale).toInt().coerceAtLeast(1),
                            (info.size.height * scale).toInt().coerceAtLeast(1),
                        )
                    }
                }
            } else {
                resolver.openInputStream(uri)?.use { stream ->
                    BitmapFactory.decodeStream(stream)
                }
            }
        }.getOrNull()
    }

    private fun calculateSampleSize(width: Int, height: Int, maxDim: Int): Int {
        var sample = 1
        while (max(width, height) / sample > maxDim) {
            sample *= 2
        }
        return sample
    }

    override fun onCleared() {
        super.onCleared()
        timerJob?.cancel()
        videoScanJob?.cancel()
        videoPlayJob?.cancel()
        videoPreviewJob?.cancel()
        cameraEngine.release()
        faceLandmarkerHelper.close()
        videoSwapEngine.close()
        surfaceProcessor.release()
    }
}
