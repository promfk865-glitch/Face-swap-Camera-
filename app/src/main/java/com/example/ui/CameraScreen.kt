package com.example.ui

import android.content.Intent
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.MenuBook
import androidx.compose.material.icons.filled.AddPhotoAlternate
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Face
import androidx.compose.material.icons.filled.FlipCameraAndroid
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Share
import androidx.compose.material.icons.filled.VideoLibrary
import androidx.compose.material.icons.filled.WarningAmber
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.minimumInteractiveComponentSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.compose.LocalLifecycleOwner
import coil.compose.AsyncImage
import com.example.R
import com.example.model.CharacterFace
import com.example.model.FaceSource
import com.example.model.FaceSwapUiState
import com.example.ui.theme.CoralCrimson
import com.example.ui.theme.CoralCrimsonDim
import com.example.ui.theme.MintCyan
import com.example.ui.theme.MintCyanDim
import com.example.ui.theme.ObsidianBg
import com.example.ui.theme.SurfaceCard
import com.example.ui.theme.SurfaceGlass
import com.example.ui.theme.SurfaceStroke
import com.example.ui.theme.TextMuted
import com.example.ui.theme.TextPrimary
import com.example.viewmodel.FaceSwapViewModel

@Composable
fun CameraScreen(
    uiState: FaceSwapUiState,
    hasCameraPermission: Boolean,
    hasAudioPermission: Boolean,
    showAppSettingsShortcut: Boolean,
    onRequestPermissions: () -> Unit,
    onOpenAppSettings: () -> Unit,
    viewModel: FaceSwapViewModel,
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    val fallbackContentPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent(),
        onResult = { uri ->
            uri?.let { viewModel.uploadCustomFace(it) }
        },
    )

    val photoPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia(),
        onResult = { uri ->
            uri?.let { viewModel.uploadCustomFace(it) }
        },
    )

    val fallbackVideoPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent(),
        onResult = { uri ->
            uri?.let { viewModel.openVideoForSwap(it) }
        },
    )

    val videoPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia(),
        onResult = { uri ->
            uri?.let { viewModel.openVideoForSwap(it) }
        },
    )

    val previewView = remember {
        PreviewView(context).apply {
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            scaleType = PreviewView.ScaleType.FIT_CENTER
        }
    }

    LaunchedEffect(
        hasCameraPermission,
        uiState.settings.isFrontCamera,
        uiState.settings.resolution,
    ) {
        if (hasCameraPermission) {
            viewModel.cameraEngine.bindCamera(
                lifecycleOwner = lifecycleOwner,
                previewView = previewView,
                isFrontCamera = uiState.settings.isFrontCamera,
                resolution = uiState.settings.resolution,
                onError = { msg ->
                    viewModel.reportError(msg, "ক্যামেরা চালু করতে সমস্যা হয়েছে।")
                },
            )
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(ObsidianBg),
    ) {
        if (hasCameraPermission) {
            AndroidView(
                factory = { previewView },
                modifier = Modifier.fillMaxSize(),
            )
        } else {
            PermissionPromptOverlay(
                showSettingsShortcut = showAppSettingsShortcut,
                onGrantClick = onRequestPermissions,
                onOpenSettingsClick = onOpenAppSettings,
            )
        }

        // Top Translucent Header
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .align(Alignment.TopCenter)
                .background(
                    Brush.verticalGradient(
                        listOf(ObsidianBg.copy(alpha = 0.86f), Color.Transparent),
                    ),
                )
                .windowInsetsPadding(WindowInsets.safeDrawing)
                .padding(horizontal = 16.dp, vertical = 10.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                    ) {
                        Icon(
                            imageVector = Icons.Filled.Face,
                            contentDescription = null,
                            tint = MintCyan,
                            modifier = Modifier.size(18.dp),
                        )
                        Text(
                            text = "FaceSwap Cam",
                            style = MaterialTheme.typography.titleMedium,
                            color = TextPrimary,
                        )
                        Text(
                            text = "• ফেসসোয়াপ ক্যাম",
                            style = MaterialTheme.typography.labelSmall,
                            color = TextMuted,
                        )
                    }

                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        StatusBadge(
                            isTracking = uiState.isFaceTracking,
                            isModelReady = uiState.isModelReady,
                        )
                        if (uiState.isRecording) {
                            RecordingBadge(durationMs = uiState.recordingDurationMs)
                        }
                    }
                }

                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    GlassIconButton(
                        onClick = { viewModel.flipCamera() },
                        contentDescription = "Flip Camera • ক্যামেরা ফ্লিপ",
                        testTag = "flip_camera_button",
                    ) {
                        Icon(
                            imageVector = Icons.Filled.FlipCameraAndroid,
                            contentDescription = null,
                            tint = TextPrimary,
                        )
                    }

                    GlassIconButton(
                        onClick = { viewModel.setShowGuide(true) },
                        contentDescription = "Face Guide • ফেস গাইড",
                        testTag = "guide_button",
                    ) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.MenuBook,
                            contentDescription = null,
                            tint = TextPrimary,
                        )
                    }

                    GlassIconButton(
                        onClick = { viewModel.setShowSettings(true) },
                        contentDescription = "Settings • সেটিংস",
                        testTag = "settings_button",
                    ) {
                        Icon(
                            imageVector = Icons.Filled.Settings,
                            contentDescription = null,
                            tint = TextPrimary,
                        )
                    }
                }
            }
        }

        // Error & Saved Video Banners
        Column(
            modifier = Modifier
                .align(Alignment.TopCenter)
                .windowInsetsPadding(WindowInsets.safeDrawing)
                .padding(top = 84.dp, start = 16.dp, end = 16.dp)
                .widthIn(max = 560.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            AnimatedVisibility(
                visible = uiState.errorMessage != null,
                enter = fadeIn(),
                exit = fadeOut(),
            ) {
                uiState.errorMessage?.let { msg ->
                    Surface(
                        color = Color(0xEE2A0C14),
                        shape = RoundedCornerShape(16.dp),
                        border = BorderStroke(1.dp, CoralCrimson),
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        Row(
                            modifier = Modifier.padding(12.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(10.dp),
                        ) {
                            Icon(
                                imageVector = Icons.Filled.WarningAmber,
                                contentDescription = null,
                                tint = CoralCrimson,
                            )
                            Column(modifier = Modifier.weight(1f)) {
                                Text(
                                    text = msg.en,
                                    style = MaterialTheme.typography.labelMedium,
                                    color = TextPrimary,
                                )
                                Text(
                                    text = msg.bn,
                                    style = MaterialTheme.typography.labelSmall,
                                    color = Color(0xFFFFD6DD),
                                )
                            }
                            IconButton(
                                onClick = { viewModel.dismissError() },
                                modifier = Modifier.minimumInteractiveComponentSize(),
                            ) {
                                Icon(
                                    imageVector = Icons.Filled.Close,
                                    contentDescription = "Dismiss • বন্ধ করুন",
                                    tint = TextPrimary,
                                )
                            }
                        }
                    }
                }
            }

            AnimatedVisibility(
                visible = uiState.lastSavedVideoUri != null,
                enter = fadeIn(),
                exit = fadeOut(),
            ) {
                uiState.lastSavedVideoUri?.let { videoUri ->
                    SavedVideoBanner(
                        onPlay = {
                            val intent = Intent(Intent.ACTION_VIEW).apply {
                                setDataAndType(videoUri, "video/mp4")
                                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                            }
                            runCatching { context.startActivity(intent) }
                        },
                        onShare = {
                            val intent = Intent(Intent.ACTION_SEND).apply {
                                type = "video/mp4"
                                putExtra(Intent.EXTRA_STREAM, videoUri)
                                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                            }
                            runCatching {
                                context.startActivity(Intent.createChooser(intent, "Share Video • ভিডিও শেয়ার"))
                            }
                        },
                        onDismiss = { viewModel.dismissSavedVideoBanner() },
                    )
                }
            }
        }

        // Bottom Character Carousel + Record Deck
        if (hasCameraPermission) {
            Column(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .fillMaxWidth()
                    .background(
                        Brush.verticalGradient(
                            listOf(Color.Transparent, ObsidianBg.copy(alpha = 0.94f)),
                        ),
                    )
                    .navigationBarsPadding()
                    .padding(bottom = 20.dp, top = 16.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(16.dp),
            ) {
                LazyRow(
                    modifier = Modifier
                        .fillMaxWidth()
                        .testTag("character_carousel"),
                    contentPadding = PaddingValues(horizontal = 16.dp),
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    item(key = "original") {
                        OriginalFeedChip(
                            isSelected = uiState.selectedFaceId == null,
                            onClick = { viewModel.selectFace(null) },
                        )
                    }

                    items(uiState.faces, key = { it.id }) { face ->
                        CharacterFaceCard(
                            face = face,
                            isSelected = uiState.selectedFaceId == face.id,
                            onClick = { viewModel.selectFace(face.id) },
                        )
                    }

                    item(key = "upload") {
                        UploadFaceChip(
                            isValidating = uiState.isValidatingUpload,
                            onClick = {
                                runCatching {
                                    photoPickerLauncher.launch(
                                        PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly),
                                    )
                                }.onFailure {
                                    fallbackContentPicker.launch("image/*")
                                }
                            },
                        )
                    }
                }

                if (!hasAudioPermission && !uiState.isRecording) {
                    Surface(
                        onClick = if (showAppSettingsShortcut) onOpenAppSettings else onRequestPermissions,
                        color = SurfaceGlass,
                        shape = RoundedCornerShape(999.dp),
                        border = BorderStroke(1.dp, SurfaceStroke),
                    ) {
                        Text(
                            text = "Mic off (tap to allow audio) • মাইক বন্ধ (অডিও চালু করতে ট্যাপ করুন)",
                            style = MaterialTheme.typography.labelSmall,
                            color = TextMuted,
                            modifier = Modifier.padding(horizontal = 12.dp, vertical = 4.dp),
                        )
                    }
                }

                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(22.dp),
                ) {
                    Surface(
                        onClick = {
                            runCatching {
                                videoPickerLauncher.launch(
                                    PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.VideoOnly),
                                )
                            }.onFailure {
                                fallbackVideoPicker.launch("video/*")
                            }
                        },
                        color = SurfaceGlass,
                        shape = RoundedCornerShape(14.dp),
                        border = BorderStroke(1.dp, MintCyan.copy(alpha = 0.4f)),
                        modifier = Modifier.testTag("video_swap_button"),
                    ) {
                        Column(
                            modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
                            horizontalAlignment = Alignment.CenterHorizontally,
                            verticalArrangement = Arrangement.spacedBy(2.dp),
                        ) {
                            Icon(
                                imageVector = Icons.Filled.VideoLibrary,
                                contentDescription = "Video Face Swap • ভিডিও সোয়াপ",
                                tint = MintCyan,
                                modifier = Modifier.size(18.dp),
                            )
                            Text(
                                text = "Video • ভিডিও",
                                style = MaterialTheme.typography.labelSmall,
                                fontSize = 9.sp,
                                color = TextPrimary,
                            )
                        }
                    }

                    RecordShutterButton(
                        isRecording = uiState.isRecording,
                        onClick = {
                            viewModel.toggleRecording(hasAudioPermission = hasAudioPermission)
                        },
                    )

                    Spacer(modifier = Modifier.width(68.dp))
                }
            }
        }

        if (uiState.videoStudio.videoUri != null) {
            VideoSwapScreen(
                studioState = uiState.videoStudio,
                faces = uiState.faces,
                onBack = { viewModel.closeVideoStudio() },
                onSelectPerson = { viewModel.selectVideoPerson(it) },
                onAssignFaceToSelected = { viewModel.assignFaceToSelectedVideoPerson(it) },
                onUploadCharacterFace = { viewModel.uploadCustomFace(it) },
                onTogglePlay = { viewModel.toggleVideoPlayPreview() },
                onSeek = { viewModel.seekVideoPreview(it) },
                onExportVideo = { viewModel.exportSwappedVideo() },
            )
        }

        if (uiState.showSettings) {
            SettingsSheet(
                settings = uiState.settings,
                onFlipCamera = { viewModel.flipCamera() },
                onMirrorChanged = { viewModel.setMirrored(it) },
                onSmoothingChanged = { viewModel.setSmoothingStrength(it) },
                onOpacityChanged = { viewModel.setBlendOpacity(it) },
                onResolutionSelected = { viewModel.setResolution(it) },
                onDismiss = { viewModel.setShowSettings(false) },
            )
        }

        if (uiState.showGuide) {
            GuideScreen(onClose = { viewModel.setShowGuide(false) })
        }
    }
}

@Composable
private fun PermissionPromptOverlay(
    showSettingsShortcut: Boolean,
    onGrantClick: () -> Unit,
    onOpenSettingsClick: () -> Unit,
) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        contentAlignment = Alignment.Center,
    ) {
        Surface(
            color = SurfaceCard,
            shape = RoundedCornerShape(24.dp),
            border = BorderStroke(1.dp, SurfaceStroke),
            modifier = Modifier.widthIn(max = 420.dp),
        ) {
            Column(
                modifier = Modifier.padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(14.dp),
            ) {
                Box(
                    modifier = Modifier
                        .size(64.dp)
                        .clip(CircleShape)
                        .background(MintCyanDim),
                    contentAlignment = Alignment.Center,
                ) {
                    Icon(
                        imageVector = Icons.Filled.CameraAlt,
                        contentDescription = null,
                        tint = MintCyan,
                        modifier = Modifier.size(32.dp),
                    )
                }
                Text(
                    text = stringResource(R.string.permission_camera_title),
                    style = MaterialTheme.typography.titleMedium,
                    color = TextPrimary,
                )
                Text(
                    text = stringResource(R.string.permission_camera_desc),
                    style = MaterialTheme.typography.bodyMedium,
                    color = TextMuted,
                )
                Button(
                    onClick = onGrantClick,
                    colors = ButtonDefaults.buttonColors(
                        containerColor = MintCyan,
                        contentColor = ObsidianBg,
                    ),
                    shape = RoundedCornerShape(14.dp),
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(50.dp)
                        .testTag("grant_camera_button"),
                ) {
                    Text(
                        text = stringResource(R.string.permission_grant_btn),
                        fontWeight = FontWeight.Bold,
                    )
                }
                if (showSettingsShortcut) {
                    TextButton(onClick = onOpenSettingsClick) {
                        Text(
                            text = "Open App Settings • অ্যাপ সেটিংস খুলুন",
                            color = MintCyan,
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun StatusBadge(isTracking: Boolean, isModelReady: Boolean) {
    val dotColor = if (isTracking) MintCyan else TextMuted
    val label = when {
        isTracking -> "Tracking • ট্র্যাকিং"
        isModelReady -> "Find face • মুখ খুঁজছে"
        else -> "Loading AI • লোড হচ্ছে"
    }
    Surface(
        color = SurfaceGlass,
        shape = RoundedCornerShape(999.dp),
        border = BorderStroke(1.dp, if (isTracking) MintCyan.copy(alpha = 0.5f) else SurfaceStroke),
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            Box(
                modifier = Modifier
                    .size(7.dp)
                    .clip(CircleShape)
                    .background(dotColor),
            )
            Text(
                text = label,
                style = MaterialTheme.typography.labelSmall,
                color = if (isTracking) MintCyan else TextMuted,
            )
        }
    }
}

@Composable
private fun RecordingBadge(durationMs: Long) {
    val totalSec = (durationMs / 1000L).toInt()
    val formatted = "%02d:%02d".format(totalSec / 60, totalSec % 60)
    Surface(
        color = CoralCrimsonDim,
        shape = RoundedCornerShape(999.dp),
        border = BorderStroke(1.dp, CoralCrimson),
    ) {
        Text(
            text = "REC $formatted",
            style = MaterialTheme.typography.labelSmall,
            color = CoralCrimson,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp),
        )
    }
}

@Composable
private fun GlassIconButton(
    onClick: () -> Unit,
    contentDescription: String,
    testTag: String,
    content: @Composable () -> Unit,
) {
    Surface(
        onClick = onClick,
        color = SurfaceGlass,
        shape = RoundedCornerShape(14.dp),
        border = BorderStroke(1.dp, SurfaceStroke),
        modifier = Modifier
            .minimumInteractiveComponentSize()
            .size(48.dp)
            .testTag(testTag)
            .semantics(mergeDescendants = true) {},
    ) {
        Box(contentAlignment = Alignment.Center) {
            IconButton(onClick = onClick) {
                content()
            }
        }
    }
}

@Composable
private fun OriginalFeedChip(isSelected: Boolean, onClick: () -> Unit) {
    Surface(
        onClick = onClick,
        color = if (isSelected) MintCyanDim else SurfaceGlass,
        shape = RoundedCornerShape(18.dp),
        border = BorderStroke(if (isSelected) 2.dp else 1.dp, if (isSelected) MintCyan else SurfaceStroke),
        modifier = Modifier
            .width(72.dp)
            .height(92.dp)
            .testTag("face_card_original")
            .semantics(mergeDescendants = true) {},
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(6.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
        ) {
            Icon(
                imageVector = Icons.Filled.CameraAlt,
                contentDescription = "Original Camera • আসল ক্যামেরা",
                tint = if (isSelected) MintCyan else TextMuted,
                modifier = Modifier.size(24.dp),
            )
            Spacer(modifier = Modifier.height(6.dp))
            Text(
                text = "Original",
                style = MaterialTheme.typography.labelMedium,
                fontSize = 11.sp,
                color = TextPrimary,
                maxLines = 1,
            )
            Text(
                text = "আসল",
                style = MaterialTheme.typography.labelSmall,
                fontSize = 10.sp,
                color = TextMuted,
                maxLines = 1,
            )
        }
    }
}

@Composable
private fun CharacterFaceCard(
    face: CharacterFace,
    isSelected: Boolean,
    onClick: () -> Unit,
) {
    val scale by animateFloatAsState(
        targetValue = if (isSelected) 1.05f else 1.0f,
        animationSpec = spring(),
        label = "faceScale",
    )

    Surface(
        onClick = onClick,
        color = if (isSelected) MintCyanDim else SurfaceGlass,
        shape = RoundedCornerShape(18.dp),
        border = BorderStroke(if (isSelected) 2.dp else 1.dp, if (isSelected) MintCyan else SurfaceStroke),
        modifier = Modifier
            .scale(scale)
            .width(76.dp)
            .height(92.dp)
            .testTag("face_card_${face.id}")
            .semantics(mergeDescendants = true) {},
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(5.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.SpaceBetween,
        ) {
            when (val src = face.source) {
                is FaceSource.BundledDrawable -> {
                    Image(
                        painter = painterResource(id = src.resId),
                        contentDescription = "${face.nameEn} • ${face.nameBn}",
                        contentScale = ContentScale.Crop,
                        modifier = Modifier
                            .size(50.dp)
                            .clip(RoundedCornerShape(12.dp)),
                    )
                }
                is FaceSource.UploadedUri -> {
                    AsyncImage(
                        model = src.uri,
                        contentDescription = "${face.nameEn} • ${face.nameBn}",
                        contentScale = ContentScale.Crop,
                        modifier = Modifier
                            .size(50.dp)
                            .clip(RoundedCornerShape(12.dp)),
                    )
                }
            }
            Text(
                text = face.nameEn,
                style = MaterialTheme.typography.labelMedium,
                fontSize = 10.sp,
                color = if (isSelected) MintCyan else TextPrimary,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
            Text(
                text = face.nameBn,
                style = MaterialTheme.typography.labelSmall,
                fontSize = 9.sp,
                color = TextMuted,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
        }
    }
}

@Composable
private fun UploadFaceChip(isValidating: Boolean, onClick: () -> Unit) {
    Surface(
        onClick = onClick,
        color = SurfaceGlass,
        shape = RoundedCornerShape(18.dp),
        border = BorderStroke(1.dp, MintCyan.copy(alpha = 0.6f)),
        modifier = Modifier
            .width(76.dp)
            .height(92.dp)
            .testTag("upload_face_button")
            .semantics(mergeDescendants = true) {},
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(6.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
        ) {
            if (isValidating) {
                CircularProgressIndicator(
                    color = MintCyan,
                    strokeWidth = 2.dp,
                    modifier = Modifier.size(24.dp),
                )
            } else {
                Icon(
                    imageVector = Icons.Filled.AddPhotoAlternate,
                    contentDescription = "Upload Face • ছবি আপলোড",
                    tint = MintCyan,
                    modifier = Modifier.size(24.dp),
                )
            }
            Spacer(modifier = Modifier.height(6.dp))
            Text(
                text = "Upload",
                style = MaterialTheme.typography.labelMedium,
                fontSize = 11.sp,
                color = MintCyan,
                maxLines = 1,
            )
            Text(
                text = "আপলোড",
                style = MaterialTheme.typography.labelSmall,
                fontSize = 10.sp,
                color = TextMuted,
                maxLines = 1,
            )
        }
    }
}

@Composable
private fun RecordShutterButton(
    isRecording: Boolean,
    onClick: () -> Unit,
) {
    val innerSize by animateDpAsState(
        targetValue = if (isRecording) 28.dp else 56.dp,
        animationSpec = spring(),
        label = "shutterSize",
    )
    val innerCorner by animateDpAsState(
        targetValue = if (isRecording) 8.dp else 28.dp,
        animationSpec = spring(),
        label = "shutterCorner",
    )

    Box(
        modifier = Modifier
            .size(76.dp)
            .clip(CircleShape)
            .border(3.dp, TextPrimary, CircleShape)
            .background(SurfaceGlass)
            .clickable(onClick = onClick)
            .testTag("record_button"),
        contentAlignment = Alignment.Center,
    ) {
        Box(
            modifier = Modifier
                .size(innerSize)
                .clip(RoundedCornerShape(innerCorner))
                .background(CoralCrimson),
        )
    }
}

@Composable
private fun SavedVideoBanner(
    onPlay: () -> Unit,
    onShare: () -> Unit,
    onDismiss: () -> Unit,
) {
    Surface(
        color = Color(0xEE0D2220),
        shape = RoundedCornerShape(16.dp),
        border = BorderStroke(1.dp, MintCyan),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "Video saved to Movies!",
                    style = MaterialTheme.typography.labelMedium,
                    color = MintCyan,
                )
                Text(
                    text = "ভিডিও গ্যালারিতে সংরক্ষিত হয়েছে",
                    style = MaterialTheme.typography.labelSmall,
                    color = TextPrimary,
                )
            }
            Row(verticalAlignment = Alignment.CenterVertically) {
                TextButton(onClick = onPlay) {
                    Icon(Icons.Filled.PlayArrow, contentDescription = null, tint = MintCyan, modifier = Modifier.size(16.dp))
                    Spacer(Modifier.width(4.dp))
                    Text("Play • দেখুন", color = MintCyan, fontSize = 11.sp)
                }
                IconButton(onClick = onShare) {
                    Icon(Icons.Filled.Share, contentDescription = "Share • শেয়ার", tint = TextPrimary)
                }
                IconButton(onClick = onDismiss) {
                    Icon(Icons.Filled.Close, contentDescription = "Close • বন্ধ", tint = TextMuted)
                }
            }
        }
    }
}
