package com.example.ui

import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
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
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.AddPhotoAlternate
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.VideoLibrary
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import com.example.model.CharacterFace
import com.example.model.DetectedVideoPerson
import com.example.model.FaceSource
import com.example.model.VideoStudioState
import com.example.ui.theme.MintCyan
import com.example.ui.theme.MintCyanDim
import com.example.ui.theme.ObsidianBg
import com.example.ui.theme.SurfaceCard
import com.example.ui.theme.SurfaceGlass
import com.example.ui.theme.SurfaceStroke
import com.example.ui.theme.TextMuted
import com.example.ui.theme.TextPrimary

@Composable
fun VideoSwapScreen(
    studioState: VideoStudioState,
    faces: List<CharacterFace>,
    onBack: () -> Unit,
    onSelectPerson: (String) -> Unit,
    onAssignFaceToSelected: (String?) -> Unit,
    onUploadCharacterFace: (android.net.Uri) -> Unit,
    onTogglePlay: () -> Unit,
    onSeek: (Long) -> Unit,
    onExportVideo: () -> Unit,
) {
    BackHandler(enabled = !studioState.isExporting) {
        onBack()
    }

    val fallbackContentPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent(),
        onResult = { uri -> uri?.let(onUploadCharacterFace) },
    )
    val photoPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia(),
        onResult = { uri -> uri?.let(onUploadCharacterFace) },
    )

    val selectedPerson = studioState.detectedPeople.firstOrNull { it.id == studioState.selectedPersonId }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(ObsidianBg)
            .testTag("video_swap_studio"),
    ) {
        // Header
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .background(SurfaceCard)
                .windowInsetsPadding(WindowInsets.safeDrawing)
                .padding(horizontal = 12.dp, vertical = 10.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Surface(
                onClick = onBack,
                enabled = !studioState.isExporting,
                color = SurfaceGlass,
                shape = RoundedCornerShape(12.dp),
                border = BorderStroke(1.dp, SurfaceStroke),
            ) {
                Row(
                    modifier = Modifier.padding(horizontal = 10.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                        contentDescription = "Back • ফিরে যান",
                        tint = TextPrimary,
                        modifier = Modifier.size(16.dp),
                    )
                    Text(
                        text = "Camera • ক্যামেরা",
                        style = MaterialTheme.typography.labelMedium,
                        color = TextPrimary,
                    )
                }
            }

            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                Icon(
                    imageVector = Icons.Filled.VideoLibrary,
                    contentDescription = null,
                    tint = MintCyan,
                    modifier = Modifier.size(18.dp),
                )
                Text(
                    text = "Video Studio",
                    style = MaterialTheme.typography.labelLarge,
                    color = TextPrimary,
                )
            }

            Button(
                onClick = onExportVideo,
                enabled = !studioState.isScanning && !studioState.isExporting && studioState.detectedPeople.isNotEmpty(),
                colors = ButtonDefaults.buttonColors(
                    containerColor = MintCyan,
                    contentColor = ObsidianBg,
                ),
                shape = RoundedCornerShape(12.dp),
                contentPadding = PaddingValues(horizontal = 12.dp, vertical = 8.dp),
            ) {
                Icon(
                    imageVector = Icons.Filled.Download,
                    contentDescription = null,
                    modifier = Modifier.size(16.dp),
                )
                Spacer(modifier = Modifier.width(4.dp))
                Text(
                    text = if (studioState.isExporting) "${studioState.exportProgress}%" else "Save • সেভ",
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.Bold,
                )
            }
        }

        // Video Frame Preview Stage (Natural aspect, uncropped FIT)
        Box(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .background(Color(0xFF05070B)),
            contentAlignment = Alignment.Center,
        ) {
            val bmp = studioState.previewBitmap
            if (bmp != null && !bmp.isRecycled) {
                Image(
                    bitmap = bmp.asImageBitmap(),
                    contentDescription = "Video preview frame",
                    contentScale = ContentScale.Fit,
                    modifier = Modifier.fillMaxSize(),
                )
            }

            if (studioState.isScanning || studioState.isExporting) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(ObsidianBg.copy(alpha = 0.80f)),
                    contentAlignment = Alignment.Center,
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(10.dp),
                    ) {
                        CircularProgressIndicator(color = MintCyan)
                        Text(
                            text = if (studioState.isScanning) {
                                "Scanning faces (${studioState.scanProgress}%) • মুখ খোঁজা হচ্ছে (${studioState.scanProgress}%)"
                            } else {
                                "Saving swapped video (${studioState.exportProgress}%) • সেভ হচ্ছে (${studioState.exportProgress}%)"
                            },
                            style = MaterialTheme.typography.labelLarge,
                            color = TextPrimary,
                        )
                    }
                }
            }
        }

        // Controls + Detected Faces + Swap Target Deck
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .background(SurfaceCard)
                .navigationBarsPadding()
                .padding(horizontal = 12.dp, vertical = 10.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            // Timeline Scrubber
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                Surface(
                    onClick = onTogglePlay,
                    enabled = !studioState.isScanning && !studioState.isExporting,
                    color = SurfaceGlass,
                    shape = RoundedCornerShape(12.dp),
                    border = BorderStroke(1.dp, SurfaceStroke),
                    modifier = Modifier.size(42.dp),
                ) {
                    Box(contentAlignment = Alignment.Center) {
                        Icon(
                            imageVector = if (studioState.isPlayingPreview) Icons.Filled.Pause else Icons.Filled.PlayArrow,
                            contentDescription = "Play/Pause",
                            tint = MintCyan,
                        )
                    }
                }

                Slider(
                    value = studioState.previewPositionMs.toFloat(),
                    onValueChange = { onSeek(it.toLong()) },
                    valueRange = 0f..studioState.durationMs.coerceAtLeast(100L).toFloat(),
                    colors = SliderDefaults.colors(
                        thumbColor = MintCyan,
                        activeTrackColor = MintCyan,
                    ),
                    modifier = Modifier.weight(1f),
                )

                Text(
                    text = "%.1fs".format(studioState.previewPositionMs / 1000f),
                    style = MaterialTheme.typography.labelSmall,
                    color = TextMuted,
                )
            }

            if (studioState.detectedPeople.isNotEmpty()) {
                Text(
                    text = "1. Select person in video • ভিডিওর মুখ বেছে নিন (${studioState.detectedPeople.size}):",
                    style = MaterialTheme.typography.labelSmall,
                    color = TextMuted,
                )
                LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    items(studioState.detectedPeople, key = { it.id }) { person ->
                        DetectedPersonChip(
                            person = person,
                            assignedFaceName = faces.firstOrNull { it.id == person.assignedFaceId }?.nameEn,
                            isSelected = person.id == studioState.selectedPersonId,
                            onClick = { onSelectPerson(person.id) },
                        )
                    }
                }
            }

            if (selectedPerson != null) {
                Text(
                    text = "2. Replace ${selectedPerson.labelEn} (${selectedPerson.labelBn}) • নতুন মুখ দিন:",
                    style = MaterialTheme.typography.labelSmall,
                    color = TextMuted,
                )
                LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    item(key = "orig") {
                        Surface(
                            onClick = { onAssignFaceToSelected(null) },
                            color = SurfaceGlass,
                            shape = RoundedCornerShape(14.dp),
                            border = BorderStroke(
                                if (selectedPerson.assignedFaceId == null) 2.dp else 1.dp,
                                if (selectedPerson.assignedFaceId == null) MintCyan else SurfaceStroke,
                            ),
                            modifier = Modifier.size(width = 64.dp, height = 74.dp),
                        ) {
                            Column(
                                modifier = Modifier.fillMaxSize(),
                                horizontalAlignment = Alignment.CenterHorizontally,
                                verticalArrangement = Arrangement.Center,
                            ) {
                                Icon(Icons.Filled.CameraAlt, contentDescription = null, tint = MintCyan, modifier = Modifier.size(18.dp))
                                Text("Original", fontSize = 10.sp, color = TextPrimary)
                                Text("আসল", fontSize = 9.sp, color = TextMuted)
                            }
                        }
                    }

                    items(faces, key = { it.id }) { face ->
                        val active = selectedPerson.assignedFaceId == face.id
                        Surface(
                            onClick = { onAssignFaceToSelected(face.id) },
                            color = if (active) MintCyanDim else SurfaceGlass,
                            shape = RoundedCornerShape(14.dp),
                            border = BorderStroke(if (active) 2.dp else 1.dp, if (active) MintCyan else SurfaceStroke),
                            modifier = Modifier.size(width = 66.dp, height = 74.dp),
                        ) {
                            Box(modifier = Modifier.fillMaxSize().padding(4.dp)) {
                                Column(
                                    modifier = Modifier.fillMaxSize(),
                                    horizontalAlignment = Alignment.CenterHorizontally,
                                    verticalArrangement = Arrangement.SpaceBetween,
                                ) {
                                    when (val src = face.source) {
                                        is FaceSource.BundledDrawable -> Image(
                                            painter = painterResource(id = src.resId),
                                            contentDescription = face.nameEn,
                                            contentScale = ContentScale.Crop,
                                            modifier = Modifier.size(40.dp).clip(RoundedCornerShape(10.dp)),
                                        )
                                        is FaceSource.UploadedUri -> AsyncImage(
                                            model = src.uri,
                                            contentDescription = face.nameEn,
                                            contentScale = ContentScale.Crop,
                                            modifier = Modifier.size(40.dp).clip(RoundedCornerShape(10.dp)),
                                        )
                                    }
                                    Text(face.nameEn, fontSize = 9.sp, color = TextPrimary, maxLines = 1, overflow = TextOverflow.Ellipsis)
                                    Text(face.nameBn, fontSize = 8.sp, color = TextMuted, maxLines = 1, overflow = TextOverflow.Ellipsis)
                                }
                                if (active) {
                                    Box(
                                        modifier = Modifier
                                            .align(Alignment.TopEnd)
                                            .size(16.dp)
                                            .clip(CircleShape)
                                            .background(MintCyan),
                                        contentAlignment = Alignment.Center,
                                    ) {
                                        Icon(Icons.Filled.Check, contentDescription = null, tint = ObsidianBg, modifier = Modifier.size(11.dp))
                                    }
                                }
                            }
                        }
                    }

                    item(key = "upload_char") {
                        Surface(
                            onClick = {
                                runCatching {
                                    photoPickerLauncher.launch(
                                        PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly),
                                    )
                                }.onFailure {
                                    fallbackContentPicker.launch("image/*")
                                }
                            },
                            color = SurfaceGlass,
                            shape = RoundedCornerShape(14.dp),
                            border = BorderStroke(1.dp, MintCyan.copy(alpha = 0.6f)),
                            modifier = Modifier.size(width = 68.dp, height = 74.dp),
                        ) {
                            Column(
                                modifier = Modifier.fillMaxSize(),
                                horizontalAlignment = Alignment.CenterHorizontally,
                                verticalArrangement = Arrangement.Center,
                            ) {
                                Icon(Icons.Filled.AddPhotoAlternate, contentDescription = null, tint = MintCyan, modifier = Modifier.size(18.dp))
                                Text("Upload", fontSize = 10.sp, color = MintCyan)
                                Text("আপলোড", fontSize = 9.sp, color = TextMuted)
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun DetectedPersonChip(
    person: DetectedVideoPerson,
    assignedFaceName: String?,
    isSelected: Boolean,
    onClick: () -> Unit,
) {
    Surface(
        onClick = onClick,
        color = if (isSelected) MintCyanDim else SurfaceGlass,
        shape = RoundedCornerShape(14.dp),
        border = BorderStroke(if (isSelected) 2.dp else 1.dp, if (isSelected) MintCyan else SurfaceStroke),
        modifier = Modifier
            .clip(RoundedCornerShape(14.dp))
            .clickable(onClick = onClick),
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Image(
                bitmap = person.thumbnail.asImageBitmap(),
                contentDescription = person.labelEn,
                contentScale = ContentScale.Crop,
                modifier = Modifier
                    .size(38.dp)
                    .clip(RoundedCornerShape(10.dp))
                    .border(1.dp, SurfaceStroke, RoundedCornerShape(10.dp)),
            )
            Column {
                Text(person.labelEn, style = MaterialTheme.typography.labelMedium, color = TextPrimary)
                Text(person.labelBn, style = MaterialTheme.typography.labelSmall, fontSize = 10.sp, color = TextMuted)
                Text(
                    text = assignedFaceName?.let { "→ $it" } ?: "Original • আসল",
                    style = MaterialTheme.typography.labelSmall,
                    fontSize = 10.sp,
                    color = MintCyan,
                )
            }
        }
    }
}
