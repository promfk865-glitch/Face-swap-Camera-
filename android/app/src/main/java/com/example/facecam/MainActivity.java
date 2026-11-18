package com.example.facecam;

import android.Manifest;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Base64;
import android.webkit.JavascriptInterface;
import android.webkit.PermissionRequest;
import android.webkit.WebView;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.core.content.ContextCompat;
import com.getcapacitor.BridgeActivity;
import com.getcapacitor.BridgeWebChromeClient;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends BridgeActivity {

    private PermissionRequest pendingWebPermissionRequest;
    private ActivityResultLauncher<String[]> permissionLauncher;
    private boolean isRequestingPermissions = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        permissionLauncher = registerForActivityResult(
            new ActivityResultContracts.RequestMultiplePermissions(),
            grants -> {
                isRequestingPermissions = false;
                if (pendingWebPermissionRequest != null) {
                    PermissionRequest req = pendingWebPermissionRequest;
                    pendingWebPermissionRequest = null;
                    grantSatisfiedWebResources(req);
                }
            }
        );

        requestRequiredPermissions();

        WebView webView = getBridge().getWebView();
        webView.getSettings().setMediaPlaybackRequiresUserGesture(false);
        webView.addJavascriptInterface(new AndroidBridge(), "AndroidBridge");
        webView.setWebChromeClient(new BridgeWebChromeClient(getBridge()) {
            @Override
            public void onPermissionRequest(final PermissionRequest request) {
                runOnUiThread(() -> {
                    boolean needsVideo = false;
                    boolean needsAudio = false;
                    for (String res : request.getResources()) {
                        if (PermissionRequest.RESOURCE_VIDEO_CAPTURE.equals(res)) needsVideo = true;
                        if (PermissionRequest.RESOURCE_AUDIO_CAPTURE.equals(res)) needsAudio = true;
                    }

                    boolean videoOk = !needsVideo || hasCameraPermission();
                    boolean audioOk = !needsAudio || hasAudioPermission();

                    if (videoOk && audioOk) {
                        request.grant(request.getResources());
                    } else {
                        pendingWebPermissionRequest = request;
                        if (!isRequestingPermissions) {
                            requestRequiredPermissions();
                        }
                    }
                });
            }
        });
    }

    private void grantSatisfiedWebResources(PermissionRequest request) {
        List<String> granted = new ArrayList<>();
        for (String res : request.getResources()) {
            if (PermissionRequest.RESOURCE_VIDEO_CAPTURE.equals(res) && hasCameraPermission()) {
                granted.add(res);
            } else if (PermissionRequest.RESOURCE_AUDIO_CAPTURE.equals(res) && hasAudioPermission()) {
                granted.add(res);
            }
        }
        runOnUiThread(() -> {
            if (!granted.isEmpty()) {
                request.grant(granted.toArray(new String[0]));
            } else {
                request.deny();
            }
        });
    }

    private boolean hasCameraPermission() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED;
    }

    private boolean hasAudioPermission() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            == PackageManager.PERMISSION_GRANTED;
    }

    private void requestRequiredPermissions() {
        List<String> needed = new ArrayList<>();
        if (!hasCameraPermission()) {
            needed.add(Manifest.permission.CAMERA);
        }
        if (!hasAudioPermission()) {
            needed.add(Manifest.permission.RECORD_AUDIO);
        }
        if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P
            && ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED) {
            needed.add(Manifest.permission.WRITE_EXTERNAL_STORAGE);
        }
        if (!needed.isEmpty() && !isRequestingPermissions) {
            isRequestingPermissions = true;
            permissionLauncher.launch(needed.toArray(new String[0]));
        }
    }

    public class AndroidBridge {
        @JavascriptInterface
        public void requestCameraAndAudioPermissions() {
            runOnUiThread(() -> requestRequiredPermissions());
        }

        @JavascriptInterface
        public boolean saveVideoBase64(String base64Data, String fileName, String mimeType) {
            try {
                byte[] bytes = Base64.decode(base64Data, Base64.DEFAULT);
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    ContentValues values = new ContentValues();
                    values.put(MediaStore.MediaColumns.DISPLAY_NAME, fileName);
                    values.put(MediaStore.MediaColumns.MIME_TYPE, mimeType);
                    values.put(MediaStore.Video.Media.RELATIVE_PATH, Environment.DIRECTORY_MOVIES + "/FaceSwapCam");

                    ContentResolver resolver = getContentResolver();
                    Uri uri = resolver.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, values);
                    if (uri == null) return false;
                    try (OutputStream out = resolver.openOutputStream(uri)) {
                        if (out == null) return false;
                        out.write(bytes);
                    }
                    return true;
                } else {
                    File dir = new File(
                        Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES),
                        "FaceSwapCam"
                    );
                    if (!dir.exists() && !dir.mkdirs()) return false;
                    File file = new File(dir, fileName);
                    try (FileOutputStream fos = new FileOutputStream(file)) {
                        fos.write(bytes);
                    }
                    return true;
                }
            } catch (Exception e) {
                return false;
            }
        }
    }
}
