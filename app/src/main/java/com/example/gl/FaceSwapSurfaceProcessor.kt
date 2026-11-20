package com.example.gl

import android.graphics.Bitmap
import android.graphics.SurfaceTexture
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLExt
import android.opengl.EGLSurface
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.opengl.GLUtils
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.Surface
import androidx.camera.core.CameraEffect
import androidx.camera.core.SurfaceOutput
import androidx.camera.core.SurfaceProcessor
import androidx.camera.core.SurfaceRequest
import com.example.vision.FaceLandmarkerHelper
import com.example.vision.FaceMeshTriangulation
import com.example.vision.LiveFaceTrackingResult
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.ShortBuffer
import java.util.concurrent.Executor

class FaceSwapSurfaceProcessor : SurfaceProcessor {

    private val glThread = HandlerThread("FaceSwapGL").apply { start() }
    private val glHandler = Handler(glThread.looper)
    val glExecutor: Executor = Executor { runnable -> glHandler.post(runnable) }

    private var eglDisplay: EGLDisplay = EGL14.EGL_NO_DISPLAY
    private var eglContext: EGLContext = EGL14.EGL_NO_CONTEXT
    private var eglConfig: EGLConfig? = null
    private var dummySurface: EGLSurface = EGL14.EGL_NO_SURFACE

    private var oesTexId: Int = 0
    private var faceTexId: Int = 0
    private var hasFaceTexture: Boolean = false

    private var camProgram: Int = 0
    private var mouthProgram: Int = 0
    private var meshProgram: Int = 0

    private val outputTargets = mutableListOf<OutputTarget>()
    private val texMatrix = FloatArray(16)
    private val updatedMatrix = FloatArray(16)

    @Volatile
    private var latestTracking: LiveFaceTrackingResult? = null

    @Volatile
    var blendOpacity: Float = 0.92f

    @Volatile
    var userMirrored: Boolean = true

    @Volatile
    var isFrontCamera: Boolean = true

    private var triangleIndexBuffer: ShortBuffer? = null
    private var triangleIndexCount: Int = 0
    private var uvBuffer: FloatBuffer? = null
    private var alphaBuffer: FloatBuffer? = null
    private var baseAlphasArray: FloatArray? = null
    private var dynamicAlphasArray: FloatArray? = null
    private var lastValidTracking: LiveFaceTrackingResult? = null
    private var lastFaceUptimeMs: Long = 0L
    private var posBuffer: FloatBuffer? = null
    private var mouthPosBuffer: FloatBuffer? = null
    private var mouthEdgeBuffer: FloatBuffer? = null
    private var faceAvgRgb = floatArrayOf(0.5f, 0.45f, 0.42f)
    private val smoothedGain = floatArrayOf(1.0f, 1.0f, 1.0f)

    private val quadPosBuffer: FloatBuffer = createFloatBuffer(
        floatArrayOf(
            -1f, -1f,
            1f, -1f,
            -1f, 1f,
            1f, 1f,
        ),
    )

    private val quadUvBuffer: FloatBuffer = createFloatBuffer(
        floatArrayOf(
            0f, 0f,
            1f, 0f,
            0f, 1f,
            1f, 1f,
        ),
    )

    private var glInitialized = false

    private data class OutputTarget(
        val surfaceOutput: SurfaceOutput,
        val surface: Surface,
        val eglSurface: EGLSurface,
    )

    init {
        glHandler.post { initGl() }
    }

    fun updateLiveTracking(result: LiveFaceTrackingResult) {
        latestTracking = result
    }

    fun setCharacterFace(bitmap: Bitmap?, uvLandmarks: FloatArray?) {
        glHandler.post {
            if (!glInitialized) return@post
            if (bitmap == null || uvLandmarks == null || uvLandmarks.isEmpty()) {
                hasFaceTexture = false
                return@post
            }
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, faceTexId)
            GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, bitmap, 0)
            uvBuffer = createFloatBuffer(uvLandmarks)
            faceAvgRgb = FaceLandmarkerHelper.sampleSkinRgb(bitmap, uvLandmarks)
            smoothedGain[0] = 1.0f
            smoothedGain[1] = 1.0f
            smoothedGain[2] = 1.0f

            val vertexCount = uvLandmarks.size / 2
            val precomputedAlphas = FaceMeshTriangulation.vertexAlphas
            val alphas = FloatArray(vertexCount) { idx ->
                if (idx < precomputedAlphas.size) precomputedAlphas[idx] else 1f
            }
            baseAlphasArray = alphas
            dynamicAlphasArray = FloatArray(vertexCount)
            alphaBuffer = createFloatBuffer(alphas)
            hasFaceTexture = true
        }
    }

    override fun onInputSurface(request: SurfaceRequest) {
        glHandler.post {
            if (!glInitialized) initGl()
            val st = SurfaceTexture(oesTexId).apply {
                setDefaultBufferSize(request.resolution.width, request.resolution.height)
            }
            val surface = Surface(st)
            st.setOnFrameAvailableListener({ surfaceTexture ->
                renderFrame(surfaceTexture)
            }, glHandler)

            request.provideSurface(surface, glExecutor) {
                st.setOnFrameAvailableListener(null)
                surface.release()
                st.release()
            }
        }
    }

    override fun onOutputSurface(surfaceOutput: SurfaceOutput) {
        glHandler.post {
            if (!glInitialized) initGl()
            val surface = surfaceOutput.getSurface(glExecutor) {
                glHandler.post {
                    val iterator = outputTargets.iterator()
                    while (iterator.hasNext()) {
                        val item = iterator.next()
                        if (item.surfaceOutput === surfaceOutput) {
                            EGL14.eglDestroySurface(eglDisplay, item.eglSurface)
                            iterator.remove()
                        }
                    }
                    surfaceOutput.close()
                }
            }

            val eglSurface = EGL14.eglCreateWindowSurface(
                eglDisplay,
                eglConfig,
                surface,
                intArrayOf(EGL14.EGL_NONE),
                0,
            )
            if (eglSurface != null && eglSurface != EGL14.EGL_NO_SURFACE) {
                outputTargets.add(OutputTarget(surfaceOutput, surface, eglSurface))
            }
        }
    }

    private fun initGl() {
        if (glInitialized) return
        eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
        val version = IntArray(2)
        EGL14.eglInitialize(eglDisplay, version, 0, version, 1)

        val attribList = intArrayOf(
            EGL14.EGL_RED_SIZE, 8,
            EGL14.EGL_GREEN_SIZE, 8,
            EGL14.EGL_BLUE_SIZE, 8,
            EGL14.EGL_ALPHA_SIZE, 8,
            EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
            EGL_RECORDABLE_ANDROID, 1,
            EGL14.EGL_NONE,
        )
        val configs = arrayOfNulls<EGLConfig>(1)
        val numConfigs = IntArray(1)
        EGL14.eglChooseConfig(eglDisplay, attribList, 0, configs, 0, 1, numConfigs, 0)
        eglConfig = configs[0]

        eglContext = EGL14.eglCreateContext(
            eglDisplay,
            eglConfig,
            EGL14.EGL_NO_CONTEXT,
            intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 2, EGL14.EGL_NONE),
            0,
        )

        dummySurface = EGL14.eglCreatePbufferSurface(
            eglDisplay,
            eglConfig,
            intArrayOf(EGL14.EGL_WIDTH, 1, EGL14.EGL_HEIGHT, 1, EGL14.EGL_NONE),
            0,
        )
        EGL14.eglMakeCurrent(eglDisplay, dummySurface, dummySurface, eglContext)

        val textures = IntArray(2)
        GLES20.glGenTextures(2, textures, 0)
        oesTexId = textures[0]
        faceTexId = textures[1]

        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, oesTexId)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)

        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, faceTexId)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)

        camProgram = createProgram(CAM_VERTEX_SHADER, CAM_FRAGMENT_SHADER)
        mouthProgram = createProgram(MOUTH_VERTEX_SHADER, MOUTH_FRAGMENT_SHADER)
        meshProgram = createProgram(MESH_VERTEX_SHADER, MESH_FRAGMENT_SHADER)

        val indices = FaceMeshTriangulation.triangleIndices
        triangleIndexCount = indices.size
        triangleIndexBuffer = ByteBuffer.allocateDirect(indices.size * 2)
            .order(ByteOrder.nativeOrder())
            .asShortBuffer()
            .apply {
                put(indices)
                position(0)
            }

        glInitialized = true
    }

    private fun renderFrame(st: SurfaceTexture) {
        if (!glInitialized) return
        runCatching {
            st.updateTexImage()
            st.getTransformMatrix(texMatrix)
        }.onFailure {
            Log.e(TAG, "updateTexImage failed", it)
            return
        }

        val tracking = latestTracking
        val opacity = blendOpacity
        val mirrored = userMirrored
        val nowMs = android.os.SystemClock.uptimeMillis()

        var activeTracking: LiveFaceTrackingResult? = null
        var activeOpacity = opacity
        if (tracking != null && tracking.hasFace && tracking.landmarkCount > 0) {
            lastValidTracking = tracking
            lastFaceUptimeMs = nowMs
            activeTracking = tracking
        } else if (lastValidTracking != null) {
            val elapsed = nowMs - lastFaceUptimeMs
            if (elapsed <= 240L) {
                activeTracking = lastValidTracking
                activeOpacity = opacity * (1f - elapsed / 240f).coerceIn(0f, 1f)
            } else {
                lastValidTracking = null
            }
        }

        for (target in outputTargets) {
            if (!EGL14.eglMakeCurrent(eglDisplay, target.eglSurface, target.eglSurface, eglContext)) {
                continue
            }
            val size = target.surfaceOutput.size
            GLES20.glViewport(0, 0, size.width, size.height)
            GLES20.glClearColor(0.035f, 0.043f, 0.063f, 1f)
            GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)

            target.surfaceOutput.updateTransformMatrix(updatedMatrix, texMatrix)
            val cameraXMirrored = (target.surfaceOutput.targets == CameraEffect.PREVIEW) && isFrontCamera
            val flipCameraH = mirrored != cameraXMirrored

            drawCameraPass(updatedMatrix, flipCameraH)

            if (hasFaceTexture && activeTracking != null && activeOpacity > 0.01f) {
                GLES20.glEnable(GLES20.GL_BLEND)
                GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)

                if (activeTracking.jawOpen > 0.06f) {
                    drawMouthInterior(activeTracking.smoothedXY, activeTracking.jawOpen, mirrored, activeOpacity)
                }
                drawFaceMesh(
                    smoothedXY = activeTracking.smoothedXY,
                    landmarkCount = activeTracking.landmarkCount,
                    mirrored = mirrored,
                    opacity = activeOpacity,
                    liveSkinRgb = activeTracking.liveSkinRgb,
                )

                GLES20.glDisable(GLES20.GL_BLEND)
            }

            EGLExt.eglPresentationTimeANDROID(eglDisplay, target.eglSurface, st.timestamp)
            EGL14.eglSwapBuffers(eglDisplay, target.eglSurface)
        }

        EGL14.eglMakeCurrent(eglDisplay, dummySurface, dummySurface, eglContext)
    }

    private fun drawCameraPass(matrix: FloatArray, flipH: Boolean) {
        GLES20.glDisable(GLES20.GL_BLEND)
        GLES20.glUseProgram(camProgram)

        val posLoc = GLES20.glGetAttribLocation(camProgram, "aPos")
        val uvLoc = GLES20.glGetAttribLocation(camProgram, "aUv")
        val matLoc = GLES20.glGetUniformLocation(camProgram, "uTexMatrix")
        val flipLoc = GLES20.glGetUniformLocation(camProgram, "uFlipH")
        val samplerLoc = GLES20.glGetUniformLocation(camProgram, "uCamTex")

        GLES20.glEnableVertexAttribArray(posLoc)
        GLES20.glVertexAttribPointer(posLoc, 2, GLES20.GL_FLOAT, false, 0, quadPosBuffer)

        GLES20.glEnableVertexAttribArray(uvLoc)
        GLES20.glVertexAttribPointer(uvLoc, 2, GLES20.GL_FLOAT, false, 0, quadUvBuffer)

        GLES20.glUniformMatrix4fv(matLoc, 1, false, matrix, 0)
        GLES20.glUniform1f(flipLoc, if (flipH) 1f else 0f)

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, oesTexId)
        GLES20.glUniform1i(samplerLoc, 0)

        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
        GLES20.glDisableVertexAttribArray(posLoc)
        GLES20.glDisableVertexAttribArray(uvLoc)
    }

    private fun drawMouthInterior(
        smoothedXY: FloatArray,
        jawOpen: Float,
        mirrored: Boolean,
        opacity: Float,
    ) {
        val ring = FaceMeshTriangulation.innerLipRing
        var cx = 0f
        var cy = 0f
        for (idx in ring) {
            val nx = if (mirrored) 1f - smoothedXY[idx * 2] else smoothedXY[idx * 2]
            val ny = smoothedXY[idx * 2 + 1]
            cx += nx * 2f - 1f
            cy += 1f - ny * 2f
        }
        cx /= ring.size
        cy /= ring.size

        val fanVertexCount = ring.size + 2
        val posArray = FloatArray(fanVertexCount * 2)
        val edgeArray = FloatArray(fanVertexCount)

        posArray[0] = cx
        posArray[1] = cy
        edgeArray[0] = 0f

        for (i in 0..ring.size) {
            val idx = ring[i % ring.size]
            val nx = if (mirrored) 1f - smoothedXY[idx * 2] else smoothedXY[idx * 2]
            val ny = smoothedXY[idx * 2 + 1]
            posArray[(i + 1) * 2] = nx * 2f - 1f
            posArray[(i + 1) * 2 + 1] = 1f - ny * 2f
            edgeArray[i + 1] = 1f
        }

        mouthPosBuffer = updateFloatBuffer(mouthPosBuffer, posArray)
        mouthEdgeBuffer = updateFloatBuffer(mouthEdgeBuffer, edgeArray)

        GLES20.glUseProgram(mouthProgram)
        val posLoc = GLES20.glGetAttribLocation(mouthProgram, "aPos")
        val edgeLoc = GLES20.glGetAttribLocation(mouthProgram, "aEdge")
        val alphaLoc = GLES20.glGetUniformLocation(mouthProgram, "uAlpha")

        GLES20.glEnableVertexAttribArray(posLoc)
        GLES20.glVertexAttribPointer(posLoc, 2, GLES20.GL_FLOAT, false, 0, mouthPosBuffer)

        GLES20.glEnableVertexAttribArray(edgeLoc)
        GLES20.glVertexAttribPointer(edgeLoc, 1, GLES20.GL_FLOAT, false, 0, mouthEdgeBuffer)

        val mouthAlpha = (opacity * jawOpen * 4f).coerceIn(0f, 0.96f)
        GLES20.glUniform1f(alphaLoc, mouthAlpha)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_FAN, 0, fanVertexCount)

        GLES20.glDisableVertexAttribArray(posLoc)
        GLES20.glDisableVertexAttribArray(edgeLoc)
    }

    private fun drawFaceMesh(
        smoothedXY: FloatArray,
        landmarkCount: Int,
        mirrored: Boolean,
        opacity: Float,
        liveSkinRgb: FloatArray?,
    ) {
        val uvBuf = uvBuffer ?: return
        val alphaBuf = alphaBuffer ?: return
        val idxBuf = triangleIndexBuffer ?: return
        if (uvBuf.capacity() < landmarkCount * 2) return

        if (liveSkinRgb != null && liveSkinRgb.size >= 3) {
            for (c in 0..2) {
                val ratio = (liveSkinRgb[c] / faceAvgRgb[c].coerceAtLeast(0.08f)).coerceIn(0.55f, 1.55f)
                val blended = 1.0f + (ratio - 1.0f) * 0.72f
                smoothedGain[c] = smoothedGain[c] * 0.7f + blended * 0.3f
            }
        }

        val ndcPos = FloatArray(landmarkCount * 2)
        for (i in 0 until landmarkCount) {
            val nx = if (mirrored) 1f - smoothedXY[i * 2] else smoothedXY[i * 2]
            val ny = smoothedXY[i * 2 + 1]
            ndcPos[i * 2] = nx * 2f - 1f
            ndcPos[i * 2 + 1] = 1f - ny * 2f
        }
        posBuffer = updateFloatBuffer(posBuffer, ndcPos)

        // Side-turn occlusion fade so steep 45-80 deg yaw wraps smoothly behind the nose
        val base = baseAlphasArray
        if (base != null && base.size >= landmarkCount) {
            val outAlphas = dynamicAlphasArray?.takeIf { it.size >= landmarkCount }
                ?: FloatArray(landmarkCount).also { dynamicAlphasArray = it }
            val noseX = if (landmarkCount > 4) smoothedXY[4 * 2] else 0.5f
            val leftEdgeX = if (landmarkCount > 234) smoothedXY[234 * 2] else 0.3f
            val rightEdgeX = if (landmarkCount > 454) smoothedXY[454 * 2] else 0.7f
            val span = maxOf(0.04f, kotlin.math.abs(rightEdgeX - leftEdgeX))
            val midX = (leftEdgeX + rightEdgeX) * 0.5f
            val yaw = (((noseX - midX) / (span * 0.5f)) * 1.4f).coerceIn(-1f, 1f)
            val absYaw = kotlin.math.abs(yaw)
            if (absYaw < 0.35f) {
                System.arraycopy(base, 0, outAlphas, 0, landmarkCount)
            } else {
                val sideFadeStrength = ((absYaw - 0.35f) / 0.55f).coerceIn(0f, 1f)
                for (i in 0 until landmarkCount) {
                    var a = base[i]
                    val dxNorm = (smoothedXY[i * 2] - noseX) / (span * 0.5f)
                    val onHiddenSide = if (yaw < 0f) dxNorm > 0.12f else dxNorm < -0.12f
                    if (onHiddenSide) {
                        val edgeFrac = ((kotlin.math.abs(dxNorm) - 0.12f) / 0.55f).coerceIn(0f, 1f)
                        a *= 1f - sideFadeStrength * edgeFrac * 0.85f
                    }
                    outAlphas[i] = a
                }
            }
            alphaBuffer = updateFloatBuffer(alphaBuffer, outAlphas)
        }
        val activeAlphaBuf = alphaBuffer ?: alphaBuf

        GLES20.glUseProgram(meshProgram)
        val posLoc = GLES20.glGetAttribLocation(meshProgram, "aPos")
        val uvLoc = GLES20.glGetAttribLocation(meshProgram, "aUv")
        val alphaLoc = GLES20.glGetAttribLocation(meshProgram, "aAlpha")
        val texLoc = GLES20.glGetUniformLocation(meshProgram, "uFaceTex")
        val opacityLoc = GLES20.glGetUniformLocation(meshProgram, "uOpacity")
        val gainLoc = GLES20.glGetUniformLocation(meshProgram, "uColorGain")

        GLES20.glEnableVertexAttribArray(posLoc)
        GLES20.glVertexAttribPointer(posLoc, 2, GLES20.GL_FLOAT, false, 0, posBuffer)

        GLES20.glEnableVertexAttribArray(uvLoc)
        GLES20.glVertexAttribPointer(uvLoc, 2, GLES20.GL_FLOAT, false, 0, uvBuf)

        GLES20.glEnableVertexAttribArray(alphaLoc)
        GLES20.glVertexAttribPointer(alphaLoc, 1, GLES20.GL_FLOAT, false, 0, activeAlphaBuf)

        GLES20.glActiveTexture(GLES20.GL_TEXTURE1)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, faceTexId)
        GLES20.glUniform1i(texLoc, 1)
        GLES20.glUniform1f(opacityLoc, opacity)
        GLES20.glUniform3f(gainLoc, smoothedGain[0], smoothedGain[1], smoothedGain[2])

        idxBuf.position(0)
        GLES20.glDrawElements(GLES20.GL_TRIANGLES, triangleIndexCount, GLES20.GL_UNSIGNED_SHORT, idxBuf)

        GLES20.glDisableVertexAttribArray(posLoc)
        GLES20.glDisableVertexAttribArray(uvLoc)
        GLES20.glDisableVertexAttribArray(alphaLoc)
    }

    fun release() {
        glHandler.post {
            for (target in outputTargets) {
                EGL14.eglDestroySurface(eglDisplay, target.eglSurface)
            }
            outputTargets.clear()
            if (eglDisplay != EGL14.EGL_NO_DISPLAY) {
                EGL14.eglMakeCurrent(
                    eglDisplay,
                    EGL14.EGL_NO_SURFACE,
                    EGL14.EGL_NO_SURFACE,
                    EGL14.EGL_NO_CONTEXT,
                )
                if (dummySurface != EGL14.EGL_NO_SURFACE) {
                    EGL14.eglDestroySurface(eglDisplay, dummySurface)
                }
                if (eglContext != EGL14.EGL_NO_CONTEXT) {
                    EGL14.eglDestroyContext(eglDisplay, eglContext)
                }
                EGL14.eglTerminate(eglDisplay)
            }
            glInitialized = false
            glThread.quitSafely()
        }
    }

    private fun createFloatBuffer(data: FloatArray): FloatBuffer {
        return ByteBuffer.allocateDirect(data.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .apply {
                put(data)
                position(0)
            }
    }

    private fun updateFloatBuffer(existing: FloatBuffer?, data: FloatArray): FloatBuffer {
        val buf = if (existing != null && existing.capacity() >= data.size) {
            existing
        } else {
            ByteBuffer.allocateDirect(data.size * 4)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer()
        }
        buf.clear()
        buf.put(data)
        buf.position(0)
        return buf
    }

    private fun compileShader(type: Int, src: String): Int {
        val shader = GLES20.glCreateShader(type)
        GLES20.glShaderSource(shader, src)
        GLES20.glCompileShader(shader)
        return shader
    }

    private fun createProgram(vSrc: String, fSrc: String): Int {
        val prog = GLES20.glCreateProgram()
        GLES20.glAttachShader(prog, compileShader(GLES20.GL_VERTEX_SHADER, vSrc))
        GLES20.glAttachShader(prog, compileShader(GLES20.GL_FRAGMENT_SHADER, fSrc))
        GLES20.glLinkProgram(prog)
        return prog
    }

    companion object {
        private const val TAG = "FaceSwapSurfaceProc"
        private const val EGL_RECORDABLE_ANDROID = 0x3142

        private const val CAM_VERTEX_SHADER = """
            attribute vec2 aPos;
            attribute vec2 aUv;
            uniform mat4 uTexMatrix;
            uniform float uFlipH;
            varying vec2 vUv;
            void main() {
                float ux = mix(aUv.x, 1.0 - aUv.x, uFlipH);
                vUv = (uTexMatrix * vec4(ux, aUv.y, 0.0, 1.0)).xy;
                gl_Position = vec4(aPos, 0.0, 1.0);
            }
        """

        private const val CAM_FRAGMENT_SHADER = """
            #extension GL_OES_EGL_image_external : require
            precision mediump float;
            uniform samplerExternalOES uCamTex;
            varying vec2 vUv;
            void main() {
                gl_FragColor = texture2D(uCamTex, vUv);
            }
        """

        private const val MOUTH_VERTEX_SHADER = """
            attribute vec2 aPos;
            attribute float aEdge;
            varying float vEdge;
            void main() {
                vEdge = aEdge;
                gl_Position = vec4(aPos, 0.0, 1.0);
            }
        """

        private const val MOUTH_FRAGMENT_SHADER = """
            precision mediump float;
            uniform float uAlpha;
            varying float vEdge;
            void main() {
                vec3 deepCavity = vec3(0.055, 0.015, 0.025);
                vec3 lipShadow = vec3(0.14, 0.04, 0.055);
                vec3 col = mix(deepCavity, lipShadow, vEdge);
                gl_FragColor = vec4(col, uAlpha);
            }
        """

        private const val MESH_VERTEX_SHADER = """
            attribute vec2 aPos;
            attribute vec2 aUv;
            attribute float aAlpha;
            varying vec2 vUv;
            varying float vAlpha;
            void main() {
                vUv = aUv;
                vAlpha = aAlpha;
                gl_Position = vec4(aPos, 0.0, 1.0);
            }
        """

        private const val MESH_FRAGMENT_SHADER = """
            precision mediump float;
            uniform sampler2D uFaceTex;
            uniform float uOpacity;
            uniform vec3 uColorGain;
            varying vec2 vUv;
            varying float vAlpha;
            void main() {
                vec4 tex = texture2D(uFaceTex, vUv);
                vec3 matched = clamp(tex.rgb * uColorGain, 0.0, 1.0);
                gl_FragColor = vec4(matched, tex.a * vAlpha * uOpacity);
            }
        """
    }
}
