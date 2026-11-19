import { INNER_LIP_RING, SKIN_SAMPLE_INDICES } from './faceMeshTriangulation';

const CAM_VERTEX = `
  attribute vec2 aPos;
  attribute vec2 aUv;
  varying vec2 vUv;
  void main() {
    vUv = aUv;
    gl_Position = vec4(aPos, 0.0, 1.0);
  }
`;

const CAM_FRAGMENT = `
  precision mediump float;
  uniform sampler2D uCamTex;
  varying vec2 vUv;
  void main() {
    gl_FragColor = texture2D(uCamTex, vUv);
  }
`;

const MOUTH_VERTEX = `
  attribute vec2 aPos;
  attribute float aEdge;
  varying float vEdge;
  void main() {
    vEdge = aEdge;
    gl_Position = vec4(aPos, 0.0, 1.0);
  }
`;

const MOUTH_FRAGMENT = `
  precision mediump float;
  uniform float uAlpha;
  varying float vEdge;
  void main() {
    vec3 deepCavity = vec3(0.055, 0.015, 0.025);
    vec3 lipShadow = vec3(0.14, 0.04, 0.055);
    vec3 col = mix(deepCavity, lipShadow, vEdge);
    gl_FragColor = vec4(col, uAlpha);
  }
`;

const MESH_VERTEX = `
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
`;

const MESH_FRAGMENT = `
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
`;

export interface FaceSwapDrawTarget {
  personKey: string;
  faceKey: string;
  faceImage: HTMLImageElement;
  srcLandmarks: Array<{ x: number; y: number }>;
  liveLandmarks: Array<{ x: number; y: number }>;
  jawOpen: number;
}

interface CachedFaceGl {
  tex: WebGLTexture;
  uvBuf: WebGLBuffer;
  alphaBuf: WebGLBuffer;
  avgRgb: [number, number, number];
}

function compileShader(gl: WebGLRenderingContext, type: number, src: string): WebGLShader {
  const s = gl.createShader(type)!;
  gl.shaderSource(s, src);
  gl.compileShader(s);
  return s;
}

function createProgram(gl: WebGLRenderingContext, vSrc: string, fSrc: string): WebGLProgram {
  const prog = gl.createProgram()!;
  gl.attachShader(prog, compileShader(gl, gl.VERTEX_SHADER, vSrc));
  gl.attachShader(prog, compileShader(gl, gl.FRAGMENT_SHADER, fSrc));
  gl.linkProgram(prog);
  return prog;
}

function getSourceDimensions(source: TexImageSource): { w: number; h: number } {
  if (source instanceof HTMLVideoElement) {
    return { w: source.videoWidth, h: source.videoHeight };
  }
  if (source instanceof HTMLCanvasElement) {
    return { w: source.width, h: source.height };
  }
  if (source instanceof HTMLImageElement) {
    return { w: source.naturalWidth || source.width, h: source.naturalHeight || source.height };
  }
  return { w: (source as ImageBitmap).width ?? 0, h: (source as ImageBitmap).height ?? 0 };
}

export class FaceWarpGlRenderer {
  private gl: WebGLRenderingContext;
  private camProg: WebGLProgram;
  private mouthProg: WebGLProgram;
  private meshProg: WebGLProgram;
  private camTex: WebGLTexture;
  private activeSingleFace: CachedFaceGl | null = null;
  private faceCache = new Map<string, CachedFaceGl>();

  private quadPosBuf: WebGLBuffer;
  private quadUvBuf: WebGLBuffer;
  private meshPosBuf: WebGLBuffer;
  private meshIdxBuf: WebGLBuffer;
  private mouthPosBuf: WebGLBuffer;
  private mouthEdgeBuf: WebGLBuffer;
  private triangleCount = 0;
  private defaultVertexAlphas: Float32Array | null = null;

  private sampleCanvas: HTMLCanvasElement;
  private sampleCtx: CanvasRenderingContext2D | null;
  private smoothedGainByPerson = new Map<string, [number, number, number]>();
  private lastGainSampleByPerson = new Map<string, number>();

  constructor(canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl', {
      alpha: false,
      antialias: true,
      preserveDrawingBuffer: true,
    });
    if (!gl) throw new Error('WebGL unavailable');
    this.gl = gl;

    this.sampleCanvas = document.createElement('canvas');
    this.sampleCanvas.width = 64;
    this.sampleCanvas.height = 64;
    this.sampleCtx = this.sampleCanvas.getContext('2d', { willReadFrequently: true });

    this.camProg = createProgram(gl, CAM_VERTEX, CAM_FRAGMENT);
    this.mouthProg = createProgram(gl, MOUTH_VERTEX, MOUTH_FRAGMENT);
    this.meshProg = createProgram(gl, MESH_VERTEX, MESH_FRAGMENT);

    this.camTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.camTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    this.quadPosBuf = gl.createBuffer()!;
    this.quadUvBuf = gl.createBuffer()!;
    this.meshPosBuf = gl.createBuffer()!;
    this.meshIdxBuf = gl.createBuffer()!;
    this.mouthPosBuf = gl.createBuffer()!;
    this.mouthEdgeBuf = gl.createBuffer()!;
  }

  setTriangulation(indices: Uint16Array, vertexAlphas: Float32Array) {
    const gl = this.gl;
    this.triangleCount = indices.length;
    this.defaultVertexAlphas = vertexAlphas;
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.meshIdxBuf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
  }

  private getOrCreateFaceGl(
    faceKey: string,
    image: HTMLImageElement,
    srcLandmarks: Array<{ x: number; y: number }>,
  ): CachedFaceGl {
    const existing = this.faceCache.get(faceKey);
    if (existing) return existing;

    const gl = this.gl;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);

    const uvs = new Float32Array(srcLandmarks.length * 2);
    const alphas = new Float32Array(srcLandmarks.length);
    for (let i = 0; i < srcLandmarks.length; i++) {
      uvs[i * 2] = srcLandmarks[i].x;
      uvs[i * 2 + 1] = srcLandmarks[i].y;
      alphas[i] = this.defaultVertexAlphas?.[i] ?? 1.0;
    }

    const uvBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, uvBuf);
    gl.bufferData(gl.ARRAY_BUFFER, uvs, gl.STATIC_DRAW);

    const alphaBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, alphaBuf);
    gl.bufferData(gl.ARRAY_BUFFER, alphas, gl.STATIC_DRAW);

    const cached: CachedFaceGl = {
      tex,
      uvBuf,
      alphaBuf,
      avgRgb: this.sampleSkinRgb(image, srcLandmarks),
    };
    this.faceCache.set(faceKey, cached);
    return cached;
  }

  setCharacterFace(image: HTMLImageElement, srcLandmarks: Array<{ x: number; y: number }>, faceKey = image.src) {
    this.activeSingleFace = this.getOrCreateFaceGl(faceKey, image, srcLandmarks);
    this.smoothedGainByPerson.delete('live');
    this.lastGainSampleByPerson.delete('live');
  }

  clearCharacterFace() {
    this.activeSingleFace = null;
  }

  private sampleSkinRgb(
    source: CanvasImageSource,
    landmarks: Array<{ x: number; y: number }>,
  ): [number, number, number] {
    const ctx = this.sampleCtx;
    if (!ctx) return [0.5, 0.45, 0.42];
    try {
      ctx.drawImage(source, 0, 0, 64, 64);
      const data = ctx.getImageData(0, 0, 64, 64).data;
      let r = 0, g = 0, b = 0, count = 0;
      for (const idx of SKIN_SAMPLE_INDICES) {
        const lm = landmarks[idx];
        if (!lm) continue;
        const px = Math.min(63, Math.max(0, Math.round(lm.x * 63)));
        const py = Math.min(63, Math.max(0, Math.round(lm.y * 63)));
        const off = (py * 64 + px) * 4;
        r += data[off] / 255;
        g += data[off + 1] / 255;
        b += data[off + 2] / 255;
        count++;
      }
      if (count === 0) return [0.5, 0.45, 0.42];
      return [
        Math.max(0.08, r / count),
        Math.max(0.08, g / count),
        Math.max(0.08, b / count),
      ];
    } catch {
      return [0.5, 0.45, 0.42];
    }
  }

  renderFrame(
    source: TexImageSource,
    width: number,
    height: number,
    mirrored: boolean,
    landmarks: Array<{ x: number; y: number }> | null,
    jawOpen: number,
    opacity: number,
  ) {
    const gl = this.gl;
    gl.viewport(0, 0, width, height);
    gl.clearColor(0.035, 0.043, 0.063, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    const { w: srcW, h: srcH } = getSourceDimensions(source);
    if (srcW <= 0 || srcH <= 0) return;
    if (source instanceof HTMLVideoElement && source.readyState < 2) return;

    const { ndcScaleX, ndcScaleY } = this.drawBackgroundFrame(source, srcW, srcH, width, height, mirrored);
    const single = this.activeSingleFace;
    if (!single || !landmarks || this.triangleCount === 0) return;

    this.drawWarpedFacePass(
      source,
      'live',
      single,
      landmarks,
      jawOpen,
      opacity,
      mirrored,
      ndcScaleX,
      ndcScaleY,
    );
  }

  renderMultiFaceFrame(
    source: TexImageSource,
    width: number,
    height: number,
    mirrored: boolean,
    targets: FaceSwapDrawTarget[],
    opacity: number,
  ) {
    const gl = this.gl;
    gl.viewport(0, 0, width, height);
    gl.clearColor(0.035, 0.043, 0.063, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    const { w: srcW, h: srcH } = getSourceDimensions(source);
    if (srcW <= 0 || srcH <= 0) return;
    if (source instanceof HTMLVideoElement && source.readyState < 2) return;

    const { ndcScaleX, ndcScaleY } = this.drawBackgroundFrame(source, srcW, srcH, width, height, mirrored);
    if (this.triangleCount === 0 || targets.length === 0) return;

    for (const target of targets) {
      const faceGl = this.getOrCreateFaceGl(target.faceKey, target.faceImage, target.srcLandmarks);
      this.drawWarpedFacePass(
        source,
        target.personKey,
        faceGl,
        target.liveLandmarks,
        target.jawOpen,
        opacity,
        mirrored,
        ndcScaleX,
        ndcScaleY,
      );
    }
  }

  private drawBackgroundFrame(
    source: TexImageSource,
    srcW: number,
    srcH: number,
    viewW: number,
    viewH: number,
    mirrored: boolean,
  ): { ndcScaleX: number; ndcScaleY: number } {
    const gl = this.gl;
    // Always preserve full natural camera/video FOV via aspect-fit (zero zoom crop)
    const videoAspect = srcW / srcH;
    const viewAspect = viewW / viewH;
    let ndcScaleX = 1.0;
    let ndcScaleY = 1.0;
    if (Math.abs(videoAspect - viewAspect) > 0.02) {
      if (videoAspect > viewAspect) {
        ndcScaleY = viewAspect / videoAspect;
      } else {
        ndcScaleX = videoAspect / viewAspect;
      }
    }

    const leftU = mirrored ? 1 : 0;
    const rightU = mirrored ? 0 : 1;

    gl.bindTexture(gl.TEXTURE_2D, this.camTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);

    gl.disable(gl.BLEND);
    gl.useProgram(this.camProg);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadPosBuf);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([
        -ndcScaleX, -ndcScaleY,
        ndcScaleX, -ndcScaleY,
        -ndcScaleX, ndcScaleY,
        ndcScaleX, ndcScaleY,
      ]),
      gl.DYNAMIC_DRAW,
    );
    const camPosLoc = gl.getAttribLocation(this.camProg, 'aPos');
    gl.enableVertexAttribArray(camPosLoc);
    gl.vertexAttribPointer(camPosLoc, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadUvBuf);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([leftU, 1, rightU, 1, leftU, 0, rightU, 0]),
      gl.DYNAMIC_DRAW,
    );
    const camUvLoc = gl.getAttribLocation(this.camProg, 'aUv');
    gl.enableVertexAttribArray(camUvLoc);
    gl.vertexAttribPointer(camUvLoc, 2, gl.FLOAT, false, 0, 0);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.camTex);
    gl.uniform1i(gl.getUniformLocation(this.camProg, 'uCamTex'), 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.disableVertexAttribArray(camPosLoc);
    gl.disableVertexAttribArray(camUvLoc);

    return { ndcScaleX, ndcScaleY };
  }

  private drawWarpedFacePass(
    source: TexImageSource,
    personKey: string,
    faceGl: CachedFaceGl,
    landmarks: Array<{ x: number; y: number }>,
    jawOpen: number,
    opacity: number,
    mirrored: boolean,
    ndcScaleX: number,
    ndcScaleY: number,
  ) {
    const gl = this.gl;
    const now = performance.now();
    const lastSample = this.lastGainSampleByPerson.get(personKey) ?? 0;
    const smoothedGain = this.smoothedGainByPerson.get(personKey) ?? [1, 1, 1];

    if (now - lastSample > 160 && (source instanceof HTMLVideoElement || source instanceof HTMLCanvasElement)) {
      this.lastGainSampleByPerson.set(personKey, now);
      const liveRgb = this.sampleSkinRgb(source, landmarks);
      for (let c = 0; c < 3; c++) {
        const ratio = Math.min(1.55, Math.max(0.55, liveRgb[c] / faceGl.avgRgb[c]));
        const blended = 1.0 + (ratio - 1.0) * 0.72;
        smoothedGain[c] = smoothedGain[c] * 0.65 + blended * 0.35;
      }
      this.smoothedGainByPerson.set(personKey, smoothedGain);
    }

    const toNdcX = (nx: number) => ((mirrored ? 1.0 - nx : nx) * 2.0 - 1.0) * ndcScaleX;
    const toNdcY = (ny: number) => (1.0 - ny * 2.0) * ndcScaleY;

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    if (jawOpen > 0.06) {
      let cx = 0;
      let cy = 0;
      for (const idx of INNER_LIP_RING) {
        cx += toNdcX(landmarks[idx].x);
        cy += toNdcY(landmarks[idx].y);
      }
      cx /= INNER_LIP_RING.length;
      cy /= INNER_LIP_RING.length;

      const fanPos = new Float32Array((INNER_LIP_RING.length + 2) * 2);
      const fanEdge = new Float32Array(INNER_LIP_RING.length + 2);
      fanPos[0] = cx;
      fanPos[1] = cy;
      fanEdge[0] = 0.0;
      for (let i = 0; i <= INNER_LIP_RING.length; i++) {
        const lm = landmarks[INNER_LIP_RING[i % INNER_LIP_RING.length]];
        fanPos[(i + 1) * 2] = toNdcX(lm.x);
        fanPos[(i + 1) * 2 + 1] = toNdcY(lm.y);
        fanEdge[i + 1] = 1.0;
      }

      gl.useProgram(this.mouthProg);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.mouthPosBuf);
      gl.bufferData(gl.ARRAY_BUFFER, fanPos, gl.DYNAMIC_DRAW);
      const mPosLoc = gl.getAttribLocation(this.mouthProg, 'aPos');
      gl.enableVertexAttribArray(mPosLoc);
      gl.vertexAttribPointer(mPosLoc, 2, gl.FLOAT, false, 0, 0);

      gl.bindBuffer(gl.ARRAY_BUFFER, this.mouthEdgeBuf);
      gl.bufferData(gl.ARRAY_BUFFER, fanEdge, gl.DYNAMIC_DRAW);
      const mEdgeLoc = gl.getAttribLocation(this.mouthProg, 'aEdge');
      gl.enableVertexAttribArray(mEdgeLoc);
      gl.vertexAttribPointer(mEdgeLoc, 1, gl.FLOAT, false, 0, 0);

      gl.uniform1f(
        gl.getUniformLocation(this.mouthProg, 'uAlpha'),
        Math.min(0.95, opacity * (jawOpen * 4.0)),
      );
      gl.drawArrays(gl.TRIANGLE_FAN, 0, INNER_LIP_RING.length + 2);
      gl.disableVertexAttribArray(mPosLoc);
      gl.disableVertexAttribArray(mEdgeLoc);
    }

    const positions = new Float32Array(landmarks.length * 2);
    for (let i = 0; i < landmarks.length; i++) {
      positions[i * 2] = toNdcX(landmarks[i].x);
      positions[i * 2 + 1] = toNdcY(landmarks[i].y);
    }

    gl.useProgram(this.meshProg);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.meshPosBuf);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.DYNAMIC_DRAW);
    const posLoc = gl.getAttribLocation(this.meshProg, 'aPos');
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, faceGl.uvBuf);
    const uvLoc = gl.getAttribLocation(this.meshProg, 'aUv');
    gl.enableVertexAttribArray(uvLoc);
    gl.vertexAttribPointer(uvLoc, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, faceGl.alphaBuf);
    const alphaLoc = gl.getAttribLocation(this.meshProg, 'aAlpha');
    gl.enableVertexAttribArray(alphaLoc);
    gl.vertexAttribPointer(alphaLoc, 1, gl.FLOAT, false, 0, 0);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, faceGl.tex);
    gl.uniform1i(gl.getUniformLocation(this.meshProg, 'uFaceTex'), 0);
    gl.uniform1f(gl.getUniformLocation(this.meshProg, 'uOpacity'), opacity);
    gl.uniform3f(
      gl.getUniformLocation(this.meshProg, 'uColorGain'),
      smoothedGain[0],
      smoothedGain[1],
      smoothedGain[2],
    );

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.meshIdxBuf);
    gl.drawElements(gl.TRIANGLES, this.triangleCount, gl.UNSIGNED_SHORT, 0);
    gl.disableVertexAttribArray(posLoc);
    gl.disableVertexAttribArray(uvLoc);
    gl.disableVertexAttribArray(alphaLoc);
  }
}
