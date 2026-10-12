/**
 * Wrapper MediaPipe Tasks Vision — dipakai bersama oleh app.js dan capture.js.
 */

const MP_VERSION = "0.10.14";
const MP_CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

// ?delegate=cpu — paksa CPU (untuk perangkat dengan GPU/WebGL bermasalah).
const PREFERRED_DELEGATE =
  new URLSearchParams(location.search).get("delegate") === "cpu" ? "CPU" : "GPU";

/**
 * Buat instance HandLandmarker. Mencoba GPU (WebGL) dulu, lalu fallback ke CPU.
 * @returns {Promise<import('@mediapipe/tasks-vision').HandLandmarker>}
 */
export async function createHandLandmarker() {
  const vision = await import(/* @vite-ignore */ MP_CDN);
  const fileset = await vision.FilesetResolver.forVisionTasks(`${MP_CDN}/wasm`);

  const base = {
    baseOptions: { modelAssetPath: MODEL_URL },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  };

  try {
    return await vision.HandLandmarker.createFromOptions(fileset, {
      ...base,
      baseOptions: { ...base.baseOptions, delegate: PREFERRED_DELEGATE },
    });
  } catch (err) {
    if (PREFERRED_DELEGATE === "CPU") throw err;
    console.warn(`${PREFERRED_DELEGATE} delegate gagal, fallback ke CPU:`, err);
    return await vision.HandLandmarker.createFromOptions(fileset, {
      ...base,
      baseOptions: { ...base.baseOptions, delegate: "CPU" },
    });
  }
}

/**
 * Minta akses kamera dan pasang ke elemen <video>.
 * @param {HTMLVideoElement} videoEl
 */
export async function startCamera(videoEl) {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
    audio: false,
  });
  videoEl.srcObject = stream;
  await new Promise((resolve) => {
    videoEl.onloadedmetadata = () => {
      videoEl.play();
      resolve();
    };
  });
  return stream;
}

/** Hentikan seluruh track kamera. */
export function stopCamera(videoEl) {
  const stream = videoEl.srcObject;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  videoEl.srcObject = null;
}

/**
 * Gambar kerangka tangan ke canvas (koordinat landmark sudah ternormalisasi).
 * Koneksi antar landmark mengikuti topologi MediaPipe Hands.
 */
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

export function drawHand(ctx, landmarks, width, height, color = "#4f8cff") {
  ctx.clearRect(0, 0, width, height);
  if (!landmarks) return;

  ctx.lineWidth = 3;
  ctx.strokeStyle = color;
  ctx.beginPath();
  for (const [a, b] of HAND_CONNECTIONS) {
    ctx.moveTo(landmarks[a].x * width, landmarks[a].y * height);
    ctx.lineTo(landmarks[b].x * width, landmarks[b].y * height);
  }
  ctx.stroke();

  ctx.fillStyle = "#ffffff";
  for (const lm of landmarks) {
    ctx.beginPath();
    ctx.arc(lm.x * width, lm.y * height, 4, 0, Math.PI * 2);
    ctx.fill();
  }
}
