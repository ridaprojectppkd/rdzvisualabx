/**
 * Jemari — demo terjemahan real-time.
 * Alur: kamera → MediaPipe landmark → normalize → classifier → smoothing → UI/TTS.
 */

import { normalizeLandmarks, predict, TemporalSmoother } from "./landmarks.js";
import { createHandLandmarker, startCamera, stopCamera, drawHand } from "./hands.js";

const $ = (id) => document.getElementById(id);

const video = $("video");
const overlay = $("overlay");
const ctx = overlay.getContext("2d");

const startBtn = $("startBtn");
const stopBtn = $("stopBtn");
const stageHint = $("stageHint");
const modelBadge = $("modelBadge");

const letterEl = $("letter");
const letterMeta = $("letterMeta");
const meterFill = $("meterFill");
const transcriptEl = $("transcript");

const speakBtn = $("speakBtn");
const copyBtn = $("copyBtn");
const clearBtn = $("clearBtn");

const state = {
    model: null,
    landmarker: null,
    running: false,
    smoother: new TemporalSmoother({
        // ?minconf=0.4 — longgarkan/ketatkan ambang (untuk perangkat lambat).
        minConfidence: Math.min(
            Math.max(parseFloat(new URLSearchParams(location.search).get("minconf")) || 0.55, 0.1),
            0.95,
        ),
    }),
    transcript: "",
    rafId: null,
    lastVideoTime: -1,
    voice: null,
};

/* ----------------------------- model & suara ----------------------------- */

async function loadModel() {
    try {
        const res = await fetch("model.json", { cache: "no-store" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        state.model = await res.json();
        const m = state.model.meta || {};
        modelBadge.textContent = `${m.model?.toUpperCase() || "model"} · ${m.n_classes || 0} classes · accuracy ${m.test_accuracy ? (m.test_accuracy * 100).toFixed(1) + "%" : "–"
            }`;
        modelBadge.classList.remove("badge-error");
    } catch (err) {
        state.model = null;
        modelBadge.textContent = "model.json missing — run scripts/train.py";
        modelBadge.classList.add("badge-error");
        stageHint.textContent = "Model not trained yet. Record data in capture.html, then run scripts/train.py.";
        console.error(err);
    }
}

function pickIndonesianVoice() {
    const voices = window.speechSynthesis?.getVoices() || [];
    state.voice =
        voices.find((v) => v.lang?.toLowerCase().startsWith("id")) ||
        voices.find((v) => v.lang?.toLowerCase().startsWith("en")) ||
        null;
}

function speak(text) {
    if (!text || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const utter = new SpeechSynthesisUtterance(text);
    utter.lang = "id-ID";
    if (state.voice) utter.voice = state.voice;
    utter.rate = 0.95;
    window.speechSynthesis.speak(utter);
}

/* -------------------------------- kamera -------------------------------- */

function resizeOverlay() {
    overlay.width = video.videoWidth || 640;
    overlay.height = video.videoHeight || 480;
}

async function start() {
    startBtn.disabled = true;
    stageHint.textContent = "Preparing camera and model…";

    try {
        if (!state.landmarker) state.landmarker = await createHandLandmarker();
        await startCamera(video);
        resizeOverlay();
        state.running = true;
        state.smoother.reset();
        stopBtn.disabled = false;
        stageHint.textContent = "Hand detected → letters will appear.";
        loop();
    } catch (err) {
        console.error(err);
        startBtn.disabled = false;
        stageHint.textContent = `Failed to start: ${err.message}. Check camera permission & connection (MediaPipe loads from a CDN).`;
    }
}

function stop() {
    state.running = false;
    if (state.rafId) cancelAnimationFrame(state.rafId);
    stopCamera(video);
    drawHand(ctx, null, overlay.width, overlay.height);
    startBtn.disabled = false;
    stopBtn.disabled = true;
    letterEl.textContent = "–";
    letterMeta.textContent = "Camera stopped.";
    meterFill.style.width = "0%";
}

function loop() {
    if (!state.running) return;

    if (video.currentTime !== state.lastVideoTime && video.readyState >= 2) {
        state.lastVideoTime = video.currentTime;
        const result = state.landmarker.detectForVideo(video, performance.now());
        const landmarks = result.landmarks?.[0] || null;

        drawHand(ctx, landmarks, overlay.width, overlay.height);

        if (landmarks) {
            handleFrame(landmarks);
        } else {
            letterMeta.textContent = "No hand detected — show your hand to the camera.";
            meterFill.style.width = "0%";
        }
    }

    state.rafId = requestAnimationFrame(loop);
}

function handleFrame(landmarks) {
    if (!state.model) {
        letterMeta.textContent = "Model not loaded.";
        return;
    }

    const features = normalizeLandmarks(landmarks);
    const prediction = predict(state.model, features);
    if (!prediction) return;

    letterEl.textContent = prediction.label;
    meterFill.style.width = `${Math.round(prediction.confidence * 100)}%`;
    letterMeta.textContent = `Confidence ${(prediction.confidence * 100).toFixed(0)}%`;

    const stable = state.smoother.push(prediction);
    if (stable) {
        state.transcript += stable.label;
        transcriptEl.textContent = state.transcript;
        transcriptEl.scrollTop = transcriptEl.scrollHeight;
    }
}

/* ------------------------------- kontrol -------------------------------- */

startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", stop);

speakBtn.addEventListener("click", () => {
    const text = state.transcript.trim();
    if (!text) {
        letterMeta.textContent = "Nothing to speak yet.";
        return;
    }
    speak(text);
});

copyBtn.addEventListener("click", async () => {
    if (!state.transcript) return;
    try {
        await navigator.clipboard.writeText(state.transcript);
        copyBtn.textContent = "Copied ✓";
        setTimeout(() => (copyBtn.textContent = "Copy"), 1200);
    } catch {
        letterMeta.textContent = "Browser denied clipboard access.";
    }
});

clearBtn.addEventListener("click", () => {
    state.transcript = "";
    transcriptEl.textContent = "";
    state.smoother.reset();
});

video.addEventListener("resize", resizeOverlay);
window.speechSynthesis?.addEventListener("voiceschanged", pickIndonesianVoice);

loadModel();
pickIndonesianVoice();
