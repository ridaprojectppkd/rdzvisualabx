
import * as ort from
    "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.min.mjs";

const MODEL_PATH = "./web-models/yolo/yolov8n.onnx";
const INPUT_SIZE = 640;
const CONFIDENCE_THRESHOLD = 0.4;
const IOU_THRESHOLD = 0.45;

const LABELS_ID = [
    "orang", "sepeda", "mobil", "motor", "pesawat",
    "bus", "kereta", "truk", "perahu", "lampu lalu lintas",
    "hidran", "rambu stop", "meter parkir", "bangku", "burung",
    "kucing", "anjing", "kuda", "domba", "sapi",
    "gajah", "beruang", "zebra", "jerapah", "tas",
    "payung", "tas tangan", "dasi", "koper", "cakram terbang",
    "ski", "papan seluncur", "bola", "layang-layang", "tongkat bisbol",
    "sarung tangan bisbol", "skateboard", "papan selancar", "raket tenis",
    "botol", "gelas anggur", "cangkir", "garpu", "pisau",
    "sendok", "mangkuk", "pisang", "apel", "roti lapis",
    "jeruk", "brokoli", "wortel", "hot dog", "pizza",
    "donat", "kue", "kursi", "sofa", "tanaman pot",
    "tempat tidur", "meja makan", "toilet", "televisi", "laptop",
    "mouse", "remote", "keyboard", "ponsel", "microwave",
    "oven", "pemanggang roti", "wastafel", "kulkas", "buku",
    "jam", "vas", "gunting", "boneka beruang", "pengering rambut",
    "sikat gigi"
];

const video = document.getElementById("camera");
const canvas = document.getElementById("overlay");
const output = document.getElementById("objectResult");
const ctx = canvas?.getContext("2d");

let session = null;
let rafId = null;
let lastInference = 0;
let running = false;
let busy = false;

function getPosition(centerX, width) {
    if (centerX < width / 3) return "di kiri";
    if (centerX < (2 * width) / 3) return "di tengah";
    return "di kanan";
}

function intersectionOverUnion(a, b) {
    const x1 = Math.max(a.x1, b.x1);
    const y1 = Math.max(a.y1, b.y1);
    const x2 = Math.min(a.x2, b.x2);
    const y2 = Math.min(a.y2, b.y2);

    const intersection =
        Math.max(0, x2 - x1) * Math.max(0, y2 - y1);

    const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);

    return intersection / Math.max(areaA + areaB - intersection, 1e-6);
}

function nms(detections) {
    const result = [];

    detections.sort((a, b) => b.score - a.score);

    while (detections.length) {
        const current = detections.shift();
        result.push(current);

        for (let i = detections.length - 1; i >= 0; i--) {
            if (
                detections[i].classId === current.classId &&
                intersectionOverUnion(detections[i], current) > IOU_THRESHOLD
            ) {
                detections.splice(i, 1);
            }
        }

        if (result.length >= 100) break;
    }

    return result;
}

function preprocess(videoElement) {
    const sourceWidth = videoElement.videoWidth;
    const sourceHeight = videoElement.videoHeight;

    // Letterbox mempertahankan rasio aspek gambar.
    const scale = Math.min(
        INPUT_SIZE / sourceWidth,
        INPUT_SIZE / sourceHeight
    );

    const resizedWidth = Math.round(sourceWidth * scale);
    const resizedHeight = Math.round(sourceHeight * scale);
    const padX = (INPUT_SIZE - resizedWidth) / 2;
    const padY = (INPUT_SIZE - resizedHeight) / 2;

    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = INPUT_SIZE;
    tempCanvas.height = INPUT_SIZE;

    const tempCtx = tempCanvas.getContext("2d", {
        willReadFrequently: true
    });

    tempCtx.fillStyle = "rgb(114,114,114)";
    tempCtx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);

    tempCtx.drawImage(
        videoElement,
        0, 0, sourceWidth, sourceHeight,
        padX, padY, resizedWidth, resizedHeight
    );

    const pixels = tempCtx.getImageData(
        0, 0, INPUT_SIZE, INPUT_SIZE
    ).data;

    const chw = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
    const plane = INPUT_SIZE * INPUT_SIZE;

    for (let i = 0; i < plane; i++) {
        chw[i] = pixels[i * 4] / 255;
        chw[plane + i] = pixels[i * 4 + 1] / 255;
        chw[2 * plane + i] = pixels[i * 4 + 2] / 255;
    }

    return {
        tensor: new ort.Tensor("float32", chw, [
            1, 3, INPUT_SIZE, INPUT_SIZE
        ]),
        scale,
        padX,
        padY,
        sourceWidth,
        sourceHeight
    };
}

function decode(outputTensor, meta) {
    const data = outputTensor.data;
    const dims = outputTensor.dims;

    // YOLOv8 ONNX bisa menghasilkan [1,84,8400]
    // atau [1,8400,84], bergantung format ekspor.
    const channelsFirst = dims[1] === 84;
    const numCandidates = channelsFirst ? dims[2] : dims[1];
    const numChannels = channelsFirst ? dims[1] : dims[2];

    if (numChannels !== 84) {
        throw new Error(`Bentuk output YOLO tidak dikenal: ${dims}`);
    }

    const getValue = (candidate, channel) =>
        channelsFirst
            ? data[channel * numCandidates + candidate]
            : data[candidate * numChannels + channel];

    const detections = [];

    for (let i = 0; i < numCandidates; i++) {
        let classId = 0;
        let score = 0;

        for (let c = 0; c < 80; c++) {
            const value = getValue(i, 4 + c);
            if (value > score) {
                score = value;
                classId = c;
            }
        }

        if (score < CONFIDENCE_THRESHOLD) continue;

        const cx = getValue(i, 0);
        const cy = getValue(i, 1);
        const w = getValue(i, 2);
        const h = getValue(i, 3);

        const x1 = (cx - w / 2 - meta.padX) / meta.scale;
        const y1 = (cy - h / 2 - meta.padY) / meta.scale;
        const x2 = (cx + w / 2 - meta.padX) / meta.scale;
        const y2 = (cy + h / 2 - meta.padY) / meta.scale;

        detections.push({
            classId,
            score,
            x1: Math.max(0, Math.min(meta.sourceWidth, x1)),
            y1: Math.max(0, Math.min(meta.sourceHeight, y1)),
            x2: Math.max(0, Math.min(meta.sourceWidth, x2)),
            y2: Math.max(0, Math.min(meta.sourceHeight, y2))
        });
    }

    return nms(detections);
}

const audioToggle = document.getElementById("audioEnabled");

let lastSpokenText = "";
let lastSpokenAt = 0;

function speakDetections(detections) {
    const currentToggle = document.getElementById("audioEnabled") ?? audioToggle;

    if (!currentToggle?.checked) return;
    if (!("speechSynthesis" in window)) return;
    if (!Array.isArray(detections) || detections.length === 0) {
        lastSpokenText = "";
        return;
    }

    const topObjects = [...detections]
        .sort((a, b) => (b.score ?? 0) - (a.score ?? 0))
        .slice(0, 3);

    const message = topObjects
        .map((item) => {
            const label = LABELS_ID[item.classId] ?? `objek ${item.classId}`;
            const position = getPosition(
                (item.x1 + item.x2) / 2,
                video?.videoWidth || canvas?.width || 1
            );

            return `${label} ${position}`;
        })
        .filter(Boolean)
        .join(". ");

    if (!message) return;
    if (message === lastSpokenText || Date.now() - lastSpokenAt < 3000) return;

    lastSpokenText = message;
    lastSpokenAt = Date.now();

    const utterance = new SpeechSynthesisUtterance(
        `Objek terdeteksi. ${message}`
    );

    utterance.lang = "id-ID";
    utterance.rate = 0.9;
    utterance.volume = 1;

    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
}

function drawDetections(detections) {
    if (!ctx || !canvas || !video.videoWidth) return;

    const width = video.videoWidth;
    const height = video.videoHeight;

    if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
    }

    ctx.clearRect(0, 0, width, height);
    ctx.lineWidth = Math.max(2, width / 320);
    ctx.font = `${Math.max(14, width / 35)}px sans-serif`;

    const summaries = [];

    for (const det of detections) {
        const name = LABELS_ID[det.classId] ?? `objek ${det.classId}`;
        const position = getPosition((det.x1 + det.x2) / 2, width);

        ctx.strokeStyle = "#00e676";
        ctx.fillStyle = "#00e676";
        ctx.strokeRect(
            det.x1, det.y1,
            det.x2 - det.x1,
            det.y2 - det.y1
        );

        const label =
            `${name} ${Math.round(det.score * 100)}% ${position}`;

        const textWidth = ctx.measureText(label).width;
        const labelY = Math.max(22, det.y1);

        ctx.fillRect(
            det.x1,
            labelY - 22,
            textWidth + 10,
            24
        );

        ctx.fillStyle = "#002b19";
        ctx.fillText(label, det.x1 + 5, labelY - 5);

        summaries.push(label);
    }

    output.textContent = summaries.length
        ? summaries.join(" • ")
        : "Belum ada objek yang terdeteksi.";

    speakDetections(detections);
}

async function processFrame() {
    // Kode processFrame yang sudah ada tetap dipertahankan.

    if (!running) return;

    const now = performance.now();

    // Batasi inferensi agar HP tidak terlalu terbebani.
    if (
        session &&
        !busy &&
        video.readyState >= 2 &&
        now - lastInference >= 150
    ) {
        busy = true;
        lastInference = now;

        try {
            const meta = preprocess(video);
            const feeds = {
                [session.inputNames[0]]: meta.tensor
            };

            const results = await session.run(feeds);
            const detections = decode(
                results[session.outputNames[0]],
                meta
            );

            drawDetections(detections);
        } catch (error) {
            console.error("Inferensi YOLO gagal:", error);
            output.textContent = "Gagal menjalankan deteksi. Periksa Console.";
        } finally {
            busy = false;
        }
    }

    rafId = requestAnimationFrame(processFrame);
}

export async function initObjectDetection(videoElement = video) {
    if (!videoElement || !output) {
        throw new Error("Elemen #camera atau #objectResult tidak ditemukan.");
    }

    output.textContent = "Memuat model YOLO...";

    session = await ort.InferenceSession.create(MODEL_PATH, {
        executionProviders: ["wasm"]
    });

    videoElement.addEventListener("loadedmetadata", () => {
        if (canvas) {
            canvas.width = videoElement.videoWidth;
            canvas.height = videoElement.videoHeight;
        }
    });

    running = true;
    output.textContent = "Model siap. Mencari objek...";
    processFrame();
}

export function stopObjectDetection() {
    running = false;

    if (rafId !== null) {
        cancelAnimationFrame(rafId);
        rafId = null;
    }

    busy = false;

    if (session) {
        session.release();
        session = null;
    }

    if (ctx && canvas) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

}

const startButton = document.getElementById("startDetection");
const stopButton = document.getElementById("stopDetection");

let cameraStream = null;
let starting = false;

async function startCameraDetection() {
    if (starting || cameraStream) return;

    starting = true;
    startButton.disabled = true;

    try {
        if (!navigator.mediaDevices?.getUserMedia) {
            throw new Error(
                "Kamera memerlukan HTTPS atau localhost."
            );
        }

        output.textContent = "Meminta izin kamera...";

        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: { ideal: "environment" }
            },
            audio: false
        });

        video.srcObject = cameraStream;
        await video.play();

        output.textContent = "Memuat model AI...";

        await initObjectDetection(video);

        startButton.disabled = true;
        stopButton.disabled = false;

    } catch (error) {
        console.error("Gagal memulai deteksi:", error);

        output.textContent =
            "Kamera atau model gagal dimulai: " + error.message;

        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = null;
        }

        video.srcObject = null;
        startButton.disabled = false;
        stopButton.disabled = true;
    } finally {
        starting = false;
    }
}

function stopCameraDetection() {
    stopObjectDetection();

    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }

    video.srcObject = null;
    startButton.disabled = false;
    stopButton.disabled = true;

    if (output) {
        output.textContent = "Kamera dihentikan.";
    }

    if (ctx && canvas) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
}

startButton?.addEventListener("click", startCameraDetection);
stopButton?.addEventListener("click", stopCameraDetection);
document.getElementById("audioEnabled")?.addEventListener("change", (event) => {
    if (event.target.checked) {
        const suara = new SpeechSynthesisUtterance(
            "Halo, RDZ Visualab siap membantu."
        );

        suara.lang = "id-ID";
        suara.volume = 1;
        window.speechSynthesis.speak(suara);
    }
});

