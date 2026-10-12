/**
 * Inti matematika Jemari — dipakai bersama oleh app.js (demo) dan
 * capture.js (perekam dataset).
 *
 * PENTING: normalizeLandmarks() di sini harus identik dengan
 * normalize_landmarks() di scripts/extract_landmarks.py. Kalau salah satu
 * berubah, model hasil training tidak akan cocok dengan runtime browser.
 */

/**
 * Normalisasi landmark tangan: invarian terhadap posisi dan ukuran tangan.
 * @param {Array<{x:number,y:number,z:number}>} landmarks 21 titik MediaPipe
 * @returns {number[]} vektor fitur 63 dimensi (21 × [x, y, z])
 */
export function normalizeLandmarks(landmarks) {
  const wrist = landmarks[0];

  // 1. Geser sehingga pergelangan tangan berada di origin.
  const translated = landmarks.map((lm) => [lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z]);

  // 2. Bagi dengan jarak terjauh pergelangan → landmark mana pun.
  let scale = 0;
  for (const [x, y, z] of translated) {
    const d = Math.sqrt(x * x + y * y + z * z);
    if (d > scale) scale = d;
  }
  if (scale < 1e-6) scale = 1;

  // 3. Ratakan menjadi vektor 63 dimensi.
  const out = new Array(63);
  let i = 0;
  for (const [x, y, z] of translated) {
    out[i++] = x / scale;
    out[i++] = y / scale;
    out[i++] = z / scale;
  }
  return out;
}

/** Terapkan StandardScaler yang sudah disimpan di model.json. */
function applyScaler(features, scaler) {
  if (!scaler) return features;
  return features.map((v, i) => (v - scaler.mean[i]) / (scaler.scale[i] || 1));
}

/** K-Nearest Neighbours: mayoritas dari k tetangga terdekat. */
function knnPredict(model, features) {
  const scaled = applyScaler(features, model.scaler);
  const k = model.k;
  const dists = [];

  for (let i = 0; i < model.samples.length; i++) {
    const s = model.samples[i];
    let d = 0;
    for (let j = 0; j < scaled.length; j++) {
      const diff = scaled[j] - s[j];
      d += diff * diff;
    }
    dists.push([d, model.labels[i]]);
  }

  dists.sort((a, b) => a[0] - b[0]);

  const votes = new Map();
  for (let i = 0; i < Math.min(k, dists.length); i++) {
    const label = dists[i][1];
    votes.set(label, (votes.get(label) || 0) + 1);
  }

  let bestLabel = -1;
  let bestVotes = -1;
  for (const [label, count] of votes) {
    // Seri diselesaikan ke indeks kelas terkecil — sama dengan np.argmax di Python.
    if (count > bestVotes || (count === bestVotes && label < bestLabel)) {
      bestVotes = count;
      bestLabel = label;
    }
  }

  return { index: bestLabel, confidence: bestVotes / Math.min(k, dists.length) };
}

function relu(x) {
  return x > 0 ? x : 0;
}

function softmax(values) {
  const max = Math.max(...values);
  const exps = values.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

/** MLP kecil: relu di semua layer tersembunyi, softmax di output. */
function mlpPredict(model, features) {
  let vec = applyScaler(features, model.scaler);

  for (let li = 0; li < model.layers.length; li++) {
    const layer = model.layers[li];
    const isLast = li === model.layers.length - 1;

    // layer.weights berbentuk [in][out], sama dengan coefs_ milik scikit-learn.
    const out = layer.bias.slice();
    for (let i = 0; i < vec.length; i++) {
      const row = layer.weights[i];
      const v = vec[i];
      for (let o = 0; o < row.length; o++) out[o] += v * row[o];
    }

    if (!isLast) {
      for (let o = 0; o < out.length; o++) out[o] = relu(out[o]);
    }
    vec = out;
  }

  const probs = model.out_activation === "softmax" ? softmax(vec) : vec;
  let index = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[index]) index = i;

  return { index, confidence: probs[index] };
}

/**
 * Klasifikasi satu frame.
 * @returns {{label:string, index:number, confidence:number}|null}
 */
export function predict(model, features) {
  if (!model || !model.classes?.length) return null;
  const result = model.type === "mlp" ? mlpPredict(model, features) : knnPredict(model, features);
  return {
    label: model.classes[result.index],
    index: result.index,
    confidence: result.confidence,
  };
}

/**
 * Perata-rataan temporal: kumpulkan prediksi beberapa frame terakhir, lalu
 * keluarkan label hanya bila mayoritas stabil dan confidence cukup tinggi.
 * Ini yang membuat teks di layar tidak "berkedip" antar huruf.
 */
export class TemporalSmoother {
  constructor({ windowSize = 9, minVotes = 6, minConfidence = 0.55 } = {}) {
    this.windowSize = windowSize;
    this.minVotes = minVotes;
    this.minConfidence = minConfidence;
    this.buffer = [];
    this.blocked = null; // huruf terakhir yang sudah di-commit
  }

  /** @returns {{label:string, confidence:number}|null} label yang sudah stabil */
  push(prediction) {
    if (!prediction || prediction.confidence < this.minConfidence) {
      // Jeda (tangan keluar / gestur transisi) membuka blokir pengulangan.
      this.blocked = null;
      return null;
    }

    // Jangan meng-commit huruf yang sama dua kali berturut-turut: pengguna
    // yang menahan pose tidak boleh mendapat "AAAA" di transkrip.
    if (this.blocked === prediction.label) return null;
    this.blocked = null;

    this.buffer.push(prediction);
    if (this.buffer.length > this.windowSize) this.buffer.shift();
    if (this.buffer.length < this.minVotes) return null;

    const votes = new Map();
    for (const p of this.buffer) {
      const entry = votes.get(p.label) || { count: 0, conf: 0 };
      entry.count += 1;
      entry.conf = Math.max(entry.conf, p.confidence);
      votes.set(p.label, entry);
    }

    let best = null;
    for (const [label, entry] of votes) {
      if (!best || entry.count > best.count) best = { label, ...entry };
    }

    if (!best || best.count < this.minVotes) return null;

    this.buffer = []; // reset supaya huruf berikutnya butuh gestur baru
    this.blocked = best.label;
    return { label: best.label, confidence: best.conf };
  }

  reset() {
    this.buffer = [];
    this.blocked = null;
  }
}
