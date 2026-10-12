
const signVideo = document.getElementById("signCamera");
const signCanvas = document.getElementById("signOverlay");
const signOutput = document.getElementById("signResult");
const signStart = document.getElementById("startSign");
const signStop = document.getElementById("stopSign");
const signSpeak = document.getElementById("speakSign");

let signStream = null;

function speakSignText(text) {
    if (!text || !("speechSynthesis" in window)) return;

    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "id-ID";
    utterance.rate = 0.95;

    window.speechSynthesis.speak(utterance);
}

signStart?.addEventListener("click", async () => {
    try {
        signStart.disabled = true;
        signOutput.textContent = "Meminta akses kamera...";

        signStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "user" },
            audio: false
        });

        signVideo.srcObject = signStream;
        await signVideo.play();

        signOutput.textContent =
            "Kamera aktif. Model penerjemah bahasa isyarat belum terhubung.";

        signStop.disabled = false;
    } catch (error) {
        signOutput.textContent =
            "Kamera tidak dapat dibuka: " + error.message;
        signStart.disabled = false;
    }
});

signStop?.addEventListener("click", () => {
    signStream?.getTracks().forEach(track => track.stop());
    signStream = null;

    signVideo.srcObject = null;
    signStart.disabled = false;
    signStop.disabled = true;

    signOutput.textContent = "Kamera dihentikan.";
});

signSpeak?.addEventListener("click", () => {
    speakSignText(signOutput.textContent);
});
