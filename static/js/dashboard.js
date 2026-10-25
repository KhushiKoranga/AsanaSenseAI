const poseNames = window.ASANA_POSE_NAMES || {};
const $ = (id) => document.getElementById(id);
const poseSelect = $("targetPose");
const targetStatus = $("targetStatus");
const video = $("webcamVideo");
const canvas = $("captureCanvas");
const ctx = canvas.getContext("2d");
const processedFrame = $("processedFrame");
const placeholder = $("cameraPlaceholder");
const startButton = $("startCamera");
const stopButton = $("stopCamera");

let chart = null;
let cameraStream = null;
let cameraRunning = false;
let requestInProgress = false;
let loopTimer = null;

if (window.Chart && $("accuracyChart")) {
  chart = new Chart($("accuracyChart"), {
    type: "line",
    data: { labels: [], datasets: [{ label: "Posture score", data: [], borderColor: "#8a80ff", backgroundColor: "rgba(138,128,255,.12)", fill: true, tension: .35, pointRadius: 1, borderWidth: 2 }] },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#929bb0", maxTicksLimit: 7 }, grid: { color: "#283247" } },
        y: { min: 0, max: 100, ticks: { color: "#929bb0", callback: value => value + "%" }, grid: { color: "#283247" } }
      }
    }
  });
}

function renderJoints(joints = {}) {
  const entries = Object.entries(joints || {});
  $("jointCount").textContent = `${entries.filter(([, v]) => v && v.ok).length} / ${entries.length}`;
  $("jointList").innerHTML = entries.length ? entries.map(([key, v]) => {
    const angle = Number(v.angle || 0), deviation = Number(v.deviation || 0);
    const width = Math.min(100, Math.max(3, 100 - deviation * 2));
    return `<div class="joint-row"><div class="joint-top"><span class="joint-name">${key.replaceAll("_", " ").replace(/\b\w/g, c => c.toUpperCase())}</span><span class="joint-value">${angle.toFixed(1)}° · Δ ${deviation.toFixed(1)}°</span></div><div class="joint-track"><span class="${v.ok ? "" : "bad"}" style="width:${width}%"></span></div></div>`;
  }).join("") : '<p class="muted">Waiting for body landmarks…</p>';
}

function renderSuggestions(items = []) {
  $("suggestions").innerHTML = items.length
    ? items.map(item => `<li>${String(item)}</li>`).join("")
    : '<li class="muted">No major angle differences reported, or waiting for pose detection.</li>';
}

function renderStats(data) {
  const accuracy = Math.min(100, Math.max(0, Number(data.accuracy || 0)));
  const confidence = Math.min(100, Math.max(0, Number(data.ai_confidence || 0)));
  $("predictedPose").textContent = data.predicted_pose || "Waiting…";
  $("confidence").textContent = `${confidence.toFixed(1)}%`;
  $("confidenceBar").style.width = `${confidence}%`;
  $("accuracy").textContent = `${accuracy.toFixed(1)}%`;
  $("ringValue").textContent = `${Math.round(accuracy)}%`;
  $("scoreRing").style.setProperty("--score", `${accuracy * 3.6}deg`);
  $("scoreMessage").textContent = data.pose_detected ? "Keep steady and follow your reference." : "Move back until your full body is visible.";
  $("cameraStatusFooter").textContent = data.pose_detected ? "Body landmarks detected" : "Waiting for a clear full-body view";
  $("videoTarget").textContent = data.target_name || poseNames[data.target_pose] || data.target_pose || "Target pose";
  $("modelStatus").textContent = `${data.model_ready ? "Classifier ready" : "Classifier not trained yet"} · ${data.dataset_ready ? "Reference dataset loaded" : "Reference dataset missing"}`;
  renderJoints(data.joints);
  renderSuggestions(data.suggestions);
  if (chart && Array.isArray(data.history)) {
    chart.data.labels = data.history.map((point, i) => `${i + 1}`);
    chart.data.datasets[0].data = data.history.map(point => Number(point.acc || 0));
    chart.update("none");
  }
}

async function refreshStats() {
  try {
    const response = await fetch("/stats", { cache: "no-store" });
    if (response.ok) renderStats(await response.json());
  } catch (_) {}
}

async function processOneFrame() {
  if (!cameraRunning || requestInProgress || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
  requestInProgress = true;
  try {
    const scale = Math.min(1, 640 / video.videoWidth);
    canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
    canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const response = await fetch("/process_frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: canvas.toDataURL("image/jpeg", 0.72) })
    });
    const data = await response.json();
    if (!response.ok || !data.success) throw new Error(data.error || "Frame processing failed.");
    processedFrame.src = `data:image/jpeg;base64,${data.frame}`;
    placeholder.style.display = "none";
    renderStats(data);
    $("cameraStatus").textContent = "Camera connected · AI analysing";
  } catch (error) {
    $("cameraStatus").textContent = `Analysis error: ${error.message}`;
  } finally {
    requestInProgress = false;
    if (cameraRunning) loopTimer = setTimeout(processOneFrame, 180);
  }
}

async function startCamera() {
  if (cameraRunning) return;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    $("cameraStatus").textContent = "Camera needs HTTPS or localhost in a supported browser.";
    return;
  }
  startButton.disabled = true;
  $("cameraStatus").textContent = "Requesting camera permission…";
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } }
    });
    video.srcObject = cameraStream;
    await video.play();
    cameraRunning = true;
    stopButton.disabled = false;
    placeholder.style.display = "grid";
    $("cameraStatus").textContent = "Camera started · preparing AI analysis…";
    processOneFrame();
  } catch (error) {
    if (cameraStream) cameraStream.getTracks().forEach(track => track.stop());
    cameraStream = null;
    startButton.disabled = false;
    stopButton.disabled = true;
    $("cameraStatus").textContent = error.name === "NotAllowedError"
      ? "Camera permission denied. Allow camera access in browser settings."
      : `Could not start camera: ${error.message}`;
  }
}

function stopCamera() {
  cameraRunning = false;
  if (loopTimer) clearTimeout(loopTimer);
  loopTimer = null;
  if (cameraStream) cameraStream.getTracks().forEach(track => track.stop());
  cameraStream = null;
  video.srcObject = null;
  requestInProgress = false;
  startButton.disabled = false;
  stopButton.disabled = true;
  $("cameraStatus").textContent = "Camera stopped.";
  placeholder.style.display = "grid";
  placeholder.querySelector("strong").textContent = "Your camera is ready when you are";
  placeholder.querySelector("span").textContent = "Choose Start Camera and allow browser camera permission.";
  processedFrame.removeAttribute("src");
  $("predictedPose").textContent = "Waiting…";
}

startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", stopCamera);
window.addEventListener("pagehide", stopCamera);

if (poseSelect) {
  poseSelect.addEventListener("change", async () => {
    poseSelect.disabled = true;
    targetStatus.textContent = "Updating…";
    try {
      const response = await fetch("/set_target_pose", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pose: poseSelect.value })
      });
      const result = await response.json();
      if (!response.ok || !result.success) throw new Error(result.error || "Could not update target pose.");
      $("videoTarget").textContent = result.name || poseNames[result.pose] || result.pose;
      targetStatus.textContent = "Updated";
    } catch (error) {
      targetStatus.textContent = error.message;
    } finally {
      poseSelect.disabled = false;
      setTimeout(() => { targetStatus.textContent = ""; }, 3000);
    }
  });
}

refreshStats();
setInterval(refreshStats, 2000);
