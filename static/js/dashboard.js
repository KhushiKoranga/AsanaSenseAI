const poseNames = window.ASANA_POSE_NAMES || {};
const $ = (id) => document.getElementById(id);
const poseSelect = $("targetPose");
const targetStatus = $("targetStatus");

let chart = null;
const canvas = $("accuracyChart");
if (window.Chart && canvas) {
  chart = new Chart(canvas, {
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
  $("suggestions").innerHTML = items.length ? items.map(item => `<li>${item}</li>`).join("") : '<li class="muted">No major angle differences reported, or waiting for pose detection.</li>';
}

async function refreshStats() {
  try {
    const response = await fetch("/stats", { cache: "no-store" });
    if (!response.ok) return;
    const data = await response.json();
    const accuracy = Math.min(100, Math.max(0, Number(data.accuracy || 0)));
    const confidence = Math.min(100, Math.max(0, Number(data.ai_confidence || 0)));

    $("predictedPose").textContent = data.predicted_pose || "Waiting…";
    $("confidence").textContent = `${confidence.toFixed(1)}%`;
    $("confidenceBar").style.width = `${confidence}%`;
    $("accuracy").textContent = `${accuracy.toFixed(1)}%`;
    $("ringValue").textContent = `${Math.round(accuracy)}%`;
    $("scoreRing").style.setProperty("--score", `${accuracy * 3.6}deg`);
    $("scoreMessage").textContent = data.pose_detected ? "Keep steady and follow your reference." : "Stand where your full body is visible.";
    $("cameraStatus").textContent = data.pose_detected ? "Body landmarks detected" : "Waiting for pose…";
    $("videoTarget").textContent = data.target_name || poseNames[data.target_pose] || data.target_pose || "Target pose";
    $("modelStatus").textContent = `${data.model_ready ? "Classifier ready" : "Classifier not trained yet"} · ${data.dataset_ready ? "Reference dataset loaded" : "Reference dataset missing"}`;

    renderJoints(data.joints);
    renderSuggestions(data.suggestions);
    if (chart && Array.isArray(data.history)) {
      chart.data.labels = data.history.map((point, i) => `${i + 1}`);
      chart.data.datasets[0].data = data.history.map(point => Number(point.acc || 0));
      chart.update("none");
    }
  } catch (error) {
    $("cameraStatus").textContent = "Connecting to dashboard…";
  }
}

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
setInterval(refreshStats, 700);
