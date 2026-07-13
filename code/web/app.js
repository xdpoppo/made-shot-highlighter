const screens = {
  upload: document.getElementById("upload-screen"),
  progress: document.getElementById("progress-screen"),
  results: document.getElementById("results-screen"),
  error: document.getElementById("error-screen"),
};

function showScreen(name) {
  Object.values(screens).forEach((s) => s.classList.remove("active"));
  screens[name].classList.add("active");
}

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");

dropzone.addEventListener("click", () => fileInput.click());

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-over");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-over");
  })
);
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) uploadFile(file);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) uploadFile(fileInput.files[0]);
});

const stageLabel = document.getElementById("stage-label");
const progressFill = document.getElementById("progress-fill");
const frameCounter = document.getElementById("frame-counter");
const etaCounter = document.getElementById("eta-counter");

const STAGE_LABELS = {
  queued: "Queued...",
  detecting: "Scanning footage for made shots",
  cutting: "Cutting highlight clips",
  stitching: "Stitching full reel",
};

function fmtEta(seconds) {
  if (seconds == null) return "ETA --";
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `ETA ${m}m ${s}s`;
}

function fmtTime(t) {
  const m = Math.floor(t / 60);
  const s = (t % 60).toFixed(2).padStart(5, "0");
  return `${m}:${s}`;
}

let pollTimer = null;

async function uploadFile(file) {
  showScreen("progress");
  stageLabel.textContent = "Uploading...";
  progressFill.style.width = "0%";
  frameCounter.textContent = "Frame 0 / 0";
  etaCounter.textContent = "ETA --";

  const form = new FormData();
  form.append("file", file);

  try {
    const res = await fetch("/api/upload", { method: "POST", body: form });
    if (!res.ok) throw new Error(`Upload failed (${res.status})`);
    const { job_id } = await res.json();
    pollTimer = setInterval(() => pollStatus(job_id), 900);
  } catch (err) {
    showError(err.message);
  }
}

async function pollStatus(jobId) {
  try {
    const res = await fetch(`/api/status/${jobId}`);
    if (!res.ok) throw new Error(`Status check failed (${res.status})`);
    const data = await res.json();

    if (data.stage === "error") {
      clearInterval(pollTimer);
      showError(data.error || "Detection failed.");
      return;
    }

    stageLabel.textContent = STAGE_LABELS[data.stage] || data.stage;
    progressFill.style.width = `${Math.round((data.progress || 0) * 100)}%`;
    frameCounter.textContent = `Frame ${data.frames_done.toLocaleString()} / ${
      (data.frames_total || 0).toLocaleString()
    }`;
    etaCounter.textContent = fmtEta(data.eta_seconds);

    if (data.stage === "done") {
      clearInterval(pollTimer);
      renderResults(jobId, data);
    }
  } catch (err) {
    clearInterval(pollTimer);
    showError(err.message);
  }
}

function renderResults(jobId, data) {
  document.getElementById("stat-makes").textContent = data.makes.length;

  const reelBlock = document.getElementById("reel-block");
  const clipsBlock = document.getElementById("clips-block");
  const emptyState = document.getElementById("empty-state");

  if (data.reel) {
    reelBlock.classList.remove("hidden");
    const url = `/api/file/${jobId}/clips/${data.reel}`;
    document.getElementById("reel-video").src = url;
    document.getElementById("reel-download").href = url;
    document.getElementById("reel-download").setAttribute("download", "huge_highlights_reel.mp4");
  } else {
    reelBlock.classList.add("hidden");
  }

  if (data.makes.length) {
    clipsBlock.classList.remove("hidden");
    emptyState.classList.add("hidden");
    const grid = document.getElementById("clips-grid");
    grid.innerHTML = "";
    data.makes.forEach((m, i) => {
      const url = `/api/file/${jobId}/clips/${m.clip}`;
      const card = document.createElement("div");
      card.className = "clip-card";
      card.innerHTML = `
        <div class="clip-card-head">
          <span class="clip-label">Make ${i + 1}</span>
          <span class="clip-time">${fmtTime(m.time)}</span>
        </div>
        <video controls src="${url}"></video>
        <a class="btn-download-small" href="${url}" download="huge_highlights_make_${String(i + 1).padStart(2, "0")}.mp4">Download clip</a>
      `;
      grid.appendChild(card);
    });
  } else {
    clipsBlock.classList.add("hidden");
    emptyState.classList.remove("hidden");
  }

  showScreen("results");
}

function showError(message) {
  document.getElementById("error-detail").textContent = message;
  showScreen("error");
}

document.getElementById("new-video-btn").addEventListener("click", () => {
  fileInput.value = "";
  showScreen("upload");
});
document.getElementById("error-retry-btn").addEventListener("click", () => {
  fileInput.value = "";
  showScreen("upload");
});
