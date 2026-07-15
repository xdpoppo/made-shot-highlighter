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

let currentResult = null;  // { jobId, data } — kept so team renames can re-render

function teamName(side) {
  const el = document.getElementById(side === "left" ? "team-left" : "team-right");
  const typed = el && el.value.trim();
  if (typed) return typed;
  return side === "left" ? "Team 1" : "Team 2";
}

function sideBadge(side) {
  if (side !== "left" && side !== "right") return "";
  return `<span class="side-badge ${side}"><span class="team-dot"></span><span class="js-teamname" data-side="${side}">${teamName(side)}</span></span>`;
}

// Update only the visible team-name text (no video rebuild) when a name changes.
function refreshTeamLabels() {
  document.querySelectorAll(".js-teamname[data-side]").forEach((el) => {
    el.textContent = teamName(el.dataset.side);
  });
}

const DOWN_ICON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 4v12M6 12l6 6 6-6"/></svg>`;

function renderResults(jobId, data) {
  currentResult = { jobId, data };
  document.getElementById("stat-makes").textContent = data.makes.length;

  const sides = ["left", "right"].filter((s) => data.makes.some((m) => m.side === s));
  const hasSides = sides.length > 0;

  // summary team pills
  const summaryTeams = document.getElementById("summary-teams");
  summaryTeams.innerHTML = hasSides
    ? sides
        .map((s) => {
          const n = data.makes.filter((m) => m.side === s).length;
          return `${sideBadge(s)}<span style="color:var(--text-dim);font-size:0.78rem;margin-left:-0.15rem">${n}</span>`;
        })
        .join("")
    : "";

  // full combined reel (only meaningful when >1 make)
  const reelBlock = document.getElementById("reel-block");
  if (data.reel && data.makes.length > 1) {
    reelBlock.classList.remove("hidden");
    const url = `/api/file/${jobId}/clips/${data.reel}`;
    document.getElementById("reel-video").src = url;
    document.getElementById("reel-download").href = url;
    document.getElementById("reel-download").setAttribute("download", "huge_highlights_reel.mp4");
  } else {
    reelBlock.classList.add("hidden");
  }

  // team naming bar only when sides exist
  document.getElementById("teams-block").classList.toggle("hidden", !hasSides);

  document.getElementById("empty-state").classList.toggle("hidden", data.makes.length > 0);

  if (hasSides) {
    renderByTeam(sides);
    document.getElementById("clips-block").classList.add("hidden");
  } else {
    document.getElementById("team-reels-block").classList.add("hidden");
    renderFlatClips();
  }

  showScreen("results");
}

// Two-column "By Team" view: each side gets a reel (or its single clip) plus
// compact downloadable timestamp chips for its individual makes.
function renderByTeam(sides) {
  const { jobId, data } = currentResult;
  const block = document.getElementById("team-reels-block");
  const container = document.getElementById("team-reels");
  block.classList.remove("hidden");
  container.innerHTML = "";

  sides.forEach((side) => {
    const sideMakes = data.makes
      .map((m, i) => ({ ...m, idx: i }))
      .filter((m) => m.side === side);
    if (!sideMakes.length) return;

    const reelName = (data.side_reels || {})[side] || sideMakes[0].clip;
    const reelUrl = `/api/file/${jobId}/clips/${reelName}`;
    const multi = sideMakes.length > 1;

    const chips = sideMakes
      .map((m) => {
        const clipUrl = `/api/file/${jobId}/clips/${m.clip}`;
        return `<a class="make-chip" href="${clipUrl}" download="${teamName(side).replace(/\s+/g, "_")}_${fmtTime(m.time).replace(":", "m")}.mp4" title="Download this make">${fmtTime(m.time)} ${DOWN_ICON}</a>`;
      })
      .join("");

    const col = document.createElement("div");
    col.className = `team-col ${side}`;
    col.innerHTML = `
      <div class="team-col-head">
        <span class="team-col-name"><span class="team-dot"></span><span class="js-teamname" data-side="${side}">${teamName(side)}</span></span>
        <span class="team-col-count">${sideMakes.length} make${sideMakes.length === 1 ? "" : "s"}</span>
      </div>
      <video controls src="${reelUrl}"></video>
      <div class="make-chips">${chips}</div>
      <a class="btn-download-small" href="${reelUrl}" download="${teamName(side).replace(/\s+/g, "_")}_reel.mp4">Download ${multi ? "reel" : "clip"}</a>
    `;
    container.appendChild(col);
  });
}

// Fallback when makes carry no side info: a simple flat grid of clips.
function renderFlatClips() {
  const { jobId, data } = currentResult;
  const clipsBlock = document.getElementById("clips-block");
  if (!data.makes.length) {
    clipsBlock.classList.add("hidden");
    return;
  }
  clipsBlock.classList.remove("hidden");
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
}

// Live re-label when the user types a team name — text only, no video reload
["team-left", "team-right"].forEach((id) => {
  document.getElementById(id).addEventListener("input", () => {
    if (currentResult) refreshTeamLabels();
  });
});

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
