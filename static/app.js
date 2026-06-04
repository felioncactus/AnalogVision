const form = document.querySelector("#convertForm");
const dropzone = document.querySelector("#dropzone");
const input = document.querySelector("#videoInput");
const fileName = document.querySelector("#fileName");
const startButton = document.querySelector("#startButton");
const statusText = document.querySelector("#statusText");
const timeText = document.querySelector("#timeText");
const meterFill = document.querySelector("#meterFill");
const comparePanel = document.querySelector("#comparePanel");
const downloadLink = document.querySelector("#downloadLink");
const downloadGifLink = document.querySelector("#downloadGifLink");
const curtainSlider = document.querySelector("#curtainSlider");
const afterLayer = document.querySelector("#afterLayer");
const curtainHandle = document.querySelector("#curtainHandle");
const curtainStage = document.querySelector("#curtainStage");

const videos = {
  beforeCurtain: document.querySelector("#beforeCurtain"),
  afterCurtain: document.querySelector("#afterCurtain"),
  beforeSplit: document.querySelector("#beforeSplit"),
  afterSplit: document.querySelector("#afterSplit"),
};

let activeFile = null;
let pollTimer = null;
let curtainSyncAttached = false;

dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.classList.add("dragging");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("dragging");
});

dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragging");
  const file = event.dataTransfer.files[0];
  if (file) {
    activeFile = file;
    fileName.textContent = file.name;
  }
});

input.addEventListener("change", () => {
  const file = input.files[0];
  if (file) {
    activeFile = file;
    fileName.textContent = file.name;
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = activeFile || input.files[0];
  if (!file) {
    setStatus("Choose a video first.", 0);
    return;
  }

  const body = new FormData();
  body.append("video", file);
  body.append("style", document.querySelector("#style").value);
  body.append("convert43", document.querySelector("#convert43").checked ? "true" : "false");
  body.append("degradeAudio", document.querySelector("#degradeAudio").checked ? "true" : "false");

  startButton.disabled = true;
  comparePanel.hidden = true;
  downloadLink.hidden = true;
  downloadGifLink.hidden = true;
  setStatus("Uploading video...", 0.02);

  const response = await fetch("/api/convert", { method: "POST", body });
  const data = await response.json();
  if (!response.ok) {
    startButton.disabled = false;
    setStatus(data.error || "Upload failed.", 0);
    return;
  }
  pollJob(data.job_id);
});

function pollJob(jobId) {
  clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    const response = await fetch(`/api/jobs/${jobId}`);
    const job = await response.json();
    setStatus(job.message || job.state, job.progress || 0, job);

    if (job.state === "complete") {
      clearInterval(pollTimer);
      startButton.disabled = false;
      showResult(job);
    }

    if (job.state === "error") {
      clearInterval(pollTimer);
      startButton.disabled = false;
      statusText.textContent = job.message || "Conversion failed.";
      statusText.style.color = "var(--danger)";
    }
  }, 900);
}

function setStatus(message, progress) {
  statusText.style.color = "";
  statusText.textContent = message;
  meterFill.style.width = `${Math.round(progress * 100)}%`;
  timeText.textContent = "ETA --";
  if (arguments.length > 2) {
    const job = arguments[2];
    const elapsed = formatTime(job.elapsed_seconds || 0);
    const eta = job.eta_seconds === null || job.eta_seconds === undefined ? "--" : formatTime(job.eta_seconds);
    timeText.textContent = `Elapsed ${elapsed} / ETA ${eta}`;
  }
}

function showResult(job) {
  const before = job.input_url;
  const after = job.output_url;
  for (const video of [videos.beforeCurtain, videos.beforeSplit]) {
    video.src = before;
  }
  for (const video of [videos.afterCurtain, videos.afterSplit]) {
    video.src = after;
  }
  downloadLink.href = job.download_url;
  downloadLink.download = "";
  downloadLink.hidden = false;
  if (job.gif_download_url) {
    downloadGifLink.href = job.gif_download_url;
    downloadGifLink.download = "";
    downloadGifLink.hidden = false;
  }
  comparePanel.hidden = false;
  updateCurtain();
  syncCurtainPlayback();
}

function syncCurtainPlayback() {
  videos.beforeCurtain.play().catch(() => {});
  videos.afterCurtain.play().catch(() => {});
  if (curtainSyncAttached) {
    return;
  }
  curtainSyncAttached = true;
  videos.beforeCurtain.addEventListener("timeupdate", () => {
    if (Math.abs(videos.beforeCurtain.currentTime - videos.afterCurtain.currentTime) > 0.18) {
      videos.afterCurtain.currentTime = videos.beforeCurtain.currentTime;
    }
  });
}

function formatTime(seconds) {
  const safe = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(safe / 60);
  const rest = Math.floor(safe % 60);
  if (minutes === 0) {
    return `${rest}s`;
  }
  return `${minutes}m ${String(rest).padStart(2, "0")}s`;
}

curtainSlider.addEventListener("input", updateCurtain);
window.addEventListener("resize", updateCurtain);

function updateCurtain() {
  const value = Number(curtainSlider.value);
  const width = curtainStage.clientWidth || 1;
  afterLayer.style.width = `${100 - value}%`;
  afterLayer.style.setProperty("--stage-width", `${width}px`);
  curtainHandle.style.left = `${value}%`;
}

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((tab) => tab.classList.remove("active"));
    document.querySelectorAll(".compare-view").forEach((view) => view.classList.remove("active"));
    button.classList.add("active");
    document.querySelector(`#${button.dataset.view}View`).classList.add("active");
  });
});
