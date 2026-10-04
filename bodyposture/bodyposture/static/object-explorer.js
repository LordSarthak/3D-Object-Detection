const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const context = overlay.getContext("2d");
const cameraSelect = document.getElementById("camera-select");
const toggleButton = document.getElementById("scan-toggle");
const scanOnceButton = document.getElementById("scan-once");
const errorBox = document.getElementById("error");
const stageHint = document.getElementById("stage-hint");

let stream = null;
let isScanning = false;
let requestInProgress = false;
let animationHandle = null;
let lastScanAt = 0;
let currentRequest = null;
let lastResult = null;

function setStatus(label, state = "") {
  const status = document.getElementById("status");
  status.classList.toggle("running", state === "running");
  status.classList.toggle("busy", state === "busy");
  document.getElementById("status-label").textContent = label;
}

function setError(message) {
  errorBox.textContent = message;
  errorBox.classList.add("visible");
}

function clearError() {
  errorBox.textContent = "";
  errorBox.classList.remove("visible");
}

async function refreshCameras() {
  const selected = cameraSelect.value;
  const devices = await navigator.mediaDevices.enumerateDevices();
  const cameras = devices.filter(device => device.kind === "videoinput");
  cameraSelect.replaceChildren(new Option("Default camera", ""));
  cameras.forEach((device, index) => {
    cameraSelect.add(new Option(device.label || `Camera ${index + 1}`, device.deviceId));
  });
  if (cameras.some(camera => camera.deviceId === selected)) cameraSelect.value = selected;
}

async function openCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("Camera access requires a supported browser. Open this app at http://127.0.0.1:5000 in Chrome or Edge.");
  }
  const videoOptions = {
    width: { ideal: 960 },
    height: { ideal: 540 },
    frameRate: { ideal: 30, max: 30 },
  };
  if (cameraSelect.value) videoOptions.deviceId = { exact: cameraSelect.value };
  stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: videoOptions });
  video.srcObject = stream;
  await video.play();
  await new Promise(resolve => {
    if (video.videoWidth) resolve();
    else video.addEventListener("loadedmetadata", resolve, { once: true });
  });
  video.classList.add("visible");
  document.getElementById("placeholder").hidden = true;
  cameraSelect.disabled = true;
  await refreshCameras();
  cameraSelect.disabled = true;
}

function makeFrame() {
  const scale = Math.min(1, 768 / Math.max(video.videoWidth, video.videoHeight));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  canvas.getContext("2d", { alpha: false }).drawImage(video, 0, 0, canvas.width, canvas.height);
  return { image: canvas.toDataURL("image/jpeg", 0.78), width: canvas.width, height: canvas.height };
}

async function analyzeCurrentFrame() {
  if (!stream || requestInProgress || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
  requestInProgress = true;
  scanOnceButton.disabled = true;
  lastScanAt = performance.now();
  setStatus("Analyzing frame…", "busy");
  stageHint.textContent = "AI is reading the current frame · first scan may take longer";

  const frame = makeFrame();
  const form = new FormData();
  form.set("image", frame.image.slice(frame.image.indexOf(",") + 1));
  currentRequest = new AbortController();
  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      body: form,
      signal: currentRequest.signal,
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `Image analysis failed (${response.status}).`);
    renderResults(result);
    stageHint.textContent = "Objects in green · detailed regions in amber and blue";
    document.getElementById("model-state").textContent = "Florence-2 base · model ready on this computer";
  } catch (error) {
    if (error.name !== "AbortError") {
      setError(error.message || "The image could not be analyzed.");
      stageHint.textContent = "Scan paused · check the message below the preview";
      setStatus("Scan issue");
    }
  } finally {
    currentRequest = null;
    requestInProgress = false;
    scanOnceButton.disabled = requestInProgress;
    if (isScanning && stream) {
      setStatus("Camera live · results update as they arrive", "running");
    }
  }
}

function renderResults(result) {
  lastResult = result;
  document.getElementById("object-count").textContent = result.objects.length;
  document.getElementById("part-count").textContent = result.parts.length;
  renderList("object-list", result.objects, "No familiar object labels in this frame.");
  renderList("part-list", result.parts, "No separate component details were returned for this view.");
  renderStageLabels(result);
  drawBoxes(result);
  clearError();
}

function renderStageLabels(result) {
  const labels = document.getElementById("stage-labels");
  labels.replaceChildren();
  const entries = [
    ...result.objects.map(item => ({ item, prefix: "Object", className: "" })),
    ...result.parts.map(item => ({ item, prefix: "Detail", className: "detail" })),
  ];
  entries.slice(0, 8).forEach(({ item, prefix, className }) => {
    const label = document.createElement("span");
    label.className = `stage-label ${className}`.trim();
    label.textContent = `${prefix}: ${item.label}`;
    labels.append(label);
  });
}

function renderList(elementId, items, emptyText) {
  const list = document.getElementById(elementId);
  list.replaceChildren();
  if (!items.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = emptyText;
    list.append(empty);
    return;
  }
  items.forEach(item => {
    const row = document.createElement("div");
    row.className = "label-item";
    const swatch = document.createElement("span");
    swatch.className = "label-swatch";
    const label = document.createElement("span");
    label.className = "label-text";
    label.textContent = item.label;
    row.append(swatch, label);
    list.append(row);
  });
}

function drawBoxes(result) {
  const bounds = overlay.getBoundingClientRect();
  if (!bounds.width || !bounds.height) return;
  const ratio = window.devicePixelRatio || 1;
  overlay.width = Math.round(bounds.width * ratio);
  overlay.height = Math.round(bounds.height * ratio);
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, bounds.width, bounds.height);

  const scale = Math.min(bounds.width / result.width, bounds.height / result.height);
  const offsetX = (bounds.width - result.width * scale) / 2;
  const offsetY = (bounds.height - result.height * scale) / 2;
  const groups = [
    { items: result.objects, color: "#75bd8b" },
    { items: result.parts, color: "#e2a94e" },
  ];
  groups.forEach(group => group.items.forEach(item => {
    const [x1, y1, x2, y2] = item.box;
    const x = offsetX + x1 * scale;
    const y = offsetY + y1 * scale;
    const width = Math.max(0, (x2 - x1) * scale);
    const height = Math.max(0, (y2 - y1) * scale);
    if (width < 1 || height < 1) return;
    context.strokeStyle = group.color;
    context.lineWidth = 2.5;
    context.strokeRect(x, y, width, height);
    const label = item.label.slice(0, 40);
    context.font = "700 13px Segoe UI, sans-serif";
    const labelWidth = Math.min(bounds.width - 12, context.measureText(label).width + 16);
    const labelX = Math.max(6, Math.min(bounds.width - labelWidth - 6, x));
    const labelTop = y >= 25 ? y - 24 : Math.min(bounds.height - 22, y + 3);
    context.fillStyle = "rgb(20 28 23 / 94%)";
    context.fillRect(labelX, labelTop, labelWidth, 22);
    context.fillStyle = group.color;
    context.fillRect(labelX, labelTop, 3, 22);
    context.fillStyle = "#fff";
    context.fillText(label, labelX + 9, labelTop + 15, labelWidth - 12);
  }));
}

async function closeCamera() {
  isScanning = false;
  if (animationHandle !== null) {
    if (typeof video.cancelVideoFrameCallback === "function") video.cancelVideoFrameCallback(animationHandle);
    else cancelAnimationFrame(animationHandle);
    animationHandle = null;
  }
  currentRequest?.abort();
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }
  video.pause();
  video.srcObject = null;
  video.classList.remove("visible");
  document.getElementById("placeholder").hidden = false;
  cameraSelect.disabled = false;
  context.clearRect(0, 0, overlay.width, overlay.height);
  document.getElementById("stage-labels").replaceChildren();
  setStatus("Camera off");
  toggleButton.textContent = "Start scanning";
  scanOnceButton.disabled = false;
  stageHint.textContent = "Start scanning to load the AI model";
}

function scanLoop() {
  if (!isScanning) return;
  const now = performance.now();
  if (!requestInProgress && now - lastScanAt >= 1700) void analyzeCurrentFrame();
  animationHandle = requestAnimationFrame(scanLoop);
}

async function startScanning() {
  clearError();
  toggleButton.disabled = true;
  scanOnceButton.disabled = true;
  setStatus("Opening camera…", "busy");
  stageHint.textContent = "Connecting camera";
  try {
    await openCamera();
    isScanning = true;
    toggleButton.textContent = "Stop scanning";
    scanOnceButton.disabled = false;
    toggleButton.disabled = false;
    setStatus("Camera live · preparing first AI scan", "running");
    stageHint.textContent = "First scan loads the model and may take a little while";
    document.getElementById("model-state").textContent = "Florence-2 base · preparing model; first run downloads the weights";
    void analyzeCurrentFrame();
    scanLoop();
    stream.getVideoTracks()[0].addEventListener("ended", () => {
      if (isScanning) {
        void closeCamera();
        setError("The camera disconnected. Reconnect it and start scanning again.");
      }
    }, { once: true });
  } catch (error) {
    await closeCamera();
    setError(error.message || "The camera could not be started.");
  } finally {
    toggleButton.disabled = false;
  }
}

toggleButton.addEventListener("click", () => {
  if (isScanning) void closeCamera();
  else void startScanning();
});

scanOnceButton.addEventListener("click", async () => {
  clearError();
  if (!stream) {
    scanOnceButton.disabled = true;
    setStatus("Opening camera…", "busy");
    try {
      await openCamera();
      setStatus("Camera ready", "running");
      stageHint.textContent = "Capturing one frame for analysis";
    } catch (error) {
      await closeCamera();
      setError(error.message || "The camera could not be started.");
      return;
    } finally {
      scanOnceButton.disabled = false;
    }
  }
  await analyzeCurrentFrame();
  if (!isScanning) await closeCamera();
});

window.addEventListener("resize", () => {
  if (document.getElementById("object-count").textContent !== "0") {
    drawBoxesFromCanvas();
  }
});

function drawBoxesFromCanvas() {
  if (lastResult) drawBoxes(lastResult);
}

async function refreshModelState() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    const result = await response.json();
    if (result.state === "ready") {
      document.getElementById("model-state").textContent = "Florence-2 base · model ready on this computer";
    } else if (result.state === "loading" || result.state === "analyzing") {
      document.getElementById("model-state").textContent = "Florence-2 base · model is loading or analyzing";
    } else if (result.state === "error") {
      document.getElementById("model-state").textContent = "Florence-2 base · model needs attention; start a scan for details";
    }
  } catch (error) {
    document.getElementById("model-state").textContent = "Local model status is not available";
  }
}

refreshCameras().catch(() => {
  document.getElementById("model-state").textContent = "Camera list will appear after browser permission is granted";
});
void refreshModelState();
