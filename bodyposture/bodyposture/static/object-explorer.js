const SCAN_INTERVAL_MS = 500;

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
let selectedObject = null;
let backendReady = false;
let cameraOpening = false;

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
  const scale = Math.min(1, 640 / Math.max(video.videoWidth, video.videoHeight));
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
  stageHint.textContent = "Checking this frame for familiar objects…";

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
    if (
      result.engine !== "YOLOv8n"
      || !Number.isFinite(result.analysis_ms)
      || !Array.isArray(result.objects)
      || result.objects.some(item => !Number.isFinite(item.confidence) || typeof item.insight !== "string")
    ) {
      throw new Error("The app server is outdated. Restart main.py to enable fast, item-specific results.");
    }
    renderResults(result);
    const detectedCount = result.objects.length;
    stageHint.textContent = detectedCount
      ? `${detectedCount} object${detectedCount === 1 ? "" : "s"} recognized in this view`
      : "No familiar object category recognized in this view";
    document.getElementById("model-state").textContent = "YOLOv8n · local detector";
    document.getElementById("analysis-time").textContent = `Analyzed in ${(result.analysis_ms / 1000).toFixed(1)} s`;
  } catch (error) {
    if (error.name !== "AbortError") {
      setError(error.message || "The image could not be analyzed.");
      stageHint.textContent = "Scan paused · check the message below the preview";
      setStatus("Scan issue");
    }
  } finally {
    currentRequest = null;
    requestInProgress = false;
    scanOnceButton.disabled = !backendReady || cameraOpening || requestInProgress;
    if (isScanning && stream) {
      setStatus("Camera live · results update as they arrive", "running");
    }
  }
}

function renderResults(result) {
  lastResult = result;
  document.getElementById("object-count").textContent = result.objects.length;
  renderList("object-list", result.objects, "No familiar object category recognized in this frame. Try a clearer, closer view with the object fully visible.");
  renderStageLabels(result);
  setSelectedObject(result.objects[0] || null);
  drawBoxes(result);
  clearError();
}

function renderStageLabels(result) {
  const labels = document.getElementById("stage-labels");
  labels.replaceChildren();
  result.objects.slice(0, 8).forEach(item => {
    const label = document.createElement("span");
    label.className = "stage-label";
    label.textContent = `${item.label} · ${Math.round(item.confidence * 100)}%`;
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
  items.forEach((item, index) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "label-item object-choice";
    row.dataset.index = String(index);
    row.classList.toggle("selected", item === selectedObject);
    row.setAttribute("aria-pressed", String(item === selectedObject));
    const swatch = document.createElement("span");
    swatch.className = "label-swatch";
    const label = document.createElement("span");
    label.className = "label-text";
    label.textContent = item.label;
    const confidence = document.createElement("span");
    confidence.className = "object-confidence";
    confidence.textContent = `${Math.round(item.confidence * 100)}%`;
    row.append(swatch, label, confidence);
    row.addEventListener("click", () => {
      setSelectedObject(item);
      drawBoxes(lastResult);
    });
    list.append(row);
  });
}

function setSelectedObject(item) {
  selectedObject = item;
  document.querySelectorAll(".object-choice").forEach(row => {
    const isSelected = item !== null
      && lastResult?.objects[Number(row.dataset.index)] === item;
    row.classList.toggle("selected", isSelected);
    row.setAttribute("aria-pressed", String(isSelected));
  });
  document.getElementById("selected-label").textContent = item?.label || "Nothing recognized yet";
  document.getElementById("selected-confidence").textContent = item
    ? `${Math.round(item.confidence * 100)}% match`
    : "—";
  document.getElementById("selected-insight").textContent = item
    ? item.insight
    : "Try moving closer, improving the light, and keeping the whole object visible.";
}

function drawBoxes(result) {
  if (!result) return;
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
  result.objects.forEach(item => {
    const [x1, y1, x2, y2] = item.box;
    const x = offsetX + x1 * scale;
    const y = offsetY + y1 * scale;
    const width = Math.max(0, (x2 - x1) * scale);
    const height = Math.max(0, (y2 - y1) * scale);
    if (width < 1 || height < 1) return;
    const isSelected = item === selectedObject;
    const color = isSelected ? "#bce4c8" : "#75bd8b";
    context.strokeStyle = color;
    context.lineWidth = isSelected ? 3.5 : 2;
    context.strokeRect(x, y, width, height);
    const label = item.label.slice(0, 40);
    context.font = "700 13px Segoe UI, sans-serif";
    const labelWidth = Math.min(bounds.width - 12, context.measureText(label).width + 16);
    const labelX = Math.max(6, Math.min(bounds.width - labelWidth - 6, x));
    const labelTop = y >= 25 ? y - 24 : Math.min(bounds.height - 22, y + 3);
    context.fillStyle = "rgb(20 28 23 / 94%)";
    context.fillRect(labelX, labelTop, labelWidth, 22);
    context.fillStyle = color;
    context.fillRect(labelX, labelTop, 3, 22);
    context.fillStyle = "#fff";
    context.fillText(label, labelX + 9, labelTop + 15, labelWidth - 12);
  });
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
  toggleButton.textContent = "Start live recognition";
  scanOnceButton.disabled = !backendReady;
  stageHint.textContent = lastResult
    ? "Camera off · last analyzed result is still shown"
    : "Start live recognition to identify objects in view";
}

function scanLoop() {
  if (!isScanning) return;
  const now = performance.now();
  if (!requestInProgress && now - lastScanAt >= SCAN_INTERVAL_MS) void analyzeCurrentFrame();
  animationHandle = requestAnimationFrame(scanLoop);
}

async function startScanning() {
  if (!backendReady) {
    setError("Restart main.py to enable the fast local object detector.");
    return;
  }
  clearError();
  cameraOpening = true;
  toggleButton.disabled = true;
  scanOnceButton.disabled = true;
  setStatus("Opening camera…", "busy");
  stageHint.textContent = "Connecting camera";
  try {
    await openCamera();
    isScanning = true;
    toggleButton.textContent = "Stop live recognition";
    scanOnceButton.disabled = false;
    toggleButton.disabled = false;
    setStatus("Camera live · initializing local detector", "running");
    stageHint.textContent = "Loading local detector for the first scan";
    document.getElementById("model-state").textContent = "YOLOv8n · initializing local detector";
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
    cameraOpening = false;
    toggleButton.disabled = !backendReady;
  }
}

toggleButton.addEventListener("click", () => {
  if (isScanning) void closeCamera();
  else void startScanning();
});

scanOnceButton.addEventListener("click", async () => {
  if (!backendReady) {
    setError("Restart main.py to enable the fast local object detector.");
    return;
  }
  clearError();
  if (!stream) {
    scanOnceButton.disabled = true;
    cameraOpening = true;
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
      cameraOpening = false;
      scanOnceButton.disabled = !backendReady;
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
    if (!response.ok) throw new Error(`Detector status returned ${response.status}.`);
    const result = await response.json();
    if (result.engine !== "YOLOv8n") {
      backendReady = false;
      toggleButton.disabled = true;
      scanOnceButton.disabled = true;
      document.getElementById("model-state").textContent = "Restart main.py to activate the fast detector";
      setError("The running app server is outdated. Stop it and restart main.py before scanning.");
      return;
    }
    backendReady = true;
    toggleButton.disabled = cameraOpening;
    scanOnceButton.disabled = cameraOpening || requestInProgress;
    if (result.state === "ready") {
      document.getElementById("model-state").textContent = "YOLOv8n · local detector ready";
    } else if (result.state === "loading" || result.state === "analyzing") {
      document.getElementById("model-state").textContent = "YOLOv8n · preparing the local detector";
    } else if (result.state === "error") {
      document.getElementById("model-state").textContent = "YOLOv8n · detector needs attention";
    }
  } catch {
    backendReady = false;
    cameraOpening = false;
    toggleButton.disabled = true;
    scanOnceButton.disabled = true;
    document.getElementById("model-state").textContent = "Local model status is not available";
  }
}

refreshCameras().catch(() => {
  setError("Camera list is unavailable. Check browser camera permissions and try again.");
});
void refreshModelState();
fetch("/api/detector/warmup", { method: "POST" })
  .then(response => {
    if (!response.ok) throw new Error(`Detector warmup returned ${response.status}.`);
    return response.json();
  })
  .then(() => refreshModelState())
  .catch(() => {
    void refreshModelState();
  });
window.setInterval(() => void refreshModelState(), 1500);
