const MINIMUM_PHOTOS = 8;
const STORAGE_KEY = "local-vision-studio.scan-id";

const video = document.getElementById("capture-video");
const cameraButton = document.getElementById("camera-button");
const captureButton = document.getElementById("capture-button");
const fileInput = document.getElementById("photo-files");
const photoCount = document.getElementById("photo-count");
const buildButton = document.getElementById("build-button");
const identifyButton = document.getElementById("identify-button");
const errorBox = document.getElementById("error-message");
const progress = document.getElementById("build-progress");
const dimensionInput = document.getElementById("dimension-input");

let stream = null;
let scanId = localStorage.getItem(STORAGE_KEY);
let photoTotal = 0;
let toolsReady = false;
let latestPhoto = null;
let building = false;

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.add("visible");
}

function clearError() {
  errorBox.textContent = "";
  errorBox.classList.remove("visible");
}

async function readResponse(response) {
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `Request failed (${response.status}).`);
  return result;
}

function updateControls() {
  document.getElementById("photo-count").textContent = `${photoTotal} photo${photoTotal === 1 ? "" : "s"}`;
  captureButton.disabled = !stream || photoTotal >= 30 || building;
  fileInput.disabled = photoTotal >= 30 || building;
  document.getElementById("capture-hint").textContent = photoTotal < MINIMUM_PHOTOS
    ? `Add ${MINIMUM_PHOTOS - photoTotal} more photo${MINIMUM_PHOTOS - photoTotal === 1 ? "" : "s"} to meet the minimum (up to 30). Cover a full circle with overlapping views.`
    : `${photoTotal} views captured. Add more angles if part of the object is hidden; the limit is 30.`;
  buildButton.disabled = !scanId || photoTotal < MINIMUM_PHOTOS || !toolsReady || building;
  identifyButton.disabled = !latestPhoto || building;
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("Camera access requires a supported browser. Open this app at http://127.0.0.1:5000.");
  }
  stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } },
  });
  video.srcObject = stream;
  await video.play();
  await new Promise(resolve => {
    if (video.videoWidth) resolve();
    else video.addEventListener("loadedmetadata", resolve, { once: true });
  });
  video.classList.add("visible");
  document.getElementById("camera-empty").hidden = true;
  cameraButton.textContent = "Stop camera";
  updateControls();
}

function stopCamera() {
  stream?.getTracks().forEach(track => track.stop());
  stream = null;
  video.pause();
  video.srcObject = null;
  video.classList.remove("visible");
  document.getElementById("camera-empty").hidden = false;
  captureButton.disabled = true;
  cameraButton.textContent = "Start camera";
}

async function ensureScan() {
  if (scanId) {
    const response = await fetch(`/api/reconstruction/scans/${scanId}`);
    if (response.ok) {
      await response.json();
      return scanId;
    }
    if (response.status === 404) {
      scanId = null;
      localStorage.removeItem(STORAGE_KEY);
      photoTotal = 0;
    } else {
      await readResponse(response);
    }
  }
  const scan = await readResponse(await fetch("/api/reconstruction/scans", { method: "POST" }));
  scanId = scan.id;
  localStorage.setItem(STORAGE_KEY, scanId);
  return scanId;
}

function imageDataFromCanvas(canvas) {
  return canvas.toDataURL("image/jpeg", 0.88);
}

async function submitPhoto(dataUrl) {
  clearError();
  const currentScan = await ensureScan();
  const form = new FormData();
  form.set("image", dataUrl.slice(dataUrl.indexOf(",") + 1));
  const result = await readResponse(await fetch(`/api/reconstruction/scans/${currentScan}/photos`, {
    method: "POST",
    body: form,
  }));
  photoTotal = result.photos;
  latestPhoto = dataUrl;
  document.getElementById("build-progress").textContent = `${result.filename} saved on this computer.`;
  updateControls();
}

cameraButton.addEventListener("click", async () => {
  clearError();
  if (stream) {
    stopCamera();
    return;
  }
  cameraButton.disabled = true;
  try {
    await startCamera();
  } catch (error) {
    showError(error.message || "Could not open the camera.");
  } finally {
    cameraButton.disabled = false;
  }
});

captureButton.addEventListener("click", async () => {
  if (!stream || !video.videoWidth) return;
  captureButton.disabled = true;
  try {
    const scale = Math.min(1, 1600 / Math.max(video.videoWidth, video.videoHeight));
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(video.videoWidth * scale);
    canvas.height = Math.round(video.videoHeight * scale);
    canvas.getContext("2d", { alpha: false }).drawImage(video, 0, 0, canvas.width, canvas.height);
    await submitPhoto(imageDataFromCanvas(canvas));
  } catch (error) {
    showError(error.message || "Could not save that photo.");
  } finally {
    updateControls();
  }
});

async function fileToDataUrl(file) {
  const image = new Image();
  const objectUrl = URL.createObjectURL(file);
  try {
    image.src = objectUrl;
    await image.decode();
    const scale = Math.min(1, 1600 / Math.max(image.naturalWidth, image.naturalHeight));
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(image.naturalWidth * scale);
    canvas.height = Math.round(image.naturalHeight * scale);
    canvas.getContext("2d", { alpha: false }).drawImage(image, 0, 0, canvas.width, canvas.height);
    return imageDataFromCanvas(canvas);
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

fileInput.addEventListener("change", async () => {
  clearError();
  const files = [...fileInput.files];
  fileInput.value = "";
  for (const file of files) {
    if (photoTotal >= 30) {
      showError("This scan already has the maximum of 30 photos.");
      break;
    }
    try {
      await submitPhoto(await fileToDataUrl(file));
    } catch (error) {
      showError(error.message || `Could not add ${file.name}.`);
      break;
    }
  }
});

identifyButton.addEventListener("click", async () => {
  if (!latestPhoto) return;
  clearError();
  identifyButton.disabled = true;
  const form = new FormData();
  form.set("image", latestPhoto.slice(latestPhoto.indexOf(",") + 1));
  try {
    const result = await readResponse(await fetch("/api/analyze", { method: "POST", body: form }));
    const panel = document.getElementById("identification");
    panel.replaceChildren();
    const objects = result.objects.map(item => item.label);
    const descriptions = result.parts.map(item => item.label);
    if (!objects.length && !descriptions.length) {
      panel.textContent = "No object labels or region descriptions were returned for that view.";
    } else {
      if (objects.length) {
        const title = document.createElement("strong");
        title.textContent = "Recognized objects";
        const list = document.createElement("ul");
        list.className = "labels";
        objects.slice(0, 12).forEach(label => {
          const row = document.createElement("li");
          row.textContent = label;
          list.append(row);
        });
        panel.append(title, list);
      }
      if (descriptions.length) {
        const title = document.createElement("p");
        title.textContent = "Visible region descriptions (AI-generated, may be inaccurate)";
        const list = document.createElement("ul");
        list.className = "labels";
        descriptions.slice(0, 8).forEach(label => {
          const row = document.createElement("li");
          row.textContent = label;
          list.append(row);
        });
        panel.append(title, list);
      }
    }
  } catch (error) {
    showError(error.message || "The photo could not be identified.");
  } finally {
    updateControls();
  }
});

async function refreshScan() {
  if (!scanId) {
    photoTotal = 0;
    updateControls();
    return;
  }
  try {
    const scan = await readResponse(await fetch(`/api/reconstruction/scans/${scanId}`, { cache: "no-store" }));
    photoTotal = scan.images.length;
    if (scan.state === "ready") renderModel(scan);
    else if (scan.state === "building") {
      building = true;
      progress.textContent = `${scan.stage}…`;
    } else if (scan.state === "error" && scan.error) {
      progress.textContent = "";
      showError(scan.error);
    }
  } catch {
    scanId = null;
    photoTotal = 0;
    localStorage.removeItem(STORAGE_KEY);
  }
  updateControls();
}

function renderModel(scan) {
  progress.textContent = "Reconstruction complete.";
  const results = document.getElementById("results");
  results.replaceChildren();
  const stats = document.createElement("p");
  stats.className = "result-stats";
  const [x, y, z] = scan.mesh_extent_m;
  stats.textContent = `${scan.vertex_count.toLocaleString()} vertices · ${scan.face_count.toLocaleString()} faces · scaled extents ${x} × ${y} × ${z} m`;
  results.append(stats);
  for (const [filename, label] of [["model.obj", "Download OBJ"], ["model.ply", "Download PLY"]]) {
    const link = document.createElement("a");
    link.className = "result-link";
    link.href = `/api/reconstruction/scans/${scan.id}/files/${filename}`;
    link.download = filename;
    link.textContent = label;
    results.append(link);
  }
}

async function pollProgress() {
  if (!scanId || !building) return;
  try {
    const scan = await readResponse(await fetch(`/api/reconstruction/scans/${scanId}`, { cache: "no-store" }));
    if (scan.state === "building") {
      progress.textContent = `${scan.stage}… This may take several minutes.`;
    } else {
      building = false;
      if (scan.state === "ready") renderModel(scan);
      else if (scan.error) showError(scan.error);
      updateControls();
    }
  } catch (error) {
    building = false;
    showError(error.message || "Could not read the reconstruction status.");
    updateControls();
  }
}

buildButton.addEventListener("click", async () => {
  clearError();
  const dimension = Number(dimensionInput.value);
  if (!Number.isFinite(dimension) || dimension <= 0) {
    showError("Enter the measured longest dimension in centimeters.");
    dimensionInput.focus();
    return;
  }
  building = true;
  progress.textContent = "Starting reconstruction…";
  updateControls();
  const timer = window.setInterval(() => void pollProgress(), 1800);
  try {
    const result = await readResponse(await fetch(`/api/reconstruction/scans/${scanId}/build`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ longest_dimension_cm: dimension }),
    }));
    building = false;
    renderModel(result);
  } catch (error) {
    building = false;
    progress.textContent = "";
    showError(error.message || "The 3D model could not be built.");
  } finally {
    window.clearInterval(timer);
    updateControls();
  }
});

document.getElementById("new-scan-button").addEventListener("click", async () => {
  clearError();
  stopCamera();
  if (scanId) {
    try {
      await readResponse(await fetch(`/api/reconstruction/scans/${scanId}`, { cache: "no-store" }));
      await readResponse(await fetch(`/api/reconstruction/scans/${scanId}`, { method: "DELETE" }));
    } catch (error) {
      showError(error.message || "The saved scan could not be cleared.");
      return;
    }
  }
  scanId = null;
  photoTotal = 0;
  latestPhoto = null;
  localStorage.removeItem(STORAGE_KEY);
  document.getElementById("results").replaceChildren();
  document.getElementById("identification").replaceChildren();
  progress.textContent = "";
  updateControls();
});

async function checkTools() {
  try {
    const result = await readResponse(await fetch("/api/reconstruction/status", { cache: "no-store" }));
    const notice = document.getElementById("dependency-status");
    if (result.colmap_available && result.mesh_scaling_available) {
      toolsReady = true;
      notice.textContent = `Ready · COLMAP found at ${result.colmap_path}. Rebuild requires 8–30 overlapping photos.`;
    } else {
      toolsReady = false;
      const missing = [];
      if (!result.colmap_available) {
        missing.push(result.colmap_error || "COLMAP CLI (install it and set COLMAP_EXE to colmap.exe)");
      }
      if (!result.mesh_scaling_available) missing.push("trimesh (install the app requirements)");
      notice.textContent = `Reconstruction is not ready: ${missing.join("; ")}. Photos and object identification still work.`;
    }
  } catch (error) {
    document.getElementById("dependency-status").textContent = error.message || "Could not check reconstruction tools.";
  }
  updateControls();
}

window.addEventListener("beforeunload", stopCamera);
void Promise.all([checkTools(), refreshScan()]);
window.setInterval(() => void pollProgress(), 2000);
