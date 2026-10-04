const DETECTION_INTERVAL_MS = 80;
const RECORD_INTERVAL_MS = 500;
const CALIBRATION_DURATION_MS = 5000;
const MIN_CALIBRATION_SAMPLES = 8;
const VISIBILITY_MINIMUM = 0.4;
const NECK_TOLERANCE_DEGREES = 12;
const TORSO_TOLERANCE_DEGREES = 10;
const PROFILE_CONNECTIONS = [[7, 11], [11, 23], [8, 12], [12, 24], [11, 12], [23, 24]];

const video = document.getElementById("camera-video");
const overlay = document.getElementById("pose-overlay");
const context = overlay.getContext("2d");
const startButton = document.getElementById("session-button");
const calibrateButton = document.getElementById("calibrate-button");
const exportButton = document.getElementById("export-button");
const cameraSelect = document.getElementById("camera-select");
const liveStatus = document.getElementById("live-status");
const errorBox = document.getElementById("error-message");
const historyCanvas = document.getElementById("history-chart");
const historyContext = historyCanvas.getContext("2d");

let poseWorker = null;
let workerReadyPromise = null;
let stream = null;
let isActive = false;
let inferenceInProgress = false;
let animationHandle = null;
let lastDetectionAt = 0;
let lastValidAt = null;
let lastRecordAt = 0;
let lastUiAt = 0;
let calibrationSamples = [];
let calibrationStartedAt = null;
let calibrationTimer = null;
let baseline = null;
let smoothedAngles = null;
let inferenceGeneration = 0;
let session = freshSession();

function freshSession() {
  return {
    startedAt: null,
    trackedSeconds: 0,
    nearSeconds: 0,
    shiftedSeconds: 0,
    samples: [],
  };
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function setError(message) {
  errorBox.textContent = message;
  errorBox.classList.add("visible");
}

function clearError() {
  errorBox.textContent = "";
  errorBox.classList.remove("visible");
}

function setStatus(label, active = false, loading = false) {
  liveStatus.classList.toggle("is-live", active);
  liveStatus.classList.toggle("is-loading", loading);
  document.getElementById("status-label").textContent = label;
}

function setBusy(busy) {
  startButton.disabled = busy;
  calibrateButton.disabled = busy || !isActive;
  cameraSelect.disabled = busy || isActive;
  exportButton.disabled = busy || session.samples.length === 0;
}

function loadPoseWorker() {
  if (workerReadyPromise) return workerReadyPromise;

  workerReadyPromise = new Promise((resolve, reject) => {
    const workerUrl = new URL("./pose-worker.js", import.meta.url);
    workerUrl.searchParams.set("v", "3");
    poseWorker = new Worker(workerUrl, { type: "module" });
    let initialized = false;
    const timeout = window.setTimeout(() => {
      if (!initialized) reject(new Error("Pose model initialization timed out. Check your internet connection and try again."));
    }, 60000);

    poseWorker.addEventListener("message", ({ data }) => {
      if (data.type === "ready") {
        initialized = true;
        window.clearTimeout(timeout);
        resolve();
        return;
      }
      if (data.type === "error") {
        if (!initialized) {
          window.clearTimeout(timeout);
          reject(new Error(data.message));
        } else {
          inferenceInProgress = false;
          poseWorker?.terminate();
          poseWorker = null;
          workerReadyPromise = null;
          if (isActive) {
            void stopSession();
            setError(`Pose detection stopped: ${data.message}`);
          }
        }
        return;
      }
      if (data.type !== "result") return;
      if (data.generation !== inferenceGeneration) return;
      inferenceInProgress = false;
      if (!isActive) return;
      processResult({ landmarks: data.landmarks ? [data.landmarks] : [] }, data.timestamp);
    });

    poseWorker.addEventListener("error", event => {
      window.clearTimeout(timeout);
      if (!initialized) reject(new Error(event.message || "The background pose worker could not start."));
      else {
        inferenceInProgress = false;
        poseWorker?.terminate();
        poseWorker = null;
        workerReadyPromise = null;
        if (isActive) {
          void stopSession();
          setError(`Pose detection stopped: ${event.message || "the background worker failed."}`);
        }
      }
    });

    poseWorker.postMessage({ type: "initialize" });
  }).catch(error => {
    poseWorker?.terminate();
    poseWorker = null;
    workerReadyPromise = null;
    throw error;
  });
  return workerReadyPromise;
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
    throw new Error("Camera access needs a supported browser and a secure local address. Open this app at http://127.0.0.1:5000 in Chrome or Edge.");
  }

  const selectedDevice = cameraSelect.value;
  const videoOptions = {
    width: { ideal: 640 },
    height: { ideal: 480 },
    frameRate: { ideal: 24, max: 30 },
  };
  if (selectedDevice) videoOptions.deviceId = { exact: selectedDevice };
  stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: videoOptions });
  video.srcObject = stream;
  await video.play();
  document.getElementById("camera-value").textContent = "Connected";
  document.getElementById("camera-placeholder").hidden = true;
  video.classList.add("visible");
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;
  await refreshCameras();
}

async function startSession() {
  clearError();
  setBusy(true);
  setStatus("Connecting camera…", false, true);
  document.getElementById("session-button-label").textContent = "Preparing…";
  try {
    await openCamera();
    setStatus("Loading pose model…", false, true);
    await loadPoseWorker();
    isActive = true;
    inferenceGeneration += 1;
    inferenceInProgress = false;
    session = freshSession();
    session.startedAt = new Date();
    lastDetectionAt = 0;
    lastValidAt = null;
    lastRecordAt = 0;
    baseline = null;
    calibrationSamples = [];
    calibrationStartedAt = null;
    smoothedAngles = null;
    resetDashboard();
    document.getElementById("camera-feedback").textContent = "Live · pose analysis runs in the background";
    document.getElementById("session-summary").hidden = true;
    setStatus("Camera live", true);
    document.getElementById("camera-value").textContent = "Connected";
    document.getElementById("session-button-label").textContent = "End session";
    document.getElementById("session-icon").innerHTML = '<rect x="6" y="6" width="12" height="12" rx="1.5" fill="currentColor"/>';
    calibrateButton.disabled = false;
    startButton.disabled = false;
    cameraSelect.disabled = true;
    stream.getVideoTracks()[0].addEventListener("ended", () => {
      if (isActive) {
        stopSession();
        setError("The webcam was disconnected. Reconnect it and start a new session.");
      }
    }, { once: true });
    drawFrameLoop();
  } catch (error) {
    await closeCamera();
    setStatus("Camera off");
    document.getElementById("session-button-label").textContent = "Start session";
    setBusy(false);
    setError(formatStartError(error));
  }
}

function formatStartError(error) {
  if (error?.name === "NotAllowedError" || error?.name === "PermissionDeniedError") {
    return "Camera permission was blocked. Allow camera access for this site in your browser settings, then try again.";
  }
  if (error?.name === "NotFoundError" || error?.name === "DevicesNotFoundError") {
    return "No webcam was found. Connect a camera and try again.";
  }
  if (error?.name === "NotReadableError" || error?.name === "TrackStartError") {
    return "The webcam is busy in another app. Close that app and try again.";
  }
  return `Could not start posture tracking: ${error?.message || "unknown error"}`;
}

function drawFrameLoop() {
  if (!isActive) return;
  if (typeof video.requestVideoFrameCallback === "function") {
    animationHandle = video.requestVideoFrameCallback(processVideoFrame);
  } else {
    animationHandle = requestAnimationFrame(processVideoFrame);
  }
}

function processVideoFrame(timestamp) {
  if (!isActive || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    drawFrameLoop();
    return;
  }

  if (!inferenceInProgress && timestamp - lastDetectionAt >= DETECTION_INTERVAL_MS) {
    lastDetectionAt = timestamp;
    inferenceInProgress = true;
    createImageBitmap(video).then(bitmap => {
      if (!isActive || !poseWorker) {
        bitmap.close();
        inferenceInProgress = false;
        return;
      }
      poseWorker.postMessage({
        type: "frame",
        bitmap,
        timestamp,
        generation: inferenceGeneration,
      }, [bitmap]);
    }).catch(error => {
      inferenceInProgress = false;
      if (!isActive) return;
      void stopSession();
      setError(`Could not prepare a camera frame for pose detection: ${error.message || error}`);
    });
  }
  drawFrameLoop();
}

function bestSide(landmarks) {
  const choices = [
    { ear: 7, shoulder: 11, hip: 23 },
    { ear: 8, shoulder: 12, hip: 24 },
  ];
  let best = null;
  for (const indices of choices) {
    const points = Object.fromEntries(
      Object.entries(indices).map(([name, index]) => [name, landmarks[index]]),
    );
    const confidence = Math.min(
      points.ear?.visibility ?? 0,
      points.shoulder?.visibility ?? 0,
      points.hip?.visibility ?? 0,
    );
    if (!best || confidence > best.confidence) best = { ...points, confidence };
  }
  return best;
}

function angleFromVertical(upper, lower) {
  return Math.atan2(Math.abs(upper.x - lower.x), Math.abs(upper.y - lower.y)) * 180 / Math.PI;
}

function processResult(result, timestamp) {
  const landmarks = result.landmarks?.[0];
  clearOverlay();
  if (calibrationStartedAt !== null) {
    const remaining = Math.max(0, CALIBRATION_DURATION_MS - (timestamp - calibrationStartedAt));
    calibrateButton.textContent = `Hold steady · ${Math.ceil(remaining / 1000)}s`;
  }
  if (!landmarks) {
    lastValidAt = null;
    document.getElementById("camera-feedback").textContent = "Looking for you · show your head, shoulder & hip";
    document.getElementById("quality-value").textContent = "Low";
    document.getElementById("quality-unit").textContent = "";
    document.getElementById("quality-caption").textContent = "No person detected";
    document.getElementById("overview-title").textContent = "Move into view";
    document.getElementById("overview-copy").textContent = "Keep your head, shoulder and hip in frame.";
    clearCurrentAngles();
    return;
  }

  const side = bestSide(landmarks);
  if (side.confidence < VISIBILITY_MINIMUM) {
    lastValidAt = null;
    document.getElementById("camera-feedback").textContent = "Move side-on · bring ear, shoulder & hip into view";
    document.getElementById("quality-value").textContent = `${Math.round(side.confidence * 100)}`;
    document.getElementById("quality-unit").textContent = "%";
    document.getElementById("quality-caption").textContent = "Landmarks unclear";
    document.getElementById("overview-title").textContent = "Improve camera view";
    document.getElementById("overview-copy").textContent = "Turn side-on and make sure your ear, shoulder and hip are visible.";
    clearCurrentAngles();
    drawLandmarks(landmarks);
    return;
  }

  const cameraFeedback = document.getElementById("camera-feedback");
  if (cameraFeedback.textContent.startsWith("Looking for") || cameraFeedback.textContent.startsWith("Move side-on")) {
    cameraFeedback.textContent = "Pose detected · analysis runs in the background";
  }
  drawLandmarks(landmarks);
  const neckAngle = angleFromVertical(side.ear, side.shoulder);
  const torsoAngle = angleFromVertical(side.shoulder, side.hip);
  if (!smoothedAngles) smoothedAngles = { neck: neckAngle, torso: torsoAngle };
  smoothedAngles.neck += (neckAngle - smoothedAngles.neck) * 0.55;
  smoothedAngles.torso += (torsoAngle - smoothedAngles.torso) * 0.55;

  const neckShift = baseline ? Math.abs(smoothedAngles.neck - baseline.neck) : null;
  const torsoShift = baseline ? Math.abs(smoothedAngles.torso - baseline.torso) : null;
  const aligned = baseline !== null
    && neckShift <= NECK_TOLERANCE_DEGREES
    && torsoShift <= TORSO_TOLERANCE_DEGREES;
  const reading = {
    time: new Date(),
    elapsed: session.startedAt ? (Date.now() - session.startedAt.getTime()) / 1000 : 0,
    neck: Number(smoothedAngles.neck.toFixed(1)),
    torso: Number(smoothedAngles.torso.toFixed(1)),
    neckShift: neckShift == null ? null : Number(neckShift.toFixed(1)),
    torsoShift: torsoShift == null ? null : Number(torsoShift.toFixed(1)),
    confidence: Number(side.confidence.toFixed(3)),
    aligned: baseline === null ? null : aligned,
  };
  if (calibrationStartedAt !== null) {
    calibrationSamples.push({ neck: smoothedAngles.neck, torso: smoothedAngles.torso });
  }

  if (lastValidAt !== null) {
    const seconds = Math.min((timestamp - lastValidAt) / 1000, 0.5);
    if (seconds > 0) {
      session.trackedSeconds += seconds;
      if (baseline) {
        if (aligned) session.nearSeconds += seconds;
        else session.shiftedSeconds += seconds;
      }
    }
  }
  lastValidAt = timestamp;

  if (timestamp - lastRecordAt >= RECORD_INTERVAL_MS) {
    session.samples.push(reading);
    lastRecordAt = timestamp;
    exportButton.disabled = false;
    document.getElementById("sample-count").textContent = session.samples.length.toLocaleString();
    document.getElementById("score-caption").textContent = baseline
      ? `${formatTime(session.nearSeconds)} near · ${formatTime(session.shiftedSeconds)} outside range`
      : `${session.samples.length} readings · calibrate for a personal comparison`;
    document.getElementById("chart-caption").textContent = baseline
      ? "Degrees changed from your calibrated position · latest 90 readings"
      : "Head and torso tilt from vertical · latest 90 readings";
  }
  if (timestamp - lastUiAt >= 160) {
    updateDashboard(reading);
    drawHistory();
    lastUiAt = timestamp;
  }
}

function drawLandmarks(landmarks) {
  context.clearRect(0, 0, overlay.width, overlay.height);
  context.lineWidth = Math.max(2, overlay.width / 320);
  context.lineCap = "round";
  for (const [from, to] of PROFILE_CONNECTIONS) {
    const a = landmarks[from];
    const b = landmarks[to];
    if ((a.visibility ?? 0) < 0.3 || (b.visibility ?? 0) < 0.3) continue;
    context.beginPath();
    context.moveTo(a.x * overlay.width, a.y * overlay.height);
    context.lineTo(b.x * overlay.width, b.y * overlay.height);
    context.strokeStyle = "#75bd8b";
    context.stroke();
  }
  for (const index of [7, 8, 11, 12, 23, 24]) {
    const point = landmarks[index];
    if ((point.visibility ?? 0) < 0.3) continue;
    context.beginPath();
    context.arc(point.x * overlay.width, point.y * overlay.height, Math.max(3, overlay.width / 130), 0, Math.PI * 2);
    context.fillStyle = "#fff";
    context.fill();
    context.beginPath();
    context.arc(point.x * overlay.width, point.y * overlay.height, Math.max(2, overlay.width / 190), 0, Math.PI * 2);
    context.fillStyle = "#477b60";
    context.fill();
  }
}

function clearOverlay() {
  context.clearRect(0, 0, overlay.width, overlay.height);
}

function finishCalibration() {
  if (calibrationTimer !== null) {
    window.clearInterval(calibrationTimer);
    calibrationTimer = null;
  }
  if (calibrationSamples.length < MIN_CALIBRATION_SAMPLES) {
    calibrationStartedAt = null;
    calibrationSamples = [];
    calibrateButton.textContent = "Calibrate · 5 sec";
    calibrateButton.disabled = !isActive;
    setError("Calibration needs a clear side view for most of the 5 seconds. Adjust the camera and try again.");
    return;
  }
  baseline = {
    neck: median(calibrationSamples.map(sample => sample.neck)),
    torso: median(calibrationSamples.map(sample => sample.torso)),
  };
  calibrationStartedAt = null;
  calibrationSamples = [];
  lastValidAt = null;
  session.trackedSeconds = 0;
  session.nearSeconds = 0;
  session.shiftedSeconds = 0;
  session.samples = [];
  lastRecordAt = 0;
  calibrateButton.textContent = "Recalibrate · 5 sec";
  calibrateButton.disabled = false;
  exportButton.disabled = true;
  document.getElementById("sample-count").textContent = "0";
  document.getElementById("score-label").textContent = "Time near your reference";
  document.getElementById("score-caption").textContent = "The share of clearly tracked time within your own range";
  document.getElementById("ring-score-label").textContent = "time near";
  document.getElementById("chart-caption").textContent = "Degrees changed from your calibrated position · latest 90 readings";
  document.getElementById("baseline-badge").textContent = "Set";
  document.getElementById("baseline-badge").classList.add("ready");
  document.getElementById("overview-title").textContent = "Reference saved";
  document.getElementById("overview-copy").textContent = "Readings are now compared with your seated reference.";
  document.getElementById("coach-copy").textContent = "The numbers show change from the position you chose. Recalibrate if you change your camera or seating position.";
  document.getElementById("score-value").textContent = "—";
  document.getElementById("score-unit").textContent = "";
  document.getElementById("ring-score").textContent = "—";
  document.getElementById("score-ring").style.setProperty("--score", "0%");
  drawHistory();
  clearError();
}

function startCalibration() {
  if (!isActive) return;
  clearError();
  if (calibrationTimer !== null) window.clearInterval(calibrationTimer);
  calibrationSamples = [];
  calibrationStartedAt = performance.now();
  calibrateButton.disabled = true;
  calibrateButton.textContent = "Hold steady · 5s";
  document.getElementById("overview-title").textContent = "Calibrating…";
  document.getElementById("overview-copy").textContent = "Sit comfortably, keep still, and remain in the side view.";
  calibrationTimer = window.setInterval(() => {
    const remaining = Math.max(0, CALIBRATION_DURATION_MS - (performance.now() - calibrationStartedAt));
    calibrateButton.textContent = `Hold steady · ${Math.ceil(remaining / 1000)}s`;
    if (remaining <= 0) finishCalibration();
  }, 100);
}

function updateDashboard(reading) {
  const nowNear = baseline && reading.aligned;
  const title = !baseline
    ? "Live measurements"
    : reading.confidence < VISIBILITY_MINIMUM
      ? "Improve camera view"
      : nowNear ? "Near your reference" : "Position has shifted";
  document.getElementById("overview-title").textContent = title;
  document.getElementById("overview-copy").textContent = baseline
    ? `${Math.round(reading.confidence * 100)}% clear view · compared with your comfortable setup`
    : "These angle estimates are not a grade. Calibrate in a comfortable position to get a useful comparison.";
  document.getElementById("quality-value").textContent = `${Math.round(reading.confidence * 100)}`;
  document.getElementById("quality-unit").textContent = "%";
  document.getElementById("quality-caption").textContent = "Ear, shoulder & hip visibility";
  document.getElementById("neck-value").textContent = baseline ? `${reading.neckShift.toFixed(0)}° change` : `${reading.neck.toFixed(0)}° estimate`;
  document.getElementById("torso-value").textContent = baseline ? `${reading.torsoShift.toFixed(0)}° change` : `${reading.torso.toFixed(0)}° estimate`;
  setAngleBar("neck-fill", baseline ? reading.neckShift : reading.neck, baseline ? NECK_TOLERANCE_DEGREES : 45, Boolean(baseline));
  setAngleBar("torso-fill", baseline ? reading.torsoShift : reading.torso, baseline ? TORSO_TOLERANCE_DEGREES : 45, Boolean(baseline));

  document.getElementById("elapsed-value").textContent = formatTime(session.trackedSeconds);
  if (baseline) {
    const percentage = session.trackedSeconds
      ? Math.round(session.nearSeconds / session.trackedSeconds * 100)
      : 0;
    document.getElementById("score-label").textContent = "Time near your reference";
    document.getElementById("score-value").textContent = `${percentage}%`;
    document.getElementById("score-unit").textContent = "";
    document.getElementById("score-caption").textContent =
      `${formatTime(session.nearSeconds)} near · ${formatTime(session.shiftedSeconds)} outside range`;
    document.getElementById("ring-score").textContent = `${percentage}%`;
    document.getElementById("score-ring").style.setProperty("--score", `${percentage}%`);
    document.getElementById("coach-copy").textContent = getLiveCoaching(reading);
  } else {
    document.getElementById("score-label").textContent = "Head angle estimate";
    document.getElementById("score-value").textContent = `${reading.neck.toFixed(0)}°`;
    document.getElementById("score-unit").textContent = "estimate";
    document.getElementById("score-caption").textContent = `Torso estimate ${reading.torso.toFixed(0)}° · set a reference for context`;
    document.getElementById("ring-score").textContent = "SET";
    document.getElementById("score-ring").style.setProperty("--score", "0%");
    document.getElementById("coach-copy").textContent =
      "These are camera angles, not a posture grade. Sit comfortably and calibrate to turn them into a personal change-over-time check.";
  }
}

function getLiveCoaching(reading) {
  if (reading.aligned) {
    return "Your head and torso angles are close to the comfortable position you set. Keep moving naturally; this is a reminder, not a target to hold rigidly.";
  }
  const shifts = [];
  if (reading.neckShift > NECK_TOLERANCE_DEGREES) shifts.push(`head angle differs by ${reading.neckShift.toFixed(0)}°`);
  if (reading.torsoShift > TORSO_TOLERANCE_DEGREES) shifts.push(`torso angle differs by ${reading.torsoShift.toFixed(0)}°`);
  const focus = shifts.length ? shifts.join(" and ") : "your measured angles are just outside the reference range";
  return `Your ${focus}. If it feels comfortable, gently return toward your own setup; do not force a fixed posture.`;
}

function showSessionSummary() {
  const summary = document.getElementById("session-summary");
  const usableSamples = session.samples.filter(sample => Number.isFinite(sample.neck) && Number.isFinite(sample.torso));
  summary.hidden = false;
  document.getElementById("summary-duration").textContent = formatTime(session.trackedSeconds);
  document.getElementById("summary-samples").textContent = usableSamples.length.toLocaleString();

  if (!usableSamples.length) {
    document.getElementById("summary-badge").textContent = "No readings";
    document.getElementById("summary-title").textContent = "No usable posture data";
    document.getElementById("summary-copy").textContent = "The camera did not see your ear, shoulder and hip clearly enough to record readings.";
    document.getElementById("summary-guidance").textContent = "Move back, turn side-on to the camera, improve the lighting, and keep your head, shoulder and hip inside the frame.";
    document.getElementById("summary-score").textContent = "—";
    document.getElementById("summary-score-label").textContent = "no data";
    document.getElementById("summary-ring").style.setProperty("--score", "0%");
    document.getElementById("summary-neck-label").textContent = "Average head angle";
    document.getElementById("summary-torso-label").textContent = "Average torso angle";
    document.getElementById("summary-neck").textContent = "—";
    document.getElementById("summary-torso").textContent = "—";
    return;
  }

  const averageNeck = usableSamples.reduce((sum, sample) => sum + sample.neck, 0) / usableSamples.length;
  const averageTorso = usableSamples.reduce((sum, sample) => sum + sample.torso, 0) / usableSamples.length;
  if (baseline) {
    const averageNeckShift = usableSamples.reduce((sum, sample) => sum + sample.neckShift, 0) / usableSamples.length;
    const averageTorsoShift = usableSamples.reduce((sum, sample) => sum + sample.torsoShift, 0) / usableSamples.length;
    const alignment = session.trackedSeconds
      ? Math.round(session.nearSeconds / session.trackedSeconds * 100)
      : 0;
    document.getElementById("summary-badge").textContent = "Compared to your reference";
    document.getElementById("summary-score-label").textContent = "time near";
    document.getElementById("summary-score").textContent = `${alignment}%`;
    document.getElementById("summary-ring").style.setProperty("--score", `${alignment}%`);
    document.getElementById("summary-neck-label").textContent = "Typical head change";
    document.getElementById("summary-torso-label").textContent = "Typical torso change";
    document.getElementById("summary-neck").textContent = `${averageNeckShift.toFixed(1)}° from reference`;
    document.getElementById("summary-torso").textContent = `${averageTorsoShift.toFixed(1)}° from reference`;
    document.getElementById("summary-title").textContent = alignment >= 80
      ? "You stayed near your setup most of the time"
      : alignment >= 50 ? "Your position varied during the session" : "Your position often differed from setup";
    document.getElementById("summary-copy").textContent =
      `For ${formatTime(session.trackedSeconds)} of clear tracking, you were near your reference for ${formatTime(session.nearSeconds)} and outside its range for ${formatTime(session.shiftedSeconds)}.`;
    document.getElementById("summary-guidance").textContent =
      "Typical change is the average angle difference from your own 5-second setup. Take a movement break if you feel stiff; the camera cannot assess health or prescribe a posture.";
  } else {
    document.getElementById("summary-badge").textContent = "Reference not set";
    document.getElementById("summary-score").textContent = "—";
    document.getElementById("summary-score-label").textContent = "no reference";
    document.getElementById("summary-ring").style.setProperty("--score", "0%");
    document.getElementById("summary-neck-label").textContent = "Average head angle estimate";
    document.getElementById("summary-torso-label").textContent = "Average torso angle estimate";
    document.getElementById("summary-neck").textContent = `${averageNeck.toFixed(1)}°`;
    document.getElementById("summary-torso").textContent = `${averageTorso.toFixed(1)}°`;
    document.getElementById("summary-title").textContent = "Camera readings collected";
    document.getElementById("summary-copy").textContent =
      `The camera tracked clear landmarks for ${formatTime(session.trackedSeconds)} and saved ${usableSamples.length} readings. These angles describe the image, not whether your posture is good or bad.`;
    document.getElementById("summary-guidance").textContent =
      "Next time, calibrate while sitting comfortably. The app can then show how your head and torso angles change from that personal reference.";
  }
}

function setAngleBar(id, value, max, hasReference = true) {
  const bar = document.getElementById(id);
  const amount = Math.min(100, Math.max(0, (value ?? 0) / max * 100));
  bar.style.setProperty("--fill", `${amount}%`);
  bar.classList.toggle("warn", value !== null && value > max);
  bar.classList.toggle("neutral", !hasReference);
}

function clearCurrentAngles() {
  document.getElementById("neck-value").textContent = "—";
  document.getElementById("torso-value").textContent = "—";
  setAngleBar("neck-fill", 0, 45, false);
  setAngleBar("torso-fill", 0, 45, false);
  document.getElementById("ring-score").textContent = "—";
  document.getElementById("score-ring").style.setProperty("--score", "0%");
  document.getElementById("score-value").textContent = "—";
  document.getElementById("score-unit").textContent = "";
  document.getElementById("score-caption").textContent = "Waiting for a clear view of your ear, shoulder and hip";
}

function formatTime(seconds) {
  const total = Math.floor(seconds);
  return `${String(Math.floor(total / 60)).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
}

function drawHistory() {
  const bounds = historyCanvas.getBoundingClientRect();
  if (!bounds.width || !bounds.height) return;
  const ratio = window.devicePixelRatio || 1;
  historyCanvas.width = Math.round(bounds.width * ratio);
  historyCanvas.height = Math.round(bounds.height * ratio);
  historyContext.setTransform(ratio, 0, 0, ratio, 0, 0);
  const width = bounds.width;
  const height = bounds.height;
  historyContext.clearRect(0, 0, width, height);
  historyContext.strokeStyle = "#edf0ec";
  historyContext.lineWidth = 1;
  for (let row = 1; row <= 2; row++) {
    const y = row * height / 3;
    historyContext.beginPath();
    historyContext.moveTo(0, y);
    historyContext.lineTo(width, y);
    historyContext.stroke();
  }
  const data = session.samples.slice(-90);
  if (data.length < 2) return;
  const chartValues = baseline
    ? data.map(sample => ({ neck: sample.neckShift, torso: sample.torsoShift }))
    : data.map(sample => ({ neck: sample.neck, torso: sample.torso }));
  const max = Math.max(20, ...chartValues.flatMap(value => [value.neck, value.torso].filter(Number.isFinite)));
  for (const [key, color] of [["neck", "#477b60"], ["torso", "#b8793d"]]) {
    historyContext.beginPath();
    historyContext.strokeStyle = color;
    historyContext.lineWidth = 2;
    let started = false;
    chartValues.forEach((value, index) => {
      if (!Number.isFinite(value[key])) return;
      const x = index / Math.max(1, chartValues.length - 1) * width;
      const y = height - Math.min(value[key] / max, 1) * height;
      if (!started) {
        historyContext.moveTo(x, y);
        started = true;
      } else historyContext.lineTo(x, y);
    });
    historyContext.stroke();
  }
}

function resetDashboard() {
  document.getElementById("elapsed-value").textContent = "00:00";
  document.getElementById("score-value").textContent = "—";
  document.getElementById("score-unit").textContent = "";
  document.getElementById("score-label").textContent = "Head angle estimate";
  document.getElementById("score-caption").textContent = "Calibrate for a meaningful comparison";
  document.getElementById("chart-caption").textContent = "Degrees from vertical · readings appear when the camera can see you";
  document.getElementById("quality-value").textContent = "—";
  document.getElementById("quality-unit").textContent = "";
  document.getElementById("quality-caption").textContent = "Waiting for camera";
  document.getElementById("ring-score").textContent = "—";
  document.getElementById("score-ring").style.setProperty("--score", "0%");
  document.getElementById("neck-value").textContent = "—";
  document.getElementById("torso-value").textContent = "—";
  setAngleBar("neck-fill", 0, 45, false);
  setAngleBar("torso-fill", 0, 45, false);
  document.getElementById("sample-count").textContent = "0";
  document.getElementById("baseline-badge").textContent = "Not set";
  document.getElementById("baseline-badge").classList.remove("ready");
  document.getElementById("ring-score-label").textContent = "set reference";
  document.getElementById("summary-neck-label").textContent = "Average head angle";
  document.getElementById("summary-torso-label").textContent = "Average torso angle";
  document.getElementById("overview-title").textContent = "Calibrate to begin";
  document.getElementById("overview-copy").textContent = "Sit in your usual comfortable position, side-on to the camera.";
  document.getElementById("coach-copy").textContent = "The angles are estimates, not a posture grade. Calibrate in a position that feels comfortable to see changes from your personal reference.";
  drawHistory();
}

function exportCsv() {
  if (!session.samples.length) return;
  const columns = ["timestamp", "elapsed_seconds", "baseline_head_angle_degrees", "baseline_torso_lean_degrees", "head_angle_degrees", "torso_lean_degrees", "head_change_degrees", "torso_change_degrees", "within_reference_range", "landmark_confidence"];
  const rows = session.samples.map(sample => [
    sample.time.toISOString(),
    sample.elapsed.toFixed(2),
    baseline?.neck.toFixed(1) ?? "",
    baseline?.torso.toFixed(1) ?? "",
    sample.neck.toFixed(1),
    sample.torso.toFixed(1),
    sample.neckShift?.toFixed(1) ?? "",
    sample.torsoShift?.toFixed(1) ?? "",
    sample.aligned == null ? "" : sample.aligned ? "yes" : "no",
    sample.confidence.toFixed(3),
  ].join(","));
  const file = new Blob([[columns.join(","), ...rows].join("\r\n")], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(file);
  const link = document.createElement("a");
  link.href = url;
  link.download = `posture-session-${session.startedAt.toISOString().replace(/[:.]/g, "-")}.csv`;
  link.click();
  URL.revokeObjectURL(url);
}

async function closeCamera() {
  isActive = false;
  if (animationHandle !== null) {
    if (typeof video.cancelVideoFrameCallback === "function") video.cancelVideoFrameCallback(animationHandle);
    else cancelAnimationFrame(animationHandle);
    animationHandle = null;
  }
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }
  video.pause();
  video.srcObject = null;
  video.classList.remove("visible");
  document.getElementById("camera-placeholder").hidden = false;
  clearOverlay();
  document.getElementById("camera-value").textContent = "Not connected";
}

async function stopSession() {
  if (isActive) showSessionSummary();
  inferenceGeneration += 1;
  await closeCamera();
  if (calibrationTimer !== null) {
    window.clearInterval(calibrationTimer);
    calibrationTimer = null;
  }
  calibrationStartedAt = null;
  calibrationSamples = [];
  setStatus("Camera off");
  document.getElementById("session-button-label").textContent = "Start session";
  document.getElementById("session-icon").innerHTML = '<path d="M8 5v14l11-7L8 5Z" fill="currentColor"/>';
  setBusy(false);
}

startButton.addEventListener("click", async () => {
  if (isActive) await stopSession();
  else await startSession();
});
calibrateButton.addEventListener("click", startCalibration);
exportButton.addEventListener("click", exportCsv);
window.addEventListener("resize", drawHistory);
window.addEventListener("pagehide", () => {
  if (stream) stream.getTracks().forEach(track => track.stop());
  poseWorker?.terminate();
});

document.getElementById("today-date").textContent = new Intl.DateTimeFormat(undefined, {
  weekday: "long",
  month: "long",
  day: "numeric",
}).format(new Date());
refreshCameras().catch(error => {
  console.warn("Could not enumerate cameras before permission is granted.", error);
});
drawHistory();
