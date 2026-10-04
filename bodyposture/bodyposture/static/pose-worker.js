const TASKS_VERSION = "1.0.1";
const TASKS_MODULE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VERSION}/vision_bundle.mjs`;
const TASKS_WASM = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VERSION}/wasm`;
const POSE_MODEL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

let landmarker = null;
let loaderGeneration = 0;

async function createVisionFileset(FilesetResolver) {
  const fileset = await FilesetResolver.forVisionTasks(TASKS_WASM, true);
  fileset.wasmLoaderPath = `${fileset.wasmLoaderPath}?cb=${Date.now()}-${++loaderGeneration}`;
  return fileset;
}

async function initialize() {
  const { FilesetResolver, PoseLandmarker } = await import(TASKS_MODULE);
  let vision = await createVisionFileset(FilesetResolver);
  const options = {
    baseOptions: { modelAssetPath: POSE_MODEL, delegate: "GPU" },
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: 0.4,
    minPosePresenceConfidence: 0.4,
    minTrackingConfidence: 0.4,
  };

  try {
    landmarker = await PoseLandmarker.createFromOptions(vision, options);
  } catch (gpuError) {
    console.warn("GPU pose delegate unavailable in worker; using CPU.", gpuError);
    vision = await createVisionFileset(FilesetResolver);
    landmarker = await PoseLandmarker.createFromOptions(vision, {
      ...options,
      baseOptions: { modelAssetPath: POSE_MODEL, delegate: "CPU" },
    });
  }
  self.postMessage({ type: "ready" });
}

self.addEventListener("message", async ({ data }) => {
  if (data.type === "initialize") {
    try {
      await initialize();
    } catch (error) {
      self.postMessage({ type: "error", message: error.message || String(error) });
    }
    return;
  }

  if (data.type !== "frame" || !landmarker) {
    data.bitmap?.close();
    return;
  }

  try {
    const result = landmarker.detectForVideo(data.bitmap, data.timestamp);
    const landmarks = result.landmarks?.[0]?.map(({ x, y, visibility }) => ({ x, y, visibility })) ?? null;
    self.postMessage({ type: "result", landmarks, timestamp: data.timestamp, generation: data.generation });
  } catch (error) {
    self.postMessage({ type: "error", message: error.message || String(error) });
  } finally {
    data.bitmap.close();
  }
});
