from __future__ import annotations

import base64
import binascii
import io
import logging
import threading
from typing import Any

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError


MODEL_ID = "florence-community/Florence-2-base"
MAX_IMAGE_BYTES = 6 * 1024 * 1024
MAX_IMAGE_SIDE = 768
OBJECT_TASK = "<OD>"
DETAIL_TASK = "<DENSE_REGION_CAPTION>"

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_BYTES + 256

_model: Any = None
_processor: Any = None
_model_state = "not_loaded"
_model_error: str | None = None
_model_lock = threading.Lock()
_inference_lock = threading.Lock()


def _load_model() -> tuple[Any, Any]:
    global _model, _processor, _model_state, _model_error

    if _model is not None and _processor is not None:
        return _model, _processor

    with _model_lock:
        if _model is not None and _processor is not None:
            return _model, _processor

        _model_state = "loading"
        _model_error = None
        try:
            import torch
            from transformers import AutoProcessor, Florence2ForConditionalGeneration

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            processor = AutoProcessor.from_pretrained(MODEL_ID)
            model = Florence2ForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
            ).to(device)
            model.eval()
        except Exception as error:
            _model_state = "error"
            _model_error = str(error)
            app.logger.exception("Could not load the Florence-2 model.")
            raise RuntimeError(
                "Could not load Florence-2. Check your internet connection and installed packages, then restart the app."
            ) from error

        _model = model
        _processor = processor
        _model_state = "ready"
        return model, processor


def _run_task(
    model: Any,
    processor: Any,
    image: Image.Image,
    task: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    import torch

    inputs = processor(text=task, images=image, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
        )
    decoded = processor.batch_decode(generated, skip_special_tokens=False)[0]
    result = processor.post_process_generation(
        decoded,
        task=task,
        image_size=image.size,
    )
    task_result = result.get(task, {})
    boxes = task_result.get("bboxes", [])
    labels = task_result.get("labels", [])

    entries = []
    for box, label in zip(boxes, labels):
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = (float(value) for value in box)
        entries.append(
            {
                "label": str(label).strip(),
                "box": [
                    max(0, min(image.width, x1)),
                    max(0, min(image.height, y1)),
                    max(0, min(image.width, x2)),
                    max(0, min(image.height, y2)),
                ],
            }
        )
    return {"items": entries}


def analyze_image(image: Image.Image) -> dict[str, Any]:
    scale = min(1.0, MAX_IMAGE_SIDE / max(image.size))
    if scale < 1:
        new_size = (round(image.width * scale), round(image.height * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    global _model_state, _model_error
    with _inference_lock:
        model, processor = _load_model()
        _model_state = "analyzing"
        try:
            objects = _run_task(model, processor, image, OBJECT_TASK, 160)
            details = _run_task(model, processor, image, DETAIL_TASK, 320)
        except Exception as error:
            _model_state = "error"
            _model_error = str(error)
            app.logger.exception("Florence-2 image analysis failed.")
            raise RuntimeError(
                "The vision model could not analyze this frame. Try a clearer image with better lighting."
            ) from error
        _model_state = "ready"

    return {
        "width": image.width,
        "height": image.height,
        "objects": objects["items"],
        "parts": details["items"],
    }


@app.get("/")
def object_explorer():
    return render_template("object_explorer.html")


@app.get("/posture")
def posture_monitor():
    return render_template("index.html")


@app.get("/api/status")
def model_status():
    return jsonify({"state": _model_state})


@app.post("/api/analyze")
def analyze_frame():
    encoded = request.form.get("image", "")
    if not encoded:
        return jsonify({"error": "The camera frame was empty. Restart the camera and try again."}), 400

    try:
        image_bytes = base64.b64decode(encoded, validate=True)
        if len(image_bytes) > MAX_IMAGE_BYTES:
            return jsonify({"error": "This camera frame is too large. Lower the camera resolution and try again."}), 413
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = source.convert("RGB")
    except (binascii.Error, UnidentifiedImageError, OSError, ValueError):
        return jsonify({"error": "The camera frame could not be read as an image."}), 400

    try:
        result = analyze_image(image)
    except RuntimeError as error:
        return jsonify({"error": str(error), "state": _model_state}), 503
    return jsonify(result)


@app.errorhandler(413)
def request_too_large(_error: Exception):
    return jsonify({"error": "The camera frame is too large to analyze. Try again with a lower camera resolution."}), 413


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
