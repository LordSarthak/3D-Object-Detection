from __future__ import annotations

import base64
import binascii
import importlib.util
import io
import logging
import math
import os
import re
import shutil
import subprocess
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image, UnidentifiedImageError


MODEL_ID = "florence-community/Florence-2-base"
MAX_IMAGE_BYTES = 6 * 1024 * 1024
MAX_IMAGE_SIDE = 768
MAX_SCAN_IMAGES = 30
MIN_SCAN_IMAGES = 8
MAX_CAPTURE_BYTES = 8 * 1024 * 1024
OBJECT_TASK = "<OD>"
DETAIL_TASK = "<DENSE_REGION_CAPTION>"
SCAN_ROOT = Path(__file__).resolve().parent / "scan_sessions"
ASSET_NAMES = {"model.ply", "model.obj"}

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["MAX_CONTENT_LENGTH"] = (MAX_CAPTURE_BYTES * 4 // 3) + 8192

_model: Any = None
_processor: Any = None
_model_state = "not_loaded"
_model_error: str | None = None
_model_lock = threading.Lock()
_inference_lock = threading.Lock()
_reconstruction_lock = threading.Lock()
_active_reconstruction_ids: set[str] = set()
_active_scans_lock = threading.Lock()


def _scan_path(scan_id: str) -> Path:
    try:
        parsed_id = uuid.UUID(scan_id)
    except (ValueError, AttributeError) as error:
        raise ValueError("Invalid scan ID.") from error
    if str(parsed_id) != scan_id:
        raise ValueError("Invalid scan ID.")
    return SCAN_ROOT / scan_id


def _read_scan(scan_id: str) -> tuple[Path, dict[str, Any]]:
    scan_path = _scan_path(scan_id)
    metadata_path = scan_path / "scan.json"
    if not metadata_path.is_file():
        raise FileNotFoundError("This scan does not exist. Start a new capture.")
    import json

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return scan_path, metadata


def _write_scan(scan_path: Path, metadata: dict[str, Any]) -> None:
    import json

    metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
    temporary_path = scan_path / "scan.json.tmp"
    temporary_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(scan_path / "scan.json")


def _find_colmap() -> str | None:
    configured = os.environ.get("COLMAP_EXE")
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file():
            return str(candidate)
        raise RuntimeError(
            "COLMAP_EXE is set but does not point to a file. Set it to the full path of colmap.exe."
        )
    return shutil.which("colmap")


def _scale_and_export_mesh(
    source_path: Path,
    output_directory: Path,
    longest_dimension_cm: float,
) -> dict[str, Any]:
    import trimesh

    mesh = trimesh.load_mesh(source_path, process=False, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        raise RuntimeError("The reconstruction did not contain a usable triangle mesh.")
    measured_extent = float(max(mesh.extents))
    if not math.isfinite(measured_extent) or measured_extent <= 0:
        raise RuntimeError("The reconstructed mesh has no measurable dimensions.")
    mesh.apply_scale((longest_dimension_cm / 100) / measured_extent)
    mesh.export(output_directory / "model.ply")
    mesh.export(output_directory / "model.obj")
    return {
        "mesh_extent_m": [round(float(value), 5) for value in mesh.extents],
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
    }


def _build_reconstruction(scan_id: str, longest_dimension_cm: float) -> dict[str, Any]:
    scan_path, metadata = _read_scan(scan_id)
    image_paths = sorted((scan_path / "images").glob("*.jpg"))
    if len(image_paths) < MIN_SCAN_IMAGES:
        raise ValueError(f"Capture at least {MIN_SCAN_IMAGES} photos before building a model.")
    if len(image_paths) > MAX_SCAN_IMAGES:
        raise ValueError(f"A scan can contain no more than {MAX_SCAN_IMAGES} photos.")
    if not math.isfinite(longest_dimension_cm) or not 0.1 <= longest_dimension_cm <= 10000:
        raise ValueError("Enter the object's longest real-world dimension in centimeters (0.1–10,000).")

    colmap = _find_colmap()
    if not colmap:
        raise RuntimeError(
            "COLMAP is not installed or is not on PATH. Install the Windows COLMAP command-line application, "
            "then set COLMAP_EXE to the full path of colmap.exe and restart the app."
        )
    if importlib.util.find_spec("trimesh") is None:
        raise RuntimeError(
            "The mesh scaling package is missing. Install the app dependencies with "
            "'python -m pip install -r ..\\requirements.txt' and restart."
        )

    work = scan_path / "reconstruction"
    database = work / "database.db"
    sparse = work / "sparse"
    dense = work / "dense"
    if work.exists():
        shutil.rmtree(work)
    dense.mkdir(parents=True)
    sparse.mkdir(parents=True)
    commands = [
        ("Matching image features", [
            colmap, "feature_extractor", "--database_path", str(database),
            "--image_path", str(scan_path / "images"),
        ]),
        ("Matching overlapping views", [
            colmap, "exhaustive_matcher", "--database_path", str(database),
        ]),
        ("Estimating camera positions", [
            colmap, "mapper", "--database_path", str(database),
            "--image_path", str(scan_path / "images"), "--output_path", str(sparse),
        ]),
    ]

    metadata.update({"state": "building", "stage": commands[0][0], "error": None})
    _write_scan(scan_path, metadata)
    try:
        for stage, command in commands:
            metadata["stage"] = stage
            _write_scan(scan_path, metadata)
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=30 * 60,
            )

        model_dir = sparse / "0"
        if not model_dir.is_dir() or not any(model_dir.iterdir()):
            raise RuntimeError(
                "COLMAP could not align the photos. Retake them with more overlap, a textured background, "
                "and no movement of the object or camera during each shot."
            )

        later_commands = [
            ("Preparing multi-view depth", [
                colmap, "image_undistorter", "--image_path", str(scan_path / "images"),
                "--input_path", str(model_dir), "--output_path", str(dense),
                "--output_type", "COLMAP",
            ]),
            ("Fusing the captured views", [
                colmap, "patch_match_stereo", "--workspace_path", str(dense),
                "--workspace_format", "COLMAP",
            ]),
            ("Building the surface", [
                colmap, "stereo_fusion", "--workspace_path", str(dense),
                "--workspace_format", "COLMAP", "--input_type", "geometric",
                "--output_path", str(dense / "fused.ply"),
            ]),
            ("Creating the mesh", [
                colmap, "poisson_mesher", "--input_path", str(dense / "fused.ply"),
                "--output_path", str(dense / "surface.ply"),
            ]),
        ]
        for stage, command in later_commands:
            metadata["stage"] = stage
            _write_scan(scan_path, metadata)
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=60 * 60,
            )

        source_mesh = dense / "surface.ply"
        if not source_mesh.is_file() or source_mesh.stat().st_size == 0:
            raise RuntimeError("COLMAP finished without producing a surface mesh.")
        metadata["stage"] = "Applying the real-world scale"
        _write_scan(scan_path, metadata)
        model_details = _scale_and_export_mesh(
            source_mesh,
            scan_path,
            longest_dimension_cm,
        )

        metadata.update({
            "state": "ready",
            "stage": "Model ready",
            "error": None,
            "longest_dimension_cm": longest_dimension_cm,
            **model_details,
        })
        _write_scan(scan_path, metadata)
        return metadata
    except subprocess.TimeoutExpired as error:
        message = f"COLMAP timed out during {metadata.get('stage', 'reconstruction')}. Try fewer photos or a faster computer."
        metadata.update({"state": "error", "error": message})
        _write_scan(scan_path, metadata)
        raise RuntimeError(message) from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip()
        detail = re.sub(r"\s+", " ", detail)[-800:]
        message = f"COLMAP failed during {metadata.get('stage', 'reconstruction')}."
        if detail:
            message += f" {detail}"
        metadata.update({"state": "error", "error": message})
        _write_scan(scan_path, metadata)
        raise RuntimeError(message) from error
    except Exception as error:
        metadata.update({"state": "error", "error": str(error)})
        _write_scan(scan_path, metadata)
        raise


def _decode_capture(encoded: str) -> Image.Image:
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
        if len(image_bytes) > MAX_CAPTURE_BYTES:
            raise OverflowError("Each photo must be smaller than 8 MB.")
        with Image.open(io.BytesIO(image_bytes)) as source:
            if source.width * source.height > 40_000_000:
                raise OverflowError("This photo's image dimensions are too large to process.")
            image = source.convert("RGB")
            image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
            return image
    except OverflowError as error:
        raise ValueError(str(error)) from error
    except (
        binascii.Error,
        Image.DecompressionBombError,
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as error:
        raise ValueError("The captured photo could not be read. Try taking it again.") from error


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


@app.get("/reconstruction")
def reconstruction_page():
    return render_template("reconstruction.html")


@app.get("/api/status")
def model_status():
    return jsonify({"state": _model_state})


@app.get("/api/reconstruction/status")
def reconstruction_status():
    try:
        colmap_path = _find_colmap()
        colmap_error = None
    except RuntimeError as error:
        colmap_path = None
        colmap_error = str(error)
    try:
        trimesh_available = importlib.util.find_spec("trimesh") is not None
    except (ImportError, ValueError):
        trimesh_available = False
    return jsonify({
        "colmap_available": colmap_path is not None,
        "colmap_path": colmap_path,
        "colmap_error": colmap_error,
        "mesh_scaling_available": trimesh_available,
        "minimum_photos": MIN_SCAN_IMAGES,
        "maximum_photos": MAX_SCAN_IMAGES,
    })


@app.post("/api/reconstruction/scans")
def create_scan():
    SCAN_ROOT.mkdir(parents=True, exist_ok=True)
    scan_id = str(uuid.uuid4())
    scan_path = _scan_path(scan_id)
    (scan_path / "images").mkdir(parents=True)
    metadata: dict[str, Any] = {
        "id": scan_id,
        "state": "capturing",
        "stage": "Capture photos around the object",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "images": [],
    }
    _write_scan(scan_path, metadata)
    return jsonify(metadata), 201


@app.post("/api/reconstruction/scans/<scan_id>/photos")
def add_scan_photo(scan_id: str):
    try:
        scan_path, metadata = _read_scan(scan_id)
        if metadata.get("state") != "capturing":
            return jsonify({"error": "Photos can only be added while a scan is being captured."}), 409
        images = metadata.get("images", [])
        if len(images) >= MAX_SCAN_IMAGES:
            return jsonify({"error": f"This scan already has the maximum of {MAX_SCAN_IMAGES} photos."}), 409
        encoded = request.form.get("image", "")
        if not encoded:
            return jsonify({"error": "No photo was received."}), 400
        image = _decode_capture(encoded)
        filename = f"photo-{len(images) + 1:02d}.jpg"
        image.save(scan_path / "images" / filename, format="JPEG", quality=90, optimize=True)
        images.append(filename)
        metadata["images"] = images
        _write_scan(scan_path, metadata)
        return jsonify({"id": scan_id, "photos": len(images), "filename": filename})
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 404


@app.get("/api/reconstruction/scans/<scan_id>")
def scan_details(scan_id: str):
    try:
        _, metadata = _read_scan(scan_id)
        return jsonify(metadata)
    except (ValueError, FileNotFoundError) as error:
        return jsonify({"error": str(error)}), 404


@app.post("/api/reconstruction/scans/<scan_id>/build")
def build_scan(scan_id: str):
    payload = request.get_json(silent=True) or {}
    try:
        _read_scan(scan_id)
    except (ValueError, FileNotFoundError) as error:
        return jsonify({"error": str(error)}), 404
    try:
        longest_dimension_cm = float(payload.get("longest_dimension_cm", ""))
    except (TypeError, ValueError) as error:
        return jsonify({"error": "Enter the object's longest real-world dimension in centimeters."}), 400

    with _active_scans_lock:
        if scan_id in _active_reconstruction_ids:
            return jsonify({"error": "This scan is already being reconstructed."}), 409
        _active_reconstruction_ids.add(scan_id)
    try:
        with _reconstruction_lock:
            metadata = _build_reconstruction(scan_id, longest_dimension_cm)
        return jsonify(metadata)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
        app.logger.exception("3D reconstruction failed for scan %s.", scan_id)
        return jsonify({"error": str(error)}), 503
    finally:
        with _active_scans_lock:
            _active_reconstruction_ids.discard(scan_id)


@app.get("/api/reconstruction/scans/<scan_id>/files/<asset_name>")
def download_scan_asset(scan_id: str, asset_name: str):
    if asset_name not in ASSET_NAMES:
        return jsonify({"error": "That file is not available for download."}), 404
    try:
        scan_path, _ = _read_scan(scan_id)
    except (ValueError, FileNotFoundError) as error:
        return jsonify({"error": str(error)}), 404
    asset_path = scan_path / asset_name
    if not asset_path.is_file():
        return jsonify({"error": "Build the 3D model before downloading it."}), 404
    return send_file(asset_path, as_attachment=True, download_name=asset_name)


@app.delete("/api/reconstruction/scans/<scan_id>")
def delete_scan(scan_id: str):
    try:
        scan_path, _ = _read_scan(scan_id)
    except (ValueError, FileNotFoundError) as error:
        return jsonify({"error": str(error)}), 404
    with _active_scans_lock:
        if scan_id in _active_reconstruction_ids:
            return jsonify({"error": "Stop the current reconstruction before deleting this scan."}), 409
    shutil.rmtree(scan_path)
    return jsonify({"deleted": True})


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
    return jsonify({"error": "The image is too large to process. Use a lower camera resolution or a smaller photo."}), 413


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
