from __future__ import annotations

import base64
import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import main


def encoded_test_photo() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (24, 16), color=(90, 130, 110)).save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class ReconstructionApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.storage = tempfile.TemporaryDirectory()
        self.scan_root_patch = patch.object(main, "SCAN_ROOT", Path(self.storage.name))
        self.scan_root_patch.start()
        main.app.config.update(TESTING=True)
        self.client = main.app.test_client()

    def tearDown(self) -> None:
        self.scan_root_patch.stop()
        self.storage.cleanup()

    def create_scan(self) -> str:
        response = self.client.post("/api/reconstruction/scans")
        self.assertEqual(response.status_code, 201)
        return response.json["id"]

    def add_photos(self, scan_id: str, count: int) -> None:
        encoded = encoded_test_photo()
        for _ in range(count):
            response = self.client.post(
                f"/api/reconstruction/scans/{scan_id}/photos",
                data={"image": encoded},
            )
            self.assertEqual(response.status_code, 200, response.json)

    def test_capture_is_persisted_until_explicitly_cleared(self) -> None:
        scan_id = self.create_scan()
        self.add_photos(scan_id, 1)

        details = self.client.get(f"/api/reconstruction/scans/{scan_id}")
        self.assertEqual(details.status_code, 200)
        self.assertEqual(details.json["images"], ["photo-01.jpg"])
        self.assertTrue((Path(self.storage.name) / scan_id / "images" / "photo-01.jpg").is_file())

        deleted = self.client.delete(f"/api/reconstruction/scans/{scan_id}")
        self.assertEqual(deleted.status_code, 200)
        self.assertFalse((Path(self.storage.name) / scan_id).exists())

    def test_rejects_invalid_images_and_unknown_scan_ids(self) -> None:
        scan_id = self.create_scan()
        invalid_photo = self.client.post(
            f"/api/reconstruction/scans/{scan_id}/photos",
            data={"image": "not-base64!"},
        )
        self.assertEqual(invalid_photo.status_code, 400)
        self.assertEqual(self.client.get("/api/reconstruction/scans/not-a-uuid").status_code, 404)

    def test_build_requires_minimum_photo_count(self) -> None:
        scan_id = self.create_scan()

        response = self.client.post(
            f"/api/reconstruction/scans/{scan_id}/build",
            json={"longest_dimension_cm": 25},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("at least 8 photos", response.json["error"])

    def test_build_reports_missing_colmap_instead_of_returning_a_fake_model(self) -> None:
        scan_id = self.create_scan()
        self.add_photos(scan_id, main.MIN_SCAN_IMAGES)

        with patch.object(main, "_find_colmap", return_value=None):
            response = self.client.post(
                f"/api/reconstruction/scans/{scan_id}/build",
                json={"longest_dimension_cm": 25},
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("COLMAP is not installed", response.json["error"])
        self.assertEqual(
            self.client.get(f"/api/reconstruction/scans/{scan_id}").json["state"],
            "capturing",
        )

    def test_download_only_serves_the_named_model_formats(self) -> None:
        scan_id = self.create_scan()

        response = self.client.get(
            f"/api/reconstruction/scans/{scan_id}/files/..%2Fscan.json"
        )

        self.assertEqual(response.status_code, 404)

    def test_scaled_exports_use_the_measured_longest_dimension_in_meters(self) -> None:
        import trimesh

        output_directory = Path(self.storage.name)
        source_path = output_directory / "raw.ply"
        trimesh.creation.box(extents=(2, 1, 0.5)).export(source_path)

        details = main._scale_and_export_mesh(source_path, output_directory, 25)

        self.assertEqual(details["mesh_extent_m"], [0.25, 0.125, 0.0625])
        self.assertTrue((output_directory / "model.ply").is_file())
        self.assertTrue((output_directory / "model.obj").is_file())
        exported = trimesh.load_mesh(output_directory / "model.ply", process=False)
        self.assertAlmostEqual(float(max(exported.extents)), 0.25)

    def test_full_build_exports_a_scaled_mesh_after_colmap_stages(self) -> None:
        import trimesh

        scan_id = self.create_scan()
        self.add_photos(scan_id, main.MIN_SCAN_IMAGES)
        calls: list[list[str]] = []

        def fake_colmap(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            if command[1] == "mapper":
                output_path = Path(command[command.index("--output_path") + 1]) / "0"
                output_path.mkdir(parents=True)
                (output_path / "registered.txt").write_text("registered", encoding="utf-8")
            elif command[1] == "poisson_mesher":
                output_path = Path(command[command.index("--output_path") + 1])
                trimesh.creation.box(extents=(2, 1, 0.5)).export(output_path)
            return subprocess.CompletedProcess(command, 0, "", "")

        with (
            patch.object(main, "_find_colmap", return_value="colmap"),
            patch.object(main.subprocess, "run", side_effect=fake_colmap),
        ):
            response = self.client.post(
                f"/api/reconstruction/scans/{scan_id}/build",
                json={"longest_dimension_cm": 25},
            )

        self.assertEqual(response.status_code, 200, response.json)
        self.assertEqual(response.json["state"], "ready")
        self.assertEqual(
            [command[1] for command in calls],
            [
                "feature_extractor",
                "exhaustive_matcher",
                "mapper",
                "image_undistorter",
                "patch_match_stereo",
                "stereo_fusion",
                "poisson_mesher",
            ],
        )
        self.assertAlmostEqual(max(response.json["mesh_extent_m"]), 0.25)
        downloaded = self.client.get(
            f"/api/reconstruction/scans/{scan_id}/files/model.obj"
        )
        self.assertEqual(downloaded.status_code, 200)
        downloaded.close()


if __name__ == "__main__":
    unittest.main()
