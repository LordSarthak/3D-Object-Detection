📝 Introduction
As the boundaries between physical and digital worlds blur, the need for realistic 3D representations of real-world objects — especially human faces and bodies — is rapidly increasing. Applications in AR/VR, virtual try-on, gaming, robotics, metaverse platforms, and 3D printing demand accurate and dynamic 3D modeling capabilities.

This project focuses on building a real-time system that converts 2D images (or video frames) into 3D models using Python and modern computer vision and deep learning frameworks. The system is capable of detecting objects or humans in a 2D space and generating 3D representations with high spatial awareness, using depth estimation, keypoint detection, and mesh generation techniques.

🎯 Objective
The primary goal is to develop a robust, scalable, and modular pipeline that takes a single 2D frame — from a camera or image file — and outputs a 3D point cloud or mesh model. The system should support real-time performance for video-based inputs and export standard 3D file formats (.obj, .ply, .glb) for further use in modeling tools or simulation environments.

🏗️ Key Features
✅ Capture 2D input from camera or file

✅ Real-time landmark detection (face, hand, full-body)

✅ Depth estimation from a single 2D image using pre-trained deep models

✅ 3D point cloud generation and mesh reconstruction

✅ Rendering and interactive visualization

✅ Export to standard 3D file formats

✅ Modular architecture for easy updates and model swapping

⚙️ Technical Goals
Efficiently handle high-resolution 2D inputs in real-time

Maintain consistent and smooth 3D reconstruction

Ensure the system is lightweight enough for laptop GPU or mid-range CPU execution

Support future enhancements like animation rigging, texture mapping, and multi-view merging

🔬 Scientific and Technological Background
Traditional 3D reconstruction relies on stereo cameras or LIDAR sensors, which are expensive and complex. Modern deep learning allows 3D modeling from a single 2D image by learning shape priors and depth information from large datasets.

This project leverages:

Monocular depth estimation models like MiDaS and MonoDepth2

Landmark detection frameworks such as MediaPipe, OpenPose, and Dlib

3D human mesh recovery techniques using HMR, SMPL, or PIFu

Mesh and point cloud generation using libraries like Open3D, Trimesh, and MeshLab

This project have all various version of real time images to 3d model conversion and also have posture analysis using computer vision.
tech stack : Yolo, OpenCV, Pillow, Transformers, pyrenderer, Pytorch, Matplotlib and various other LIB.

## Setup: live object and part explorer on Windows

The default page is the camera-based **Object Explorer**. It runs the Florence-2 base checkpoint on the local Flask server and performs general object detection and dense region captioning on camera frames. It attempts to recognize everyday objects and describe visible regions/components; its open-world labels are best-effort guesses, not a guarantee that every object or functional part will be identified. The first scan downloads the model from Hugging Face and can take a while. Processing is local to this computer; camera frames are not saved or sent to a hosted inference API.

The existing webcam **Posture Monitor** remains available at `http://127.0.0.1:5000/posture`. It runs MediaPipe Pose Landmarker Lite in the browser and does not need the Python AI packages used by Object Explorer.

The live result is a personal awareness cue, not a medical or "good posture" grade:

- Before calibration, the head and torso angles are estimates of the camera image and should not be interpreted as a score.
- For a comparison, sit in a position that feels comfortable and choose **Calibrate · 5 sec**. The app then shows how far the detected angles differ from that personal reference and how much clearly tracked time they stayed within the app's comparison range.
- "Near your reference" means the head-angle estimate is within 12° and the torso-angle estimate within 10° of your calibrated readings. These are app comparison tolerances, not health thresholds.
- The session summary reports time near the reference, typical angle changes, tracked time, and usable readings. If calibration is skipped, it explains that only raw camera-angle estimates were collected.
- Pose detection ignores frames without a clear ear, shoulder, and hip. Keep the camera at your side around shoulder height, with those landmarks visible.
- The app offers gentle reminders only; move naturally and stop if a cue is uncomfortable. Camera angles cannot diagnose a condition or prescribe a posture.
- Readings can be exported as CSV during a session. Data stays in the browser and is not stored or sent to the server.

1. Install 64-bit Python 3.14 and confirm the Python launcher sees it:

   ```powershell
   py -0p
   ```

2. From the repository root, create and activate a virtual environment:

   ```powershell
   py -3.14 -m venv .venv-posture
   .\.venv-posture\Scripts\Activate.ps1
   ```

3. Install the local object model and web app dependencies:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r .\bodyposture\requirements.txt
   ```

4. Start the Flask page from the app folder:

   ```powershell
   cd .\bodyposture\bodyposture
   python main.py
   ```

5. Open `http://127.0.0.1:5000` in current Chrome or Edge, allow camera access, and press **Start scanning**. The first analysis downloads and initializes Florence-2. The interface shows common object detections and separate natural-language descriptions of visible image regions. **Scan once** captures one frame; **Stop scanning** stops continuous scans. Expect slower updates on CPU-only laptops.

In VS Code, select `.venv-posture` using **Python: Select Interpreter** (`Ctrl+Shift+P`). If script activation is blocked, use `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again.

Object and component descriptions are generated from 2D frames. The model may miss unfamiliar, small, occluded, or visually ambiguous items, and its region descriptions should not be treated as an exhaustive or technically verified parts list. It does not reconstruct 3D models.

For posture tracking, open `/posture`, start a session, and calibrate in a comfortable seated position. Those 2D camera estimates are personal trends, not medical measurements. Posture samples can be exported while the session is running.
