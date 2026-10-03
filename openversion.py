import os
import subprocess
from PIL import Image
import open3d as o3d

# Set up paths
OPENMVG_BIN = r"C:\openmvg\openMVG_Build\Windows-AMD64-Release"  # Update to your OpenMVG bin folder
OPENMVS_BIN = r"C:\openmvs\bin"  # Update to your OpenMVS bin folder

input_folder = r"d:\PYTHON PROJECT\foot_images"
preprocessed_folder = r"d:\PYTHON PROJECT\preprocessed_images"
matches_folder = r"d:\PYTHON PROJECT\matches"
reconstruction_folder = r"d:\PYTHON PROJECT\reconstruction_sequential"
mvs_folder = r"d:\PYTHON PROJECT\mvs"
output_mesh = os.path.join(mvs_folder, "scene_dense_mesh_texture.ply")

os.makedirs(preprocessed_folder, exist_ok=True)
os.makedirs(matches_folder, exist_ok=True)
os.makedirs(reconstruction_folder, exist_ok=True)
os.makedirs(mvs_folder, exist_ok=True)

# 1. Preprocess images
for fname in os.listdir(input_folder):
    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(input_folder, fname)
        img = Image.open(img_path).convert("RGB")
        img.thumbnail((1024, 1024))
        img.save(os.path.join(preprocessed_folder, fname))

# 2. OpenMVG pipeline
def run(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

# 2.1 Intrinsics analysis
run([
    os.path.join(OPENMVG_BIN, "openMVG_main_SfMInit_ImageListing"),
    "-i", preprocessed_folder,
    "-o", matches_folder,
    "-d", os.path.join(OPENMVG_BIN, "sensor_width_camera_database.txt"),
    "-c", "3"  # 3 = upright camera
])

# 2.2 Compute features
run([
    os.path.join(OPENMVG_BIN, "openMVG_main_ComputeFeatures"),
    "-i", os.path.join(matches_folder, "sfm_data.json"),
    "-o", matches_folder
])

# 2.3 Compute matches
run([
    os.path.join(OPENMVG_BIN, "openMVG_main_ComputeMatches"),
    "-i", os.path.join(matches_folder, "sfm_data.json"),
    "-o", matches_folder
])

# 2.4 Incremental reconstruction
run([
    os.path.join(OPENMVG_BIN, "openMVG_main_IncrementalSfM"),
    "-i", os.path.join(matches_folder, "sfm_data.json"),
    "-m", matches_folder,
    "-o", reconstruction_folder
])

# 2.5 Structure from Known Poses (robust triangulation)
run([
    os.path.join(OPENMVG_BIN, "openMVG_main_ComputeStructureFromKnownPoses"),
    "-i", os.path.join(reconstruction_folder, "sfm_data.bin"),
    "-m", matches_folder,
    "-f", os.path.join(matches_folder, "matches.f.bin"),
    "-o", os.path.join(reconstruction_folder, "robust.ply")
])

# 3. OpenMVS pipeline
# 3.1 Convert OpenMVG to OpenMVS format
run([
    os.path.join(OPENMVG_BIN, "openMVG_main_openMVG2openMVS"),
    "-i", os.path.join(reconstruction_folder, "sfm_data.bin"),
    "-d", os.path.join(reconstruction_folder, "robust.ply"),
    "-o", os.path.join(mvs_folder, "scene.mvs")
])

# 3.2 Densify point cloud
run([
    os.path.join(OPENMVS_BIN, "DensifyPointCloud.exe"),
    os.path.join(mvs_folder, "scene.mvs"),
    "--resolution-level", "1",
    "-o", os.path.join(mvs_folder, "scene_dense.mvs")
])

# 3.3 Reconstruct mesh
run([
    os.path.join(OPENMVS_BIN, "ReconstructMesh.exe"),
    os.path.join(mvs_folder, "scene_dense.mvs"),
    "-o", os.path.join(mvs_folder, "scene_dense_mesh.mvs")
])

# 3.4 Texture mesh
run([
    os.path.join(OPENMVS_BIN, "TextureMesh.exe"),
    os.path.join(mvs_folder, "scene_dense_mesh.mvs"),
    "-o", output_mesh
])

# 4. Visualize the mesh
if os.path.exists(output_mesh):
    mesh = o3d.io.read_triangle_mesh(output_mesh)
    o3d.visualization.draw_geometries([mesh])
else:
    print("Mesh not found. Check OpenMVG/OpenMVS output for errors.")