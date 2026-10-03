import os
import subprocess
from PIL import Image
import open3d as o3d

# Set up paths
COLMAP_BIN = r"C:\colmap\colmap.exe"  # Update to your COLMAP executable path

input_folder = r"d:\PYTHON PROJECT\foot_images"
preprocessed_folder = r"d:\PYTHON PROJECT\preprocessed_images"
colmap_db = r"d:\PYTHON PROJECT\colmap.db"
sparse_folder = r"d:\PYTHON PROJECT\sparse"
dense_folder = r"d:\PYTHON PROJECT\dense"
output_mesh = os.path.join(dense_folder, "meshed-poisson.ply")

os.makedirs(preprocessed_folder, exist_ok=True)
os.makedirs(sparse_folder, exist_ok=True)
os.makedirs(dense_folder, exist_ok=True)

# 1. Preprocess images
for fname in os.listdir(input_folder):
    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(input_folder, fname)
        img = Image.open(img_path).convert("RGB")
        img.thumbnail((1024, 1024))
        img.save(os.path.join(preprocessed_folder, fname))

def run(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

# 2. Feature extraction
run([
    COLMAP_BIN, "feature_extractor",
    "--database_path", colmap_db,
    "--image_path", preprocessed_folder
])

# 3. Exhaustive matcher
run([
    COLMAP_BIN, "exhaustive_matcher",
    "--database_path", colmap_db
])

# 4. Sparse reconstruction (mapper)
run([
    COLMAP_BIN, "mapper",
    "--database_path", colmap_db,
    "--image_path", preprocessed_folder,
    "--output_path", sparse_folder
])

# 5. Image undistortion (needed for dense reconstruction)
run([
    COLMAP_BIN, "image_undistorter",
    "--image_path", preprocessed_folder,
    "--input_path", os.path.join(sparse_folder, "0"),
    "--output_path", dense_folder,
    "--output_type", "COLMAP"
])

# 6. Dense stereo
run([
    COLMAP_BIN, "patch_match_stereo",
    "--workspace_path", dense_folder,
    "--workspace_format", "COLMAP",
    "--PatchMatchStereo.geom_consistency", "true"
])

# 7. Dense fusion
run([
    COLMAP_BIN, "stereo_fusion",
    "--workspace_path", dense_folder,
    "--workspace_format", "COLMAP",
    "--input_type", "geometric",
    "--output_path", os.path.join(dense_folder, "fused.ply")
])

# 8. Surface reconstruction (Poisson)
run([
    COLMAP_BIN, "poisson_mesher",
    "--input_path", os.path.join(dense_folder, "fused.ply"),
    "--output_path", output_mesh
])

# 9. Visualize the mesh
if os.path.exists(output_mesh):
    mesh = o3d.io.read_triangle_mesh(output_mesh)
    o3d.visualization.draw_geometries([mesh])
else:
    print("Mesh not found. Check COLMAP output for errors.")