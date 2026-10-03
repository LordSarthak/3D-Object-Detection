import os
import subprocess
from PIL import Image
import open3d as o3d

# Path to Meshroom batch executable (update this to your Meshroom install location)
MESHROOM_BIN = r"C:\Meshroom\meshroom_batch.exe"

# Folder containing your original images
input_folder = r"d:\PYTHON PROJECT\foot_images"

# Temporary folder for preprocessed images
preprocessed_folder = r"d:\PYTHON PROJECT\preprocessed_images"
os.makedirs(preprocessed_folder, exist_ok=True)

# Output folder for Meshroom results
output_folder = r"d:\PYTHON PROJECT\meshroom_output"
os.makedirs(output_folder, exist_ok=True)

# Preprocess images using Pillow
for fname in os.listdir(input_folder):
    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(input_folder, fname)
        img = Image.open(img_path)
        # Example preprocessing: resize to max 1024x1024, convert to RGB
        img = img.convert("RGB")
        img.thumbnail((1024, 1024))
        save_path = os.path.join(preprocessed_folder, fname)
        img.save(save_path)

# List all preprocessed image files
image_files = [f for f in os.listdir(preprocessed_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
if len(image_files) < 2:
    raise RuntimeError("Please provide at least two images in the input folder for photogrammetry.")

print(f"Found {len(image_files)} preprocessed images for reconstruction.")

# Run Meshroom in batch mode
cmd = [
    MESHROOM_BIN,
    "--input", preprocessed_folder,
    "--output", output_folder
]
print("Running Meshroom photogrammetry pipeline...")
subprocess.run(cmd, check=True)
print("Meshroom processing complete.")

# Meshroom typically outputs 'texturedMesh.obj' in the output folder
mesh_path_obj = os.path.join(output_folder, "texturedMesh.obj")
mesh_path_ply = os.path.join(output_folder, "texturedMesh.ply")

# Try to load the mesh (OBJ or PLY)
if os.path.exists(mesh_path_ply):
    mesh_path = mesh_path_ply
elif os.path.exists(mesh_path_obj):
    mesh_path = mesh_path_obj
else:
    raise FileNotFoundError("No mesh file found in Meshroom output. Check Meshroom logs for errors.")

print(f"Loading mesh from: {mesh_path}")
mesh = o3d.io.read_triangle_mesh(mesh_path)
if not mesh.has_triangles():
    raise RuntimeError("Loaded mesh does not contain any triangles. Reconstruction may have failed.")

# Visualize the mesh
o3d.visualization.draw_geometries([mesh])