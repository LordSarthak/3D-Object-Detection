import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import cv2
import torch
from PIL import Image
from transformers import GLPNImageProcessor, GLPNForDepthEstimation

# Load GLPN model and processor
feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

# Load and resize image
image = Image.open('trial.jpg')
new_height = 480 if image.height > 480 else image.height
new_height -= (new_height % 32)
new_width = int(new_height * image.width / image.height)
diff = new_width % 32
new_width = new_width - diff if diff < 16 else new_width + 32 - diff
new_size = (new_width, new_height)
image = image.resize(new_size)

# Prepare input for model
inputs = feature_extractor(images=image, return_tensors="pt")

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

# Convert depth output to numpy
pad = 16
output = predicted_depth.squeeze().cpu().numpy() * 1000.0  # Convert to mm
output = output[pad:-pad, pad:-pad]  # Remove padding
image = image.crop((pad, pad, image.width - pad, image.height - pad))  # Crop RGB image accordingly

# Apply bilateral filter to smooth depth while preserving edges
output_filtered = cv2.bilateralFilter(output.astype(np.float32), d=5, sigmaColor=50, sigmaSpace=50)

# Display depth and image side by side
fig, ax = plt.subplots(1, 2)
ax[0].imshow(image)
ax[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
ax[1].imshow(output_filtered, cmap='plasma')
ax[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
plt.tight_layout()
plt.show()

# Convert to Open3D format
width, height = image.size
depth_o3d = o3d.geometry.Image(output_filtered.astype(np.float32))  # Keep depth as float
image_o3d = o3d.geometry.Image(np.array(image))  # Convert RGB to Open3D format

# Create an RGBD image
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    image_o3d, depth_o3d, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False
)

# Define camera intrinsics (Adjust focal lengths as needed)
camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
camera_intrinsic.set_intrinsics(width, height, 500, 500, width / 2, height / 2)

# Generate 3D point cloud
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)

# Denoise point cloud by downsampling
pcd = pcd.voxel_down_sample(voxel_size=0.005)

# Estimate normals for better visualization
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Show point cloud interactively
o3d.visualization.draw_geometries_with_editing([pcd])
