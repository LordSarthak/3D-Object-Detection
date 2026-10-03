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

# Load and resize image (keeping aspect ratio)
image = Image.open('trial.jpg')
new_height = 480 if image.height > 480 else image.height
new_height -= (new_height % 32)
new_width = int(new_height * image.width / image.height)
new_width -= new_width % 32
image = image.resize((new_width, new_height))

# Prepare input for model
inputs = feature_extractor(images=image, return_tensors="pt")

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

# Convert depth output to numpy
depth_map = predicted_depth.squeeze().cpu().numpy() * 1000.0  # Convert to mm

# **Enhancement: Super-resolution depth upsampling**
depth_map = cv2.resize(depth_map, (new_width * 2, new_height * 2), interpolation=cv2.INTER_CUBIC)
image = image.resize((new_width * 2, new_height * 2))  # Resize RGB image accordingly

# **Enhancement: Edge-aware filtering for noise reduction**
depth_filtered = cv2.ximgproc.guidedFilter(
    guide=cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY),
    src=depth_map.astype(np.float32),
    radius=5,
    eps=1e-2
)

# Show improved depth map
fig, ax = plt.subplots(1, 2)
ax[0].imshow(image)
ax[0].set_title("RGB Image")
ax[0].axis("off")
ax[1].imshow(depth_filtered, cmap='plasma')
ax[1].set_title("Enhanced Depth Map")
ax[1].axis("off")
plt.tight_layout()
plt.show()

# Convert to Open3D format
depth_o3d = o3d.geometry.Image(depth_filtered.astype(np.float32))
image_o3d = o3d.geometry.Image(np.array(image))

# Create an RGBD image
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    image_o3d, depth_o3d, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False
)

# Define camera intrinsics (Modify focal lengths if needed)
camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
camera_intrinsic.set_intrinsics(new_width * 2, new_height * 2, 500, 500, new_width, new_height)

# Generate 3D point cloud
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)

# **Enhancement: Remove outliers for a cleaner point cloud**
pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

# **Enhancement: Refine normal estimation**
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50))

# **Enhancement: Improve surface quality with smoothing**
pcd.orient_normals_consistent_tangent_plane(100)

# **Enhancement: Better visualization settings**
o3d.visualization.draw_geometries_with_editing([pcd])
