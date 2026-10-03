import os
import matplotlib.pyplot as plt
from PIL import Image
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import torch
import numpy as np
import open3d as o3d

# Load model and feature extractor
feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

# Folder containing multiple angles of the same object
image_folder = "images"  # Change this to your actual folder
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# Initialize point cloud for merging multiple angles
merged_pcd = o3d.geometry.PointCloud()

# Process each image in the folder
for img_name in sorted(os.listdir(image_folder)):  # Sort ensures proper order
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Filter image files
        continue

    img_path = os.path.join(image_folder, img_name)
    image = Image.open(img_path)

    # Resize image to fit model
    new_height = 480 if image.height > 480 else image.height
    new_height -= (new_height % 32)
    new_width = int(new_height * image.width / image.height)
    diff = new_width % 32
    new_width = new_width - diff if diff < 16 else new_width + 32 - diff
    new_size = (new_width, new_height)
    image = image.resize(new_size)

    # Process image through model
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Remove padding and normalize depth
    pad = 16
    output = predicted_depth.squeeze().cpu().numpy() * 1000.0
    output = output[pad:-pad, pad:-pad]
    image = image.crop((pad, pad, image.width - pad, image.height - pad))

    # Save depth image
    depth_image = (output * 255 / np.max(output)).astype('uint8')
    depth_img_path = os.path.join(output_folder, f"depth_{img_name}")
    Image.fromarray(depth_image).save(depth_img_path)

    # Display results
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image)
    ax[0].set_title(f"Original Image: {img_name}")
    ax[0].axis("off")

    ax[1].imshow(output, cmap='plasma')
    ax[1].set_title("Depth Map")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(2)  # Show for 2 seconds
    plt.close()

    # 3D Point Cloud Generation
    width, height = image.size
    image_np = np.array(image)

    depth_o3d = o3d.geometry.Image(depth_image)
    image_o3d = o3d.geometry.Image(image_np)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        image_o3d, depth_o3d, convert_rgb_to_intensity=False
    )

    # Camera Intrinsics (Adjust focal length if needed)
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsic.set_intrinsics(width, height, 500, 500, width / 2, height / 2)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)

    # Merge multiple views into one point cloud
    merged_pcd += pcd

# Save and visualize final merged point cloud
pcd_output_path = os.path.join(output_folder, "merged_pointcloud.ply")
o3d.io.write_point_cloud(pcd_output_path, merged_pcd)
print(f"Saved merged point cloud: {pcd_output_path}")

o3d.visualization.draw_geometries([merged_pcd])  # Show final 3D reconstruction

print("Processing complete!")
