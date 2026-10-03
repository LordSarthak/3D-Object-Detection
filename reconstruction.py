import cv2
import numpy as np
import torch
import open3d as o3d
from torchvision import transforms
from PIL import Image
from midas import *
#from midas.transforms import Resize, NormalizeImage, PrepareForNet

# 1. Camera Calibration (Feature Detection and Matching)
def feature_matching(img_paths):
    # Use ORB for keypoint detection and feature matching
    orb = cv2.ORB_create()
    keypoints, descriptors = [], []
    for img_path in img_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        kp, des = orb.detectAndCompute(img, None)
        keypoints.append(kp)
        descriptors.append(des)

    # Match features between images using FLANN
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = []
    for i in range(len(img_paths) - 1):
        matches.append(bf.match(descriptors[i], descriptors[i + 1]))
    
    return keypoints, matches

# 2. Depth Estimation using MiDaS
def load_midas():
    model = DPTDepthModel(backbone="vitb", path="D:/PYTHON PROJECT/midas/dpt_hybrid.pth")
    transform = transforms.Compose([
        Resize(384, 384),
        NormalizeImage(),
        PrepareForNet()
    ])
    return model, transform

def estimate_depth(model, transform, img_path):
    img = Image.open(img_path).convert("RGB")
    img_input = transform({"image": img})["image"]
    with torch.no_grad():
        depth_map = model.forward(img_input.unsqueeze(0).to("cpu"))  # Change device if needed
    return depth_map.squeeze().cpu().numpy()

# 3. 3D Reconstruction from Depth Maps
def generate_point_cloud(depth_maps, img_paths, keypoints, matches):
    point_clouds = []
    for i, depth_map in enumerate(depth_maps):
        h, w = depth_map.shape
        # Create a point cloud based on depth map
        points = []
        for y in range(h):
            for x in range(w):
                depth = depth_map[y, x]
                if depth > 0:  # Only consider valid depth points
                    points.append([x, y, depth])
        
        # Convert to Open3D PointCloud
        points = np.array(points)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_clouds.append(point_cloud)
    
    # Merge the point clouds
    full_point_cloud = point_clouds[0]
    for pc in point_clouds[1:]:
        full_point_cloud += pc
    
    return full_point_cloud

# 4. Mesh Generation
def generate_mesh_from_point_cloud(point_cloud):
    # Estimate normals
    point_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # Poisson surface reconstruction for mesh
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=9)
    
    return mesh

# 5. Visualization
def visualize_3d_model(mesh):
    o3d.visualization.draw_geometries([mesh])

# Main function to run the process
def reconstruct_3d(image_folder):
    img_paths = [image_folder + f"/img{i}.jpg" for i in range(1, 5)]  # Example: img1.jpg, img2.jpg, etc.
    
    keypoints, matches = feature_matching(img_paths)
    model, transform = load_midas()
    
    depth_maps = []
    for img_path in img_paths:
        depth_map = estimate_depth(model, transform, img_path)
        depth_maps.append(depth_map)
    
    # Generate point cloud
    point_cloud = generate_point_cloud(depth_maps, img_paths, keypoints, matches)
    
    # Create mesh from point cloud
    mesh = generate_mesh_from_point_cloud(point_cloud)
    
    # Visualize the 3D mesh
    visualize_3d_model(mesh)

# Example usage
image_folder = "D:/PYTHON PROJECT/images"
reconstruct_3d(image_folder)