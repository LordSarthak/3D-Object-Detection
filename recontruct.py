import cv2
import numpy as np
import open3d as o3d
import glob
import os

def load_images_from_folder(folder):
    images = []
    for filename in sorted(glob.glob(os.path.join(folder, '*.jpg'))):
        img = cv2.imread(filename)
        if img is not None:
            images.append(img)
    return images

def detect_and_compute_features(images):
    sift = cv2.SIFT_create()
    keypoints_list = []
    descriptors_list = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)
    return keypoints_list, descriptors_list

def match_features(des1, des2):
    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE = 1
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches

def estimate_pose(kp1, kp2, matches, K):
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
    
    return R, t, pts1, pts2, mask_pose

def triangulate_points(kp1, kp2, matches, K, R, t):
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    proj1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    proj2 = np.hstack((R, t))
    
    P1 = K @ proj1
    P2 = K @ proj2

    pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3d = pts4d[:3, :] / pts4d[3, :]
    return pts3d.T

def create_point_cloud(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def main():
    folder = 'D:/PYTHON PROJECT/images'  # Your images folder
    images = load_images_from_folder(folder)
    
    if len(images) < 2:
        print("Need at least 2 images.")
        return

    print(f"Loaded {len(images)} images.")

    keypoints_list, descriptors_list = detect_and_compute_features(images)

    # Camera matrix (Assume fx = fy, cx, cy at center) --> Update for your camera
    img_h, img_w = images[0].shape[:2]
    f = 0.8 * img_w  # Focal length guess
    K = np.array([
        [f, 0, img_w / 2],
        [0, f, img_h / 2],
        [0, 0, 1]
    ])

    all_points = []

    for i in range(len(images)-1):
        kp1 = keypoints_list[i]
        kp2 = keypoints_list[i+1]
        des1 = descriptors_list[i]
        des2 = descriptors_list[i+1]

        matches = match_features(des1, des2)

        if len(matches) < 8:
            print(f"Not enough matches between image {i} and {i+1}. Skipping.")
            continue

        R, t, pts1, pts2, mask_pose = estimate_pose(kp1, kp2, matches, K)

        pts3d = triangulate_points(kp1, kp2, matches, K, R, t)

        all_points.append(pts3d)

    if not all_points:
        print("No 3D points found.")
        return

    all_points = np.vstack(all_points)

    # Create and visualize point cloud
    pcd = create_point_cloud(all_points)
    o3d.visualization.draw_geometries([pcd])

    # Optional: Save point cloud
    o3d.io.write_point_cloud("output_reconstruction.ply", pcd)
    print("Point cloud saved as output_reconstruction.ply")

if __name__ == "__main__":
    main()
