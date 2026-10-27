import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
from camera_params import fx_rgb, fy_rgb, cx_rgb, cy_rgb

def load_pointcloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    if pcd.is_empty():
        raise ValueError(f"Error in upload of: {file_path}")
    
    return pcd

def downsample_pointcloud(pcd, voxel_size=0.01):
    points_numpy = np.array(pcd.points)
    invalid_indices = np.where(np.all(points_numpy == [0,0,0], axis = 1))[0]
    pcd = pcd.select_by_index(invalid_indices, invert = True)

    pcd_sampled = pcd.voxel_down_sample(voxel_size=voxel_size)

    return pcd_sampled

def visualize_pointcloud(pcd):
    if not isinstance(pcd, o3d.geometry.PointCloud):
        raise ValueError("Not a pointcloud")
    
    o3d.visualization.draw_geometries([pcd])

def pointcloud_to_2d_image(
    pcd,
    fx, fy, cx, cy,
    img_width=640,
    img_height=480,
    point_radius=2,
    dilate_iter=1,
    grayscale=False,
    plots=True
    ):

    points = np.asarray(pcd.points)
    if len(pcd.colors) > 0:
        colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    else:
        colors = np.ones_like(points, dtype=np.uint8) * 255

    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
    valid = Z > 0
    X, Y, Z = X[valid], Y[valid], Z[valid]
    colors = colors[valid]

    u = (fx * (X / Z) + cx).astype(np.int32)
    v = (fy * (Y / Z) + cy).astype(np.int32)

    mask = (u >= 0) & (u < img_width) & (v >= 0) & (v < img_height)
    u, v = u[mask], v[mask]
    colors = colors[mask]

    image = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    for ui, vi, c in zip(u, v, colors):
        cv2.circle(image, (ui, vi), point_radius, c.tolist(), -1)
        
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if plots:
        plt.figure(figsize=(6, 4))
        plt.imshow(image, cmap='gray' if grayscale else None)
        plt.title("PointCloud → 2D Projection")
        plt.axis("off")
        plt.show()

    return image

def apply_mask_to_image(image, mask, plots=True):
    if image.shape[:2] != mask.shape:
        raise ValueError("Not coherentt dimensions")

    masked_image = image.copy()

    # general structure to allow to work on grayscale image or RGB
    if image.ndim == 2:
        masked_image[mask == 0] = 0
    elif image.ndim == 3:
        masked_image[mask == 0, :] = 0
    else:
        raise ValueError("Unsupported image shape")

    if plots:
        plt.figure(figsize=(6, 4))
        if masked_image.ndim == 2:
            plt.imshow(masked_image, cmap="gray")
        else:
            plt.imshow(masked_image)
        plt.axis("off")
        plt.show()

    return masked_image

def load_project_visualize(path, fx, fy, cx, cy, img_width=640, img_height=480, greyscale=False, plots=True):

    pcd = load_pointcloud(path)

    print("Downsamplig the pointcloud to lower the computation")
    pcd_sampled = downsample_pointcloud(pcd)

    image_2D = pointcloud_to_2d_image(pcd, fx, fy, cx, cy, grayscale=greyscale, plots=plots)

    return pcd, pcd_sampled, image_2D

if __name__ == '__main__':
    path = "test/image000.pcd"
    pcd, image_2D = load_project_visualize(path, fx_rgb, fy_rgb, cx_rgb, cy_rgb)