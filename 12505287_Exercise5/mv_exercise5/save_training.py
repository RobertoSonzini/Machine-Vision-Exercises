import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from camera_params import fx_rgb, fy_rgb, cx_rgb, cy_rgb

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

input_folder = "training"
output_folder = os.path.join(os.getcwd(), "output_images")
os.makedirs(output_folder, exist_ok=True)

pcd_files = [f for f in os.listdir(input_folder) if f.endswith('.pcd')]

img_width, img_height = 640, 480

for file_name in pcd_files:
    pcd_path = os.path.join(input_folder, file_name)
    pcd = o3d.io.read_point_cloud(pcd_path)
    
    image = pointcloud_to_2d_image(
        pcd, fx_rgb, fy_rgb, cx_rgb, cy_rgb,
        img_width=img_width, img_height=img_height,
        point_radius=2, plots=False
    )

    output_path = os.path.join(output_folder, file_name.replace('.pcd', '.png'))
    cv2.imwrite(output_path, image)

print(f"Images saved: {output_folder}")

