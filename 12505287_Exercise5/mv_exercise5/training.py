from pathlib import Path
import open3d as o3d       
import numpy as np         
import cv2
import numpy as np
import matplotlib as plt
import math
import random

from camera_params import fx_rgb, fy_rgb, cx_rgb, cy_rgb
from pcl import *

def visualize_training_set(training_images_gray, training_labels, classes,
                                    n_display=10, images_per_row=5):
    selected_indices = []
    
    for cls in classes:
        cls_indices = [i for i, label in enumerate(training_labels) if label == cls]
        if cls_indices:
            selected_indices.append(random.choice(cls_indices))

    remaining = n_display - len(selected_indices)
    if remaining > 0:
        all_indices = list(range(len(training_images_gray)))
        available_indices = [i for i in all_indices if i not in selected_indices]
        selected_indices.extend(random.sample(available_indices, remaining))

    n_rows = math.ceil(n_display / images_per_row)
    plt.figure(figsize=(images_per_row * 3, n_rows * 3))

    for i, idx in enumerate(selected_indices):
        img = training_images_gray[idx]
        label = training_labels[idx]

        plt.subplot(n_rows, images_per_row, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(label)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def prepare_training_data_auto(classes, n_per_class, base_path,
                               fx, fy, cx, cy,
                               img_width=640, img_height=480, plots = True):
    training_pcd_paths = []
    training_labels = []

    for class_name, n_images in zip(classes, n_per_class):
        for i in range(n_images):
            filename = f"{class_name}{i:03d}.pcd"
            file_path = Path(base_path) / filename
            training_pcd_paths.append(file_path)
            training_labels.append(class_name)

    training_images_gray = []

    for pcd_file in training_pcd_paths:
        pcd = o3d.io.read_point_cloud(str(pcd_file))

        img = pointcloud_to_2d_image(
            pcd, fx, fy, cx, cy, grayscale=True, plots=False
        )

        training_images_gray.append(img)

    if plots:
        visualize_training_set(training_images_gray, training_labels, classes)

    return training_images_gray, training_labels

if __name__ == '__main__':
    classes = ["book", "cookiebox", "cup", "ketchup", "sugar", "sweets", "tea"]
    n_per_class = [17, 22, 17, 22, 22, 22, 22]
    base_path = "training"

    training_images_gray, training_labels = prepare_training_data_auto(classes, n_per_class, base_path, fx_rgb, fy_rgb, cx_rgb, cy_rgb)

    print(len(training_images_gray))