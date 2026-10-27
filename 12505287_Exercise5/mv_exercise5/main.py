from pathlib import Path
import open3d as o3d       
import numpy as np         
import cv2                 
import os 
import matplotlib.pyplot as plt
import time

from camera_params import fx_rgb, fy_rgb, cx_rgb, cy_rgb
from pcl import *
from helper_functions import *
from clustering import *
from training import *
from sift import *

def pipeline(test_image, whole_image, plots, show_final_results=True):
    training_path = "training"
    test_path = "test"
    image = test_image

    test_image_path = test_path + "/" + image

    # image upload and 2D image retrieving
    pcd, pcd_sampled, image_2D = load_project_visualize(test_image_path, fx_rgb, fy_rgb, cx_rgb, cy_rgb, plots=plots)

    # let's cluster all the items using DBSCAN (and filtering the useless points)
    scene_pcd, labels_for_scene_pcd = clustering_pipeline(pcd_sampled, plots=plots)

    # let's compute the label_mask in 2D
    mask_2d = clusters_to_2d_mask(scene_pcd, labels_for_scene_pcd, fx_rgb, fy_rgb, cx_rgb, cy_rgb, plots=plots)

    clusters = np.unique(mask_2d)
    clusters = clusters[clusters != 0]
    print(clusters)

    objects_images = {}
    for id in clusters:
        object = apply_mask(image_2D, mask_2d, cluster = id, plots=plots)
        objects_images[id] = object

    all_object_image = apply_mask(image_2D, mask_2d, plots=plots)

    classes = ["book", "cookiebox", "cup", "ketchup", "sugar", "sweets", "tea"]
    n_per_class = [17, 22, 17, 22, 22, 22, 22]
    base_path = "training"

    training_images_gray, training_labels = prepare_training_data_auto(classes, n_per_class, base_path, fx_rgb, fy_rgb, cx_rgb, cy_rgb, plots=plots)

    print(f"Uploaded {len(training_images_gray)} training images")

    if whole_image:
        predicted_classes = sift_whole_image(training_images_gray, training_labels, all_object_image, mask_2d, plots=plots)

        label_clusters_with_text(image_2D, mask_2d, predicted_classes, plots = show_final_results)
    else:
        cluster_classes = count_matches_in_training_per_cluster(training_images_gray, training_labels, objects_images, plots=plots)

        predicted_classes = predict_class_per_cluster(cluster_classes, plots=plots)

        label_clusters_with_text(image_2D, mask_2d, predicted_classes, plots = show_final_results)
        
    return predicted_classes

if __name__ == '__main__':
    whole_image = True

    training_path = "training"
    test_path = "test"
    image = "image001.pcd"

    test_image_path = test_path + "/" + image
    
    timings = {}

    t0 = time.perf_counter()

    t = time.perf_counter()

    # image upload and 2D image retrieving
    pcd, pcd_sampled, image_2D = load_project_visualize(test_image_path, fx_rgb, fy_rgb, cx_rgb, cy_rgb)

    timings["load_and_projection"] = time.perf_counter() - t

    t = time.perf_counter()
    # let's cluster all the items using DBSCAN (and filtering the useless points)
    scene_pcd, labels_for_scene_pcd = clustering_pipeline(pcd_sampled)
    timings["clustering"] = time.perf_counter() - t

    t = time.perf_counter()
    # let's compute the labell_mask in 2D
    mask_2d = clusters_to_2d_mask(scene_pcd, labels_for_scene_pcd, fx_rgb, fy_rgb, cx_rgb, cy_rgb)
    timings["mask_creation"] = time.perf_counter() - t

    clusters = np.unique(mask_2d)
    clusters = clusters[clusters != 0]
    print(clusters)

    t = time.perf_counter()
    objects_images = {}
    for id in clusters:
        print("Current id:", id)
        object = apply_mask(image_2D, mask_2d, cluster = id)
        objects_images[id] = object

    all_object_image = apply_mask(image_2D, mask_2d)

    timings["object_extraction"] = time.perf_counter() - t

    classes = ["book", "cookiebox", "cup", "ketchup", "sugar", "sweets", "tea"]
    n_per_class = [17, 22, 17, 22, 22, 22, 22]
    base_path = "training"

    t = time.perf_counter()
    training_images_gray, training_labels = prepare_training_data_auto(classes, n_per_class, base_path, fx_rgb, fy_rgb, cx_rgb, cy_rgb)

    print(f"Uploaded {len(training_images_gray)} training images")
    timings["training_loading"] = time.perf_counter() - t

    t = time.perf_counter()
    if whole_image:
        predicted_classes = sift_whole_image(training_images_gray, training_labels, all_object_image, mask_2d)

        label_clusters_with_text(image_2D, mask_2d, predicted_classes)
    else:
        cluster_classes = count_matches_in_training_per_cluster(training_images_gray, training_labels, objects_images)

        predicted_classes = predict_class_per_cluster(cluster_classes)

        label_clusters_with_text(image_2D, mask_2d, predicted_classes)
    
    timings["recognition"] = time.perf_counter() - t

    timings["total"] = time.perf_counter() - t0

    print("\n--- Execution times ---")
    for k, v in timings.items():
        print(f"{k:25s}: {v:.3f} s")