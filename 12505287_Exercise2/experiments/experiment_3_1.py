#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Machine Vision (376.081)
Exercise 2: Object recognition with SIFT & RANSAC
Matthias Hirschmanner 2025
Automation & Control Institute, TU Wien

Tutors: machinevision@acin.tuwien.ac.at

"""
from pathlib import Path

import numpy as np
import cv2

from find_homography import find_homography_leastsquares, find_homography_ransac
from helper_functions import *

def plot_mean_histogram(distances_per_ratio, title="Mean LS vs RANSAC Distances"):
    ratios = [entry['ratio'] for entry in distances_per_ratio]
    means = [entry['mean'] for entry in distances_per_ratio]
    sdvs = [entry['sdv'] for entry in distances_per_ratio]

    x = np.arange(len(ratios))  # posizioni discrete per le barre
    ratios_labels = [f"{r:.1f}" for r in ratios]

    plt.figure(figsize=(8,5))
    plt.bar(x, means, yerr=sdvs, capsize=5, alpha=0.7, color='skyblue', edgecolor='black')
    
    plt.xticks(x, ratios_labels)
    plt.xlabel("Lowe ratio")
    plt.ylabel("Mean distance (pixels)")
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def homography_point_difference(homography_1: np.ndarray, homography_2: np.ndarray, object_img: np.ndarray) -> np.ndarray:
    h, w = object_img.shape[:2]

    points = np.array([[0,0], [w,0], [w,h], [0,h]], dtype=np.float32).reshape(-1,1,2)

    points_H1 = cv2.perspectiveTransform(points, homography_1)
    points_H2 = cv2.perspectiveTransform(points, homography_2)

    distances = np.linalg.norm(points_H1 - points_H2, axis=2).flatten()

    return distances

if __name__ == '__main__':
    # You can change the parameters here. You should not need to change anything else
    image_nr = 2
    save_image = False
    use_matplotlib = False
    debug = False

    ransac_confidence = 0.95
    ransac_inlier_threshold = 5.

    ratios = np.arange(0.2, 1., 0.1)

    # Get path
    current_path = Path(__file__).parent

    # Load images.
    # scene_img  -> image in which we want to detect the object (trainImage in OpenCV)
    # object_img -> image of the object we want to detect (queryImage in OpenCV)
    scene_img = cv2.imread(str(current_path.joinpath("data/image")) + str(image_nr) + ".jpg")
    if scene_img is None:
        raise FileNotFoundError("Couldn't load image in " + str(current_path))
    scene_img_gray = cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY)  # trainImage

    object_img = cv2.imread(str(current_path.joinpath("data/object.jpg")))
    if object_img is None:
        raise FileNotFoundError("Couldn't load image in " + str(current_path))
    object_img_gray = cv2.cvtColor(object_img, cv2.COLOR_BGR2GRAY)  # queryImage

    # Get SIFT keypoints_1 and descriptors
    sift = cv2.SIFT_create()
    target_keypoints, target_descriptors = sift.detectAndCompute(scene_img_gray, None)
    source_keypoints, source_descriptors = sift.detectAndCompute(object_img_gray, None)

    # FLANN (Fast Library for Approximate Nearest Neighbors) parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # For each keypoint in object_img get the 3 best matches
    original_matches = flann.knnMatch(source_descriptors, target_descriptors, k=2)

    distances_per_ratio = []

    for ratio in ratios:
        matches = filter_matches_with_ratio(original_matches, ratio)

        if debug:
            draw_params = dict(flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            matches_img = cv2.drawMatches(object_img,
                                      source_keypoints,
                                      scene_img,
                                      target_keypoints,
                                      matches,
                                      None,
                                      **draw_params)
            show_image(matches_img, "Matches", save_image=save_image, use_matplotlib=use_matplotlib)

        # Convert keypoints arrays with shape (n, 2) in the OpenCV convention (x,y).
        # source_points[i] is the matching keypoint to target_points[i]
        source_points = np.array([source_keypoints[match.queryIdx].pt for match in matches])
        target_points = np.array([target_keypoints[match.trainIdx].pt for match in matches])

        homography, _, _ = find_homography_ransac(source_points,
                                                target_points,
                                                confidence=ransac_confidence,
                                                inlier_threshold=ransac_inlier_threshold)
    
    
        homography_ls = find_homography_leastsquares(source_points, target_points)

        # Plot results
        plot_img = draw_rectangles(scene_img, object_img, homography_ls)
        show_image(plot_img, "Final Result", save_image=save_image, use_matplotlib=use_matplotlib)

        transformed_object_img = cv2.warpPerspective(object_img, homography_ls, dsize=scene_img.shape[1::-1])
        scene_img_blend = scene_img.copy()
        scene_img_blend[transformed_object_img != 0] = transformed_object_img[transformed_object_img != 0]
        # show_image(scene_img_blend, "Overlay Object", save_image=save_image, use_matplotlib=use_matplotlib)

        distances = homography_point_difference(homography_ls, homography, object_img)

        distances_per_ratio.append({'ratio': ratio,
                                    'distances': distances,
                                    'mean': distances.mean(),
                                    'sdv': np.std(distances)})

    plot_mean_histogram(distances_per_ratio)