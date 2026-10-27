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

def plot_ransac_results(results):
    confidences = np.array([r['confidence'] for r in results])
    thresholds = np.array([r['threshold'] for r in results])
    best_inliers = np.array([r['best_inliers'] for r in results])
    iterations = np.array([r['iterations'] for r in results])

    conf_unique = np.unique(confidences)
    thresh_unique = np.unique(thresholds)
    Conf, Thresh = np.meshgrid(conf_unique, thresh_unique)

    Inliers_matrix = np.zeros_like(Conf, dtype=float)
    Iter_matrix = np.zeros_like(Conf, dtype=float)

    for i, t in enumerate(thresh_unique):
        for j, c in enumerate(conf_unique):
            idx = np.where((confidences == c) & (thresholds == t))[0]
            if len(idx) > 0:
                Inliers_matrix[i, j] = best_inliers[idx[0]]
                Iter_matrix[i, j] = iterations[idx[0]]
                
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title('Best Inliers')
    plt.xlabel('Confidence')
    plt.ylabel('Threshold')
    plt.pcolormesh(Conf, Thresh, Inliers_matrix, shading='auto', cmap='viridis')
    plt.colorbar(label='Best Inliers')

    plt.subplot(1, 2, 2)
    plt.title('Iterations')
    plt.xlabel('Confidence')
    plt.ylabel('Threshold')
    plt.pcolormesh(Conf, Thresh, Iter_matrix, shading='auto', cmap='magma')
    plt.colorbar(label='Iterations')

    plt.tight_layout()
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # You can change the parameters here. You should not need to change anything else
    image_nr = 4
    save_image = False
    use_matplotlib = False

    ransac_confidence = 0.95
    ransac_inlier_threshold = 5.

    confidences = np.arange(0.7, 0.99, 0.05)
    thresholds = np.arange(3., 11., 1.)

    results = []

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
    matches = flann.knnMatch(source_descriptors, target_descriptors, k=2)
    matches = filter_matches(matches)

    # Convert keypoints arrays with shape (n, 2) in the OpenCV convention (x,y).
    # source_points[i] is the matching keypoint to target_points[i]
    source_points = np.array([source_keypoints[match.queryIdx].pt for match in matches])
    target_points = np.array([target_keypoints[match.trainIdx].pt for match in matches])

    for ransac_confidence in confidences:
        for ransac_inlier_threshold in thresholds:
            homography, best_inliers, num_iterations = find_homography_ransac(source_points,
                                                                      target_points,
                                                                      confidence=ransac_confidence,
                                                                      inlier_threshold=ransac_inlier_threshold)
            
            results.append({
                'confidence': ransac_confidence,
                'threshold': ransac_inlier_threshold,
                'best_inliers': np.sum(best_inliers),
                'iterations': num_iterations
            })

        # Plot results
        plot_img = draw_rectangles(scene_img, object_img, homography)
        # show_image(plot_img, "Final Result", save_image=save_image, use_matplotlib=use_matplotlib)

        transformed_object_img = cv2.warpPerspective(object_img, homography, dsize=scene_img.shape[1::-1])
        scene_img_blend = scene_img.copy()
        scene_img_blend[transformed_object_img != 0] = transformed_object_img[transformed_object_img != 0]
        # show_image(scene_img_blend, "Overlay Object", save_image=save_image, use_matplotlib=use_matplotlib)

    plot_ransac_results(results)
    