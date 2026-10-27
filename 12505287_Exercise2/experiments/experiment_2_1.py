#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Machine Vision (376.081)
Exercise 2: Interest Points and Descriptors
Clara Haider, Matthias Hirschmanner 2025
Automation & Control Institute, TU Wien

Tutors: machinevision@acin.tuwien.ac.at
"""

from pathlib import Path

import cv2
import numpy as np

from harris_corner import harris_corner
from descriptors import compute_descriptors
from helper_functions import *

def plot_patch_size_results(results, filename_prefix="patch_results"):
    patch_sizes = [r['patch_size'] for r in results]
    match_accuracy = [r['match_accuracy'] for r in results]
    good_matches = [r['good_matches'] for r in results]

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(patch_sizes, match_accuracy, marker='o', linestyle='-', color='blue')
    plt.title("Match Accuracy vs Patch Size")
    plt.xlabel("Patch Size")
    plt.ylabel("Match Accuracy")
    plt.xticks(patch_sizes)
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(patch_sizes, good_matches, marker='o', linestyle='-', color='green')
    plt.title("Good Matches vs Patch Size")
    plt.xlabel("Patch Size")
    plt.ylabel("# Good Matches")
    plt.xticks(patch_sizes)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    save_image = False  # Enables saving of matches image
    use_matplotlib = False  # Enables saving of matches image
    img_path_1 = 'desk/Image-00.jpg'  # Try different images
    img_path_2 = 'desk/Image-01.jpg'

    sigma1 = 1.4
    sigma2 = 3.1
    threshold = 0.3
    k = 0.04

    patch_sizes = [16, 20, 24, 28, 32, 36, 40, 44, 52]

    # Load images
    current_path = Path(__file__).parent
    img_gray_1 = cv2.imread(str(current_path.joinpath(img_path_1)), cv2.IMREAD_GRAYSCALE)
    img_gray_1_int = img_gray_1.copy()
    if img_gray_1 is None:
        raise FileNotFoundError("Couldn't load image " + str(current_path.joinpath(img_path_1)))

    img_gray_2 = cv2.imread(str(current_path.joinpath(img_path_2)), cv2.IMREAD_GRAYSCALE)
    img_gray_2_int = img_gray_2.copy()

    if img_gray_2 is None:
        raise FileNotFoundError("Couldn't load image " + str(current_path.joinpath(img_path_2)))

    # Convert images from uint8 with range [0,255] to float32 with range [0,1]
    img_gray_1 = img_gray_1.astype(np.float32) / 255.
    img_gray_2 = img_gray_2.astype(np.float32) / 255.

    # Harris corner detector
    keypoints_1 = harris_corner(img_gray_1,
                                sigma1=sigma1,
                                sigma2=sigma2,
                                k=k,
                                threshold=threshold)

    # Draw the keypoints
    keypoints_img_1 = np.zeros(shape=img_gray_1.shape, dtype=np.uint8)
    keypoints_img_1 = cv2.drawKeypoints(img_gray_1_int, keypoints_1, keypoints_img_1)
    show_image(keypoints_img_1, "Harris Corners", save_image=save_image, use_matplotlib=use_matplotlib)

    # Harris corner detector for the second image
    keypoints_2 = harris_corner(img_gray_2,
                                sigma1=sigma1,
                                sigma2=sigma2,
                                k=k,
                                threshold=threshold)
    
    # FLANN (Fast Library for Approximate Nearest Neighbors) parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    results = []

    for patch_size in patch_sizes:
        # Create descriptors
        filtered_keypoints_1, descriptors_1 = compute_descriptors(img_gray_1, keypoints_1, patch_size)
        filtered_keypoints_2, descriptors_2 = compute_descriptors(img_gray_2, keypoints_2, patch_size)

        # For each keypoint in img_gray_1 get the 2 best matches
        matches = flann.knnMatch(descriptors_1.astype(np.float32), descriptors_2.astype(np.float32), k=2)

        filtered_matches = filter_matches(matches)

        total_kp_1 = len(filtered_keypoints_1)
        total_kp_2 = len(filtered_keypoints_2)
        total_matches = len(matches)
        good_matches = len(filtered_matches)

        match_accuracy = good_matches / total_matches if total_matches > 0 else 0

        matches_img = cv2.drawMatches(img_gray_1_int,
                                  filtered_keypoints_1,
                                  img_gray_2_int,
                                  filtered_keypoints_2,
                                  filtered_matches,
                                  None)

        show_image(matches_img, "Harris Matches", save_image=save_image, use_matplotlib=use_matplotlib)

        results.append({
            'patch_size': patch_size,
            'total_kp_1': total_kp_1,
            'total_kp_2': total_kp_2,
            'total_matches': total_matches,
            'good_matches': good_matches,
            'match_accuracy': match_accuracy
        })

    plot_patch_size_results(results)