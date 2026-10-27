#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Machine Vision (376.081)
Exercise 2: Interest Points and Descriptors
Clara Haider, Matthias Hirschmanner 2025
Automation & Control Institute, TU Wien

Tutors: machinevision@acin.tuwien.ac.at

resources: https://chatgpt.com/share/6908da20-5ea8-800e-aab1-689460e408f9 (understand rotations)
"""

from pathlib import Path

import cv2
import numpy as np

from harris_corner import harris_corner
from descriptors import compute_descriptors
from helper_functions import *
from scipy.spatial.distance import cdist

def keypoints_to_xy_array(kps):
    return np.array([kp.pt for kp in kps], dtype=np.float32)

def compute_match_matrix(all_kp_remapped, tolerance=3.0):
    n = len(all_kp_remapped)
    match_matrix = np.zeros((n, n), dtype=np.float32)

    all_xy = [keypoints_to_xy_array(kps) for kps in all_kp_remapped]

    for i, kp_i in enumerate(all_xy):
        for j, kp_j in enumerate(all_xy):

            kp_ref = kp_i
            kp_other = kp_j

            if len(kp_ref) == 0:
                match_percent = 0.0
            elif len(kp_other) == 0:
                match_percent = 0.0
            else:
                dists = cdist(kp_ref, kp_other)
                min_dists = np.min(dists, axis=1)
                matched_count = np.sum(min_dists <= tolerance)
                non_matched_count = np.sum(min_dists > tolerance)
                total_points = matched_count + non_matched_count
                match_percent = matched_count / total_points * 100.0

            match_matrix[i, j] = match_percent

    return match_matrix

def count_matches(kp_list_patch, kp_list_full, tolerance=3):
    pts_full = np.array([[kp.pt[0], kp.pt[1]] for kp in kp_list_full])

    matched = 0

    for kp in kp_list_patch:
        px, py = kp.pt
        d = np.sqrt((pts_full[:,0] - px)**2 + (pts_full[:,1] - py)**2)
        if np.any(d <= tolerance):
            matched += 1

    return matched

def plot_difference_heatmap(diff_matrix, title="matching keypoints percentages"):
    n = diff_matrix.shape[0]
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(diff_matrix, cmap='viridis', vmin=0, vmax=100)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{diff_matrix[i, j]:.1f}", color='white' if diff_matrix[i, j] > 50 else 'black',
                    ha='center', va='center', fontsize=9)

    ax.set_title(title)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(range(n))
    ax.set_yticklabels(range(n))
    fig.colorbar(cax, ax=ax, label='% match')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    save_image = False  # Enables saving of matches image
    use_matplotlib = False  # Enables saving of matches image
    img_path_1 = 'desk/Image-00.jpg'

    sigma1 = 1.0
    sigma2 = 2.0
    threshold = 0.5
    k = 0.04

    scale_factors = [0.5, 1.0, 3.0]

    current_path = Path(__file__).parent
    img_gray_1 = cv2.imread(str(current_path.joinpath(img_path_1)), cv2.IMREAD_GRAYSCALE)

    if img_gray_1 is None:
        raise FileNotFoundError("Couldn't load image " + str(current_path.joinpath(img_path_1)))

    all_kp_remapped = []

    for scale in scale_factors:
        new_size = (int(img_gray_1.shape[1] * scale), int(img_gray_1.shape[0] * scale))
        img_scaled_int = cv2.resize(img_gray_1.copy(), new_size, interpolation=cv2.INTER_LINEAR)
            
        # Convert images from uint8 with range [0,255] to float32 with range [0,1]
        img_scaled = img_scaled_int.astype(np.float32) / 255.

        # Harris corner detector
        keypoints = harris_corner(img_scaled,
            sigma1=sigma1,
            sigma2=sigma2,
            k=k,
            threshold=threshold)
        
        img_bgr = cv2.cvtColor(img_scaled_int, cv2.COLOR_GRAY2BGR)
        
        keypoints_img = np.zeros_like(img_scaled_int)
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            radius = int(kp.size * 2.5) 
            cv2.circle(img_bgr, (x, y), radius, (0,0,255), 2)

        show_image(img_bgr, f"Harris Corners (scale {scale})",
           save_image=save_image, use_matplotlib=use_matplotlib)
        

        kp_global = []
        for kp in keypoints:
            x_global = kp.pt[0] / scale
            y_global = kp.pt[1] / scale
            new_kp = cv2.KeyPoint(float(x_global), float(y_global), kp.size)
            new_kp.response = kp.response
            kp_global.append(new_kp)

        all_kp_remapped.append(kp_global)

    diff_matrix = compute_match_matrix(all_kp_remapped)
    plot_difference_heatmap(diff_matrix)
