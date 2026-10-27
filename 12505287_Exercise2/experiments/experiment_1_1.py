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

def plot_percentages(results, match_percentatges, tolerance):
    angles_list = [r['angle'] for r in results]
    
    plt.figure()

    for (label, percentages) in match_percentatges:
        plt.plot(angles_list, percentages, marker='o', linestyle='-', linewidth=2, label = label)

    plt.title(f"Harris rotational invariance (match ±{tolerance}px)")
    plt.xlabel("Rotation angle (degrees)")
    plt.ylabel("Matched keypoints (%)")
    plt.ylim([90, 110])
    plt.grid(True)
    plt.legend()
    plt.show()


def matching_corners_percentage(results, tolerance):
    tolerance_px = 5

    kp_orig = results[0]['keypoints']
    kp_orig_pts = np.array([kp.pt for kp in kp_orig])

    match_percentages = []

    for r in results:

        kp_remap = r['keypoints_unrotated']
        kp_remap_pts = np.array([kp.pt for kp in kp_remap])

        matched = 0
        for pt in kp_remap_pts:
            dists = np.linalg.norm(kp_orig_pts - pt, axis=1)
            if np.any(dists <= tolerance_px):
                matched += 1

        percent_match = 100 * matched / len(kp_orig_pts) if len(kp_remap_pts) > 0 else 0
        match_percentages.append(percent_match)

    return match_percentages

def matching_corners_percentage_one_to_one(results, tolerance):
    tolerance_px = tolerance
    kp_orig_pts = np.array([kp.pt for kp in results[0]['keypoints']])
    match_percentages = []

    for r in results:

        kp_remap_pts = np.array([kp.pt for kp in r['keypoints_unrotated']])
        matched_orig_idx = set()  
        matched = 0

        for pt in kp_remap_pts:
            
            dists = np.linalg.norm(kp_orig_pts - pt, axis=1)
            
            candidate_idx = np.where(dists <= tolerance_px)[0]

            for idx in candidate_idx:
                if idx not in matched_orig_idx:
                    matched += 1
                    matched_orig_idx.add(idx)
                    break  

        percent_match = 100 * matched / len(kp_orig_pts) if len(kp_orig_pts) > 0 else 0
        match_percentages.append(percent_match)

    return match_percentages

def plot_num_keypoints_vs_angles(results):
    angles_list = [r['angle'] for r in results]
    num_kp = [len(r['keypoints']) for r in results]

    plt.figure()
    plt.plot(angles_list, num_kp, marker='o', linestyle='-', linewidth=2)
    plt.title("Keypoints detected vs Rotation angle")
    plt.xlabel("Rotation angle (degrees)")
    plt.ylabel("Number of keypoints")
    plt.grid(True)
    plt.show()

    return

def unrotate_keypoints(keypoints, img_shape_orig, angle):
    h, w = img_shape_orig
    center_x, center_y = w / 2, h / 2

    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), -angle, 1.0)
    cos = np.abs(rotation_matrix[0,0])
    sin = np.abs(rotation_matrix[0,1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    rotation_matrix[0,2] += (new_w / 2) - center_x
    rotation_matrix[1,2] += (new_h / 2) - center_y

    M_full = np.vstack([rotation_matrix, [0,0,1]])
    M_inv = np.linalg.inv(M_full)

    x_rot = np.array([kp.pt[0] for kp in keypoints])
    y_rot = np.array([kp.pt[1] for kp in keypoints])
    ones = np.ones_like(x_rot)
    pts_rot = np.stack([x_rot, y_rot, ones], axis=0)

    pts_orig = M_inv @ pts_rot
    x_orig, y_orig = pts_orig[0,:], pts_orig[1,:]

    keypoints_orig = []
    for i, kp in enumerate(keypoints):
        new_kp = cv2.KeyPoint(float(x_orig[i]), float(y_orig[i]), kp.size)
        new_kp.response = kp.response
        keypoints_orig.append(new_kp)

    return keypoints_orig

if __name__ == '__main__':
    save_image = False  # Enables saving of matches image
    use_matplotlib = False  # Enables saving of matches image
    img_path_1 = 'building/building.png'

    sigma1 = 1.0
    sigma2 = 1.5
    threshold = 0.35
    k = 0.04
    patch_size = 16

    angles = np.arange(0, 180, 15, dtype=int)

    results = []

    for angle in angles:
        # Load images
        current_path = Path(__file__).parent
        img_gray_1 = cv2.imread(str(current_path.joinpath(img_path_1)), cv2.IMREAD_GRAYSCALE)

        img_shape_orig = img_gray_1.shape

        img_gray_1 = rotate_bound(img_gray_1, angle)
        img_gray_1_int = img_gray_1.copy()
        if img_gray_1 is None:
            raise FileNotFoundError("Couldn't load image " + str(current_path.joinpath(img_path_1)))

        # Convert images from uint8 with range [0,255] to float32 with range [0,1]
        img_gray_1 = img_gray_1.astype(np.float32) / 255.

        # Harris corner detector
        keypoints = harris_corner(img_gray_1,
                                sigma1=sigma1,
                                sigma2=sigma2,
                                k=k,
                                threshold=threshold)

        # Draw the keypoints
        # keypoints_img_1 = np.zeros(shape=img_gray_1.shape, dtype=np.uint8)
        # keypoints_img_1 = cv2.drawKeypoints(img_gray_1_int, keypoints, keypoints_img_1)

        # Copia dell'immagine originale
        keypoints_img_1 = cv2.cvtColor(img_gray_1_int, cv2.COLOR_GRAY2BGR)
        overlay = keypoints_img_1.copy()

        # Disegna keypoints rossi grandi
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            cv2.circle(overlay, (x, y), radius=8, color=(0,0,255), thickness=-1)  # rosso BGR

        # Mescola overlay e immagine originale con trasparenza
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, keypoints_img_1, 1 - alpha, 0, keypoints_img_1)

        keypoints_unrotated = unrotate_keypoints(keypoints, img_shape_orig, angle)

        results.append({
            'angle': angle,
            'keypoints': keypoints,
            'keypoints_unrotated': keypoints_unrotated,
            'rotated_img_with_keypoints': keypoints_img_1,
            'rotated_img': img_gray_1_int
        })

    idxs = np.arange(0, len(angles) - 1, 4, dtype=int)
    print(idxs)
    for idx in idxs:
        show_image(results[idx]['rotated_img'], f'angle: {results[idx]['angle']}')
        show_image(results[idx]['rotated_img_with_keypoints'], f'angle: {results[idx]['angle']}')

    plot_num_keypoints_vs_angles(results)

    tolerance = 3
    percentages_naive = matching_corners_percentage(results, tolerance)
    percentages_one_to_one = matching_corners_percentage_one_to_one(results, tolerance)

    percentages = [('naive', percentages_naive), ('one_to_one', percentages_one_to_one)]

    plot_percentages(results, percentages, tolerance)