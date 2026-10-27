#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Descriptor function

Author: Roberto Sonzini Gobbi
MatrNr: 12505287
"""
from typing import List, Tuple

import numpy as np
import cv2

def compute_descriptors(img: np.ndarray,
                        keypoints: List[cv2.KeyPoint],
                        patch_size: int) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """ Calculate a descriptor on patches of the image, centred on the locations of the KeyPoints.

    Calculate a descriptor vector for each keypoint in the list. KeyPoints that are too close to the border to include
    the whole patch are filtered out. The descriptors are returned as a k x m matrix with k being the number of filtered
    KeyPoints and m being the length of a descriptor vector (patch_size**2). The descriptor at row i of
    the descriptors array is the descriptor for the KeyPoint filtered_keypoint[i].

    :param img: Grayscale input image
    :type img: np.ndarray with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param keypoints: List of keypoints at which to compute the descriptors
    :type keypoints: List[cv2.KeyPoint]

    :param patch_size: Value defining the width and height of the patch around each keypoint to calculate descriptor.
    :type patch_size: int

    :return: (filtered_keypoints, descriptors):
        filtered_keypoints: List of the filtered keypoints.
            Locations too close to the image boundary to cut out the image patch should not be contained.
        descriptors: k x m matrix containing the patch descriptors.
            Each row vector stores the descriptor vector of the respective corner.
            with k being the number of descriptors and m being the length of a descriptor (usually patch_size**2).
            The descriptor at row i belongs to the KeyPoint at filtered_keypoints[i]
    :rtype: (List[cv2.KeyPoint], np.ndarray)
    """
    ######################################################
    '''
    In my implementation, I divide each patch into 4×4 cells and compute local histograms 
    of gradient orientations, which are then concatenated and normalized to form the feature descriptor. 
    This introduces a dependency on the patch size being divisible by 4,
    which is not strictly required by the assignment. 
    I chose this structure because I wanted to experiment with a SIFT-like descriptor, 
    as it provides a more discriminative representation of the local image structure and 
    improves matching performance in my experiments.
    '''

    half = patch_size // 2
    filtered_keypoints = []
    descriptors = []

    h, w = img.shape

    cell_size = patch_size // 4
    n_bins = 8

    sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)

    sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=np.float32)

    for kp in keypoints:
        x = int(kp.pt[0])
        y = int(kp.pt[1])

        if y - half < 0 or y + half >= h or x - half < 0 or x + half >= w:
            continue

        patch = img[y-half:y+half, x-half:x+half].copy()

        Gx = cv2.filter2D(patch, -1, sobel_x)
        Gy = cv2.filter2D(patch, -1, sobel_y)

        magnitude = np.sqrt(Gx**2 + Gy**2)
        angle = np.degrees(np.arctan2(Gy, Gx))

        # sigma = patch_size / 2.0
        # yy, xx = np.mgrid[0:patch_size, 0:patch_size]
        # center = (patch_size - 1) / 2
        # gaussian = np.exp(-((xx-center)**2 + (yy-center)**2) / (2*sigma*sigma))
        # magnitude *= gaussian

        cells_mag = magnitude.reshape(4, cell_size, 4, cell_size)
        cells_ang = angle.reshape(4, cell_size, 4, cell_size)

        cells_mag = cells_mag.transpose(0,2,1,3).reshape(-1, cell_size*cell_size)
        cells_ang = cells_ang.transpose(0,2,1,3).reshape(-1, cell_size*cell_size)

        descriptor = []

        for mag, ang in zip(cells_mag, cells_ang):
            hist, _ = np.histogram(ang, bins=n_bins, range=(0,360), weights=mag)
            descriptor.extend(hist)

        descriptor = np.array(descriptor, dtype=np.float32)

        norm = np.linalg.norm(descriptor)
        if norm > 0:
            descriptor /= norm

        filtered_keypoints.append(kp)
        descriptors.append(descriptor)

    ######################################################
    return filtered_keypoints, np.array(descriptors)


def compute_descriptors_base(img: np.ndarray,
                               keypoints: List[cv2.KeyPoint],
                               patch_size: int) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """ Calculate a descriptor on patches of the image, centred on the locations of the KeyPoints.

    Calculate a descriptor vector for each keypoint in the list. KeyPoints that are too close to the border to include
    the whole patch are filtered out. The descriptors are returned as a k x m matrix with k being the number of filtered
    KeyPoints and m being the length of a descriptor vector (patch_size**2). The descriptor at row i of
    the descriptors array is the descriptor for the KeyPoint filtered_keypoint[i].

    :param img: Grayscale input image
    :type img: np.ndarray with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param keypoints: List of keypoints at which to compute the descriptors
    :type keypoints: List[cv2.KeyPoint]

    :param patch_size: Value defining the width and height of the patch around each keypoint to calculate descriptor.
    :type patch_size: int

    :return: (filtered_keypoints, descriptors):
        filtered_keypoints: List of the filtered keypoints.
            Locations too close to the image boundary to cut out the image patch should not be contained.
        descriptors: k x m matrix containing the patch descriptors.
            Each row vector stores the descriptor vector of the respective corner.
            with k being the number of descriptors and m being the length of a descriptor (usually patch_size**2).
            The descriptor at row i belongs to the KeyPoint at filtered_keypoints[i]
    :rtype: (List[cv2.KeyPoint], np.ndarray)
    """
    half = patch_size // 2
    filtered_keypoints = []
    descriptors = []

    h, w = img.shape

    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])

        if x - half < 0 or x + half >= w or y - half < 0 or y + half >= h:
            continue

        patch = img[y-half:y+half, x-half:x+half].copy()
        descriptor = patch.flatten().astype(np.float32)

        norm = np.linalg.norm(descriptor)
        if norm > 0:
            descriptor /= norm

        filtered_keypoints.append(kp)
        descriptors.append(descriptor)

    return filtered_keypoints, np.array(descriptors)