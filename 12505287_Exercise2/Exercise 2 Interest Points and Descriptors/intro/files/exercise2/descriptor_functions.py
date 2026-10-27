#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Descriptor functions

Author: Roberto Sonzini Gobbi
MatrNr: 12505287
"""
from typing import Callable

import numpy as np
import cv2

from helper_functions import circle_mask

def patch_basic(patch: np.ndarray) -> np.ndarray:
    """ Return the basic descriptor as a vector

    :param patch: Patch of the image around a corner
    :type patch: np.ndarray with shape (patch_size, patch_size)

    :return: Descriptor
    :rtype: np.ndarray with shape (1, patch_size^2)
    """
    ######################################################
    pach_copy = patch.copy()
    h, w = pach_copy.shape
    cell_size = h // 4
    n_bins = 8

    dx = np.array([-1, 0, 1]).reshape(1,3)
    dy = dx.T

    Gx = cv2.filter2D(pach_copy, -1, dx)
    Gy = cv2.filter2D(pach_copy, -1, dy)

    magnitude = np.sqrt(Gx**2 + Gy**2)
    angle = np.degrees(np.arctan2(Gy, Gx))

    sigma = h / 2.0
    y, x = np.mgrid[0:h, 0:w]
    center = (h-1)/2
    gaussian = np.exp(-((x-center)**2 + (y-center)**2) / (2*sigma*sigma))
    magnitude *= gaussian

    cells_mag = magnitude.reshape(4, cell_size, 4, cell_size)
    cells_ang = angle.reshape(4, cell_size, 4, cell_size)

    cells_mag = cells_mag.transpose(0,2,1,3).reshape(-1, cell_size*cell_size)
    cells_ang = cells_ang.transpose(0,2,1,3).reshape(-1, cell_size*cell_size)

    descriptor = []
    for mag, ang in zip(cells_mag, cells_ang):
        hist, _ = np.histogram(ang, bins=n_bins, range=(0,360), weights=mag)
        descriptor.extend(hist)

    descriptor = np.array(descriptor, dtype=np.float32)

    return np.array(descriptor)  # Replace this line

    ######################################################

def patch_norm(patch: np.ndarray) -> np.ndarray:
    """ Return the normalized basic descriptor as a vector

    :param patch: Patch of the image around a corner
    :type patch: np.ndarray with shape (patch_size, patch_size)

    :return: Descriptor
    :rtype: np.ndarray with shape (1, patch_size^2)
    """
    ######################################################
    # Write your own code here
    return np.zeros((1, patch.size))  # Replace this line

    ######################################################

def patch_sort(patch: np.ndarray) -> np.ndarray:
    """ Return the normalized and sorted basic descriptor as a vector

    :param patch: Patch of the image around a corner
    :type patch: np.ndarray with shape (patch_size, patch_size)

    :return: Descriptor
    :rtype: np.ndarray with shape (1, patch_size^2)
    """
    ######################################################
    # Write your own code here
    return np.zeros((1, patch.size))  # Replace this line

    ######################################################

def patch_sort_circle(patch: np.ndarray) -> np.ndarray:
    """ Return the normalized and sorted basic descriptor as a vector

    :param patch: Patch of the image around a corner
    :type patch: np.ndarray with shape (patch_size, patch_size)

    :return: Descriptor
    :rtype: np.ndarray with shape (1, patch_size^2)
    """
    ######################################################
    # Write your own code here
    return np.zeros((1, patch.size))  # Replace this line

    ######################################################

def block_orientations(patch: np.ndarray) -> np.ndarray:
    """ Compute orientation-histogram based descriptor from a patch

    Orientation histograms from 16 4 x 4 blocks of the patch, concatenated in row major order (1 x 128).
    Each orientation histogram should consist of 8 bins in the range [-pi, pi], each bin being weighted by the sum of
    gradient magnitudes of pixel orientations assigned to that bin.

    :param patch: Patch of the image around a corner
    :type patch: np.ndarray with shape (16, 16)

    :return: Orientation-histogram based Descriptor
    :rtype: np.ndarray with shape (1, 128)
    """
    ######################################################
    # Write your own code here
    return np.zeros((1, 128))  # Replace this line

    ######################################################

def compute_descriptors(descriptor_func: Callable,
                        img: np.ndarray,
                        locations: np.ndarray,
                        patch_size: int) -> (np.ndarray, np.ndarray):
    """ Calculate the given descriptor using descriptor_func on patches of the image, centred on the locations provided

    :param descriptor_func: Descriptor to compute at each location
    :type descriptor_func: function

    :param img: Grayscale input image
    :type img: np.ndarray with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param locations: Locations at which to compute the descriptors (n x 2)
    :type locations: np.ndarray with the shape (n x 2). First column is y (row), second is x (column).

    :param patch_size: Value defining the width and height of the patch around each location
        to pass to the descriptor function.
    :type patch_size: int

    :return: (interest_points, descriptors):
        interest_points: k x 2 array containing the image coordinates [y,x] of the corners.
            Locations too close to the image boundary to cut out the image patch should not be contained.
        descriptors: k x patch_size^2 matrix containing the patch descriptors.
            Each row vector stores the concatenated column vectors of the image patch around each corner.
            Corners too close to the image boundary to cut out the image patch should not be contained.
    :rtype: (np.ndarray, np.ndarray)
    """
    ######################################################

    half_size = patch_size // 2
    interest_points = []
    descriptors = []

    for y, x in locations:
        y, x = int(y), int(x)
        if y - half_size < 0 or y + half_size > img.shape[0] or x - half_size < 0 or x + half_size > img.shape[1]:
            continue 

        patch = img[y - half_size : y + half_size,
                    x - half_size : x + half_size]

        descriptor = descriptor_func(patch)

        interest_points.append([y, x])
        descriptors.append(descriptor)

    return np.array(interest_points), np.array(descriptors)

    ######################################################
