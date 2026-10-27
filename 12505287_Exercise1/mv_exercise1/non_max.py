#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Non-Maxima Suppression

Author: Roberto Sonzini Gobbi
MatrNr: 12505287

-> resources used: 
 - https://chatgpt.com/share/68e4e3b4-6c38-800e-82df-b1eed842e54b (to understand the functioning of np.where with indices)
 - https://chatgpt.com/share/68e4e8e6-66d4-800e-b0f2-fca58cb1ced0 
 - https://stackoverflow.com/questions/19878280/efficient-way-to-shift-2d-matrices-in-both-directions
"""

import cv2
import numpy as np

def non_max(gradients: np.array, orientations: np.array) -> np.array:
    """ Apply Non-Maxima Suppression and return an edge image.

    Filter out all the values of the gradients array which are not local maxima.
    The orientations are used to check for larger pixel values in the direction of orientation.

    :param gradients: Edge strength of the image in range [0.,1.]
    :type gradients: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param orientations: angle of gradient in range [-np.pi, np.pi]
    :type orientations: np.array with shape (height, width) with dtype = np.float32 and values in the range [-pi, pi]

    :return: Non-Maxima suppressed gradients
    :rtype: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]
    """
    ######################################################
    vertical_mask, horizontal_mask, diag_mask, rev_diag_mask = get_orientations(orientations)

    mask_max = find_if_max(gradients, vertical_mask, horizontal_mask, diag_mask, rev_diag_mask)

    edges = np.zeros(gradients.shape, dtype = np.float32)
    edges[mask_max] = gradients[mask_max]

    ######################################################

    return edges

def get_orientations(orientation):
    # I convert the orientations radiants in degrees (module 180)
    orientation_deg = np.rad2deg(orientation) % 180

    # I observe the orientations as vertical, horizontal and diagonal
    orientation_horizontal = (orientation_deg <= 22.5) | (orientation_deg > 157.5)
    orientation_diag = (orientation_deg > 22.5) & (orientation_deg <= 67.5)
    orientation_vertical = (orientation_deg > 67.5) & (orientation_deg <= 112.5)
    orientation_rev_diag = (orientation_deg > 112.5) & (orientation_deg <= 157.5)

    return orientation_vertical, orientation_horizontal, orientation_diag, orientation_rev_diag

def find_if_max(gradient, vertical_mask, horizontal_mask, diag_mask, rev_diag_mask):
    # The rolling step helps me to vectorize the operations.
    # I'm rolling the matrix in order to have at the same index all the pixels I need to evaluate the condition in a logical expression.

    neighbor_up = np.roll(gradient, -1, axis=0)
    neighbor_up[-1, :] = 0

    neighbor_down = np.roll(gradient, 1, axis=0)
    neighbor_down[0, :] = 0

    neighbor_left = np.roll(gradient, -1, axis=1)
    neighbor_left[:, -1] = 0

    neighbor_right = np.roll(gradient, 1, axis=1)
    neighbor_right[:, 0] = 0

    neighbor_diag1 = np.roll(np.roll(gradient, -1, axis=0), -1, axis=1)
    neighbor_diag1[-1, :] = 0; neighbor_diag1[:, -1] = 0

    neighbor_diag2 = np.roll(np.roll(gradient, 1, axis=0), 1, axis=1)
    neighbor_diag2[0, :] = 0 
    neighbor_diag2[:, 0] = 0

    neighbor_rev1  = np.roll(np.roll(gradient, -1, axis=0), 1, axis=1)
    neighbor_rev1[-1, :] = 0
    neighbor_rev1[:, 0] = 0

    neighbor_rev2  = np.roll(np.roll(gradient, 1, axis=0), -1, axis=1)
    neighbor_rev2[0, :] = 0
    neighbor_rev2[:, -1] = 0

    mask = np.zeros_like(gradient, dtype=bool)
    mask[vertical_mask & (gradient >= neighbor_up) & (gradient >= neighbor_down)] = True
    mask[horizontal_mask & (gradient >= neighbor_left) & (gradient >= neighbor_right)] = True
    mask[diag_mask & (gradient >= neighbor_diag1) & (gradient >= neighbor_diag2)] = True
    mask[rev_diag_mask & (gradient >= neighbor_rev1) & (gradient >= neighbor_rev2)] = True

    return mask
