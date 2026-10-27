#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Hysteresis thresholding

Author: Roberto Sonzini Gobbi
MatrNr: 12505287
"""

import cv2
import numpy as np


def hyst_thresh(edges_in: np.array, low: float, high: float) -> np.array:
    """ Apply hysteresis thresholding.

    Apply hysteresis thresholding to return the edges as a binary image. All connected pixels with value > low are
    considered a valid edge if at least one pixel has a value > high.

    :param edges_in: Edge strength of the image in range [0.,1.]
    :type edges_in: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param low: Value below all edges are filtered out
    :type low: float in range [0., 1.]

    :param high: Value which a connected element has to contain to not be filtered out
    :type high: float in range [0., 1.]

    :return: Binary edge image
    :rtype: np.array with shape (height, width) with dtype = np.float32 and values either 0 or 1
    """
    ######################################################

    # I create the mask to have the edges where the value > low
    low_mask = np.where(edges_in > low, 1, 0).astype(np.uint8)

    # To not do many conversions of the matrix edges_in, i can operate direcly on the mask.
    _, labels = cv2.connectedComponents(low_mask, connectivity=8)

    # for each object we see if the condition is satisfied and if the can mantain the border.
    mask = np.zeros(edges_in.shape).astype(bool)
    for label in range(1, labels.max() + 1):
        if label != 0: 
            label_mask = (labels == label)
            values = edges_in[label_mask]

            if (values > high).any():
                mask = mask | label_mask

    bitwise_img = mask.astype(np.float32)

    ######################################################
    return bitwise_img
