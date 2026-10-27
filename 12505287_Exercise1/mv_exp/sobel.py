#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Edge detection with the Sobel filter

Author: Roberto Sonzini Gobbi
MatrNr: 12505287
"""

import cv2
import numpy as np


def sobel(img: np.array) -> (np.array, np.array):
    """ Apply the Sobel filter to the input image and return the gradient and the orientation.

    :param img: Grayscale input image
    :type img: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]
    :return: (gradient, orientation): gradient: the normalized edge strength of the image in range [0.,1.],
                                      orientation: angle of gradient in range [-np.pi, np.pi]
    :rtype: (np.array, np.array)
    """
    ######################################################

    sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)

    sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=np.float32)

    gradient_x = cv2.filter2D(img, ddepth=-1, kernel=sobel_x)
    gradient_y = cv2.filter2D(img, ddepth=-1, kernel=sobel_y)

    gradient = np.sqrt(gradient_x**2 + gradient_y**2)

    G_min = np.min(gradient)
    G_max = np.max(gradient)

    gradient = (gradient - G_min) / (G_max - G_min  + 1e-8)

    orientation = np.arctan2(gradient_y, gradient_x)

    gradient = gradient.astype(np.float32)
    orientation = orientation.astype(np.float32)

    ######################################################
    return gradient, orientation