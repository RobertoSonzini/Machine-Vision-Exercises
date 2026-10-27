#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Blur the input image with Gaussian filter kernel

Author: Roberto Sonzini Gobbi
MatrNr: 12505287
"""

import cv2
import numpy as np
from math import ceil

from helper_functions import *

def blur_gauss(img: np.array, sigma: float) -> np.array:
    """ Blur the input image with a Gaussian filter with standard deviation of sigma.

    :param img: Grayscale input image
    :type img: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param sigma: The standard deviation of the Gaussian kernel
    :type sigma: float

    :return: Blurred image
    :rtype: np.array with shape (height, width) with dtype = np.float32 and values in the range [0.,1.]
    """
    ######################################################
    # Write your own code here

    kernel_width = 2 * ceil(3 * sigma) + 1
    kernel = gaussian_kernel(kernel_width, sigma)
    img_blur = cv2.filter2D(img, ddepth=-1, kernel=kernel)

    ######################################################
    return img_blur

def gaussian_kernel(size, sigma):
    # I define an equally spaced array.
    ax = np.linspace(-(size // 2), size // 2, size)
    # I create 2 matrices from the axes to get, in combination, the set of coordinates to build a kernel matrix from a coninuous function.
    xx, yy = np.meshgrid(ax, ax)
    # I compute the kernel matrix using the gaussian mathematical expression.
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)