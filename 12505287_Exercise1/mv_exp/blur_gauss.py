#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Blur the input image with Gaussian filter kernel

Author: Roberto Sonzini Gobbi
MatrNr: 12505287

sources: https://chatgpt.com/share/68ea70ca-d5b0-800e-b98f-9f86c3b3b1a0 (for plots)
"""

import cv2
import numpy as np
from math import ceil

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from helper_functions import *

def plot_kernel(xx, yy, kernel, sigma):
    plt.figure(figsize=(5,5))
    plt.imshow(kernel, cmap='viridis', interpolation='none')
    plt.title(f"Gaussian Kernel (sigma={sigma})")
    plt.colorbar(label='Weight')
    plt.show()

    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_surface(xx, yy, kernel, cmap='viridis')
    ax.set_title(f"Gaussian Kernel 3D (sigma={sigma})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Weight")
    
    plt.show()
    

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

    kernel_width = 2 * ceil(3 * sigma) + 1
    xx, yy, kernel = gaussian_kernel(kernel_width, sigma)

    # plot_kernel(xx, yy, kernel, sigma)

    img_blur = cv2.filter2D(img, ddepth=-1, kernel=kernel)

    ######################################################
    return img_blur

def gaussian_kernel(size, sigma):
    # I define an equally spaced array.
    ax = np.linspace(-(size // 2), size // 2, size)
    # I create 2 matrices from the axes to get, in combination, the set of coordinates to build a kernel matrix from a coninuous function.
    xx, yy = np.meshgrid(ax, ax)
    # I compute the kernel matrix using the gaussian mathematical expression.
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2)) / (sigma * np.sqrt((2. * np.pi)))
    return xx, yy, kernel / np.sum(kernel)