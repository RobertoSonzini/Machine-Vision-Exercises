#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Corner detection with the Harris corner detector

Author: Roberto Sonzini Gobbi
MatrNr: 12505287
"""
import numpy as np
import cv2


def harris_corner(img, sigma1, sigma2, k, threshold):
    """ Detect corners using the Harris corner detector

    :param img: Grayscale input image
    :type img: np.ndarray with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param sigma1: Sigma for the first Gaussian filtering
    :type sigma1: float

    :param sigma2: Sigma for the second Gaussian filtering
    :type sigma2: float

    :param k: Coefficient for harris formula
    :type k: float

    :param threshold: corner threshold
    :type threshold: float

    :return: (i_xx, i_yy, i_xy, g_xx, g_yy, g_xy, h_dense, h_nonmax, corners):
        i_xx: squared input image filtered with derivative of gaussian in x-direction
        i_yy: squared input image filtered with derivative of gaussian in y-direction
        i_xy: Multiplication of input image filtered with derivative of gaussian in x- and y-direction
        g_xx: i_xx filtered by larger gaussian
        g_yy: i_yy filtered by larger gaussian
        g_xy: i_xy filtered by larger gaussian
        h_dense: Result of harris calculation for every pixel. Array of same size as input image.
            Values normalized to 0-1
        h_nonmax: Binary mask of non-maxima suppression. Array of same size as input image.
            1 where values are NOT suppressed, 0 where they are.
        corners: n x 3 array containing all detected corners after thresholding and non-maxima suppression.
            Every row vector represents a corner with the elements [y, x, d]
            (d is the result of the harris calculation)
    :rtype: (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)

    """

    ######################################################
    
    kernel_width = int(2 * np.ceil(3*sigma1) + 1)
    kernel = cv2.getGaussianKernel(kernel_width , sigma1)
    gauss = np.outer(kernel, kernel.transpose())

    dx = np.array([-1, 0, 1]).reshape(1,3)
    dy = dx.T

    Gx = cv2.filter2D(gauss, -1, dx)
    Gy = cv2.filter2D(gauss, -1, dy)

    Ix = cv2.filter2D(img, -1, Gx)
    Iy = cv2.filter2D(img, -1, Gy)

    i_xx = Ix * Ix
    i_yy = Iy * Iy
    i_xy = Ix * Iy

    kernel_width_2 = int(2 * np.ceil(3*sigma2) + 1)
    kernel2 = cv2.getGaussianKernel(kernel_width_2, sigma2)
    gauss2 = np.outer(kernel2, kernel2)

    g_xx = cv2.filter2D(i_xx, -1, gauss2)
    g_yy = cv2.filter2D(i_yy, -1, gauss2)
    g_xy = cv2.filter2D(i_xy, -1, gauss2)

    det_M = g_xx * g_yy - g_xy**2
    trace_M = g_xx + g_yy

    h_dense = det_M - k * (trace_M**2)

    h_dense = (h_dense - np.min(h_dense)) / (np.max(h_dense) - np.min(h_dense))

    print(img.shape)
    print(h_dense.shape)

    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0

    dilation = cv2.dilate(h_dense , kernel) 
    h_nonmax = (h_dense > dilation) & (h_dense > threshold)

    corners_y, corners_x = np.nonzero(h_nonmax)
    corners_d = h_dense[corners_y, corners_x]
    corners = np.stack([corners_y, corners_x, corners_d], axis=1)

    return i_xx, i_yy, i_xy, g_xx, g_yy, g_xy, h_dense, h_nonmax, corners

    ######################################################
