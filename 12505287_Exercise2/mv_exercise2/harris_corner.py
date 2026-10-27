#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Corner detection with the Harris corner detector

Author: Roberto Sonzini Gobbi
MatrNr: 12505287
"""
import numpy as np
import cv2

from typing import List

from helper_functions import non_max

def harris_corner(img: np.ndarray,
                  sigma1: float,
                  sigma2: float,
                  k: float,
                  threshold: float) -> List[cv2.KeyPoint]:
    """ Detect corners using the Harris corner detector

    In this function, corners in a grayscale image are detected using the Harris corner detector.
    They are returned in a list of OpenCV KeyPoints (https://docs.opencv.org/4.x/d2/d29/classcv_1_1KeyPoint.html).
    Each KeyPoint includes the attributes, pt (position), size, angle, response. The attributes size and angle are not
    relevant for the Harris corner detector and can be set to an arbitrary value. The response is the result of the
    Harris corner formula.

    :param img: Grayscale input image
    :type img: np.ndarray with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param sigma1: Sigma for the first Gaussian filtering
    :type sigma1: float

    :param sigma2: Sigma for the second Gaussian filtering
    :type sigma2: float

    :param k: Coefficient for harris formula
    :type k: float

    :param threshold: Minimum value of the Harris response (R) for a corner to be detected.
    :type threshold: float

    :return: keypoints:
        corners: List of cv2.KeyPoints containing all detected corners after thresholding and non-maxima suppression.
            Each keypoint has the attribute pt[x, y], size, angle, response.
                pt: The x, y position of the detected corner in the OpenCV coordinate convention.
                size: The size of the relevant region around the keypoint. Not relevant for Harris and is set to 1.
                angle: The direction of the gradient in degree. Relative to image coordinate system (clockwise).
                response: Result of the Harris corner formula R = det(M) - k*trace(M)**2
    :rtype: List[cv2.KeyPoint]

    """
    
    ######################################################
    kernel_width = int(2 * np.ceil(3*sigma1) + 1)
    kernel = cv2.getGaussianKernel(kernel_width , sigma1)
    kernel = np.outer(kernel, kernel.transpose())

    sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)

    sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=np.float32)

    gradient_gauss_x = cv2.filter2D(kernel, ddepth=-1, kernel=sobel_x)
    gradient_gauss_y = cv2.filter2D(kernel, ddepth=-1, kernel=sobel_y)

    Ix = cv2.filter2D(img, -1, gradient_gauss_x)
    Iy = cv2.filter2D(img, -1, gradient_gauss_y)

    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    kernel_width = int(2 * np.ceil(3 * sigma2) + 1)
    kernel = cv2.getGaussianKernel(kernel_width, sigma2)
    kernel = np.outer(kernel, kernel.transpose())

    Sxx = cv2.filter2D(Ixx, -1, kernel)
    Syy = cv2.filter2D(Iyy, -1, kernel)
    Sxy = cv2.filter2D(Ixy, -1, kernel)

    det_M = Sxx * Syy - Sxy**2
    trace_M = Sxx + Syy
    R = det_M - k * (trace_M**2)

    R = (R - np.min(R)) / (np.max(R) - np.min(R))

    R[R < threshold] = 0

    h_nonmax = non_max(R)

    corners_y, corners_x = np.nonzero(h_nonmax)
    corners_d = R[corners_y, corners_x]
    corners = np.stack([corners_y, corners_x, corners_d], axis=1)

    keypoints = []
    for y, x, resp in zip(corners_y, corners_x, corners_d):
        kp = cv2.KeyPoint(
            x=float(x),
            y=float(y),
            size=1,
            angle=-1,
            response=float(resp)
        )
        keypoints.append(kp)

    ######################################################
    return keypoints