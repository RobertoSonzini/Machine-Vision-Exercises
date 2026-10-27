#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Find the projective transformation matrix (homography) between from a source image to a target image.

Author: FILL IN
MatrNr: FILL IN
"""

import numpy as np
from helper_functions import *

def find_homography_ransac(source_points: np.ndarray,
                           target_points: np.ndarray,
                           confidence: float,
                           inlier_threshold: float) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return estimated transforamtion matrix of source_points in the target image given matching points

    Return the projective transformation matrix for homogeneous coordinates. It uses the RANSAC algorithm with the
    Least-Squares algorithm to minimize the back-projection error and be robust against outliers.
    Requires at least 4 matching points.

    :param source_points: Array of points. Each row holds one point from the source (object) image [x, y]
    :type source_points: np.ndarray with shape (n, 2)

    :param target_points: Array of points. Each row holds one point from the target (scene) image [x, y].
    :type target_points: np.ndarray with shape (n, 2)

    :param confidence: Solution Confidence (in percent): Likelihood of all sampled points being inliers.
    :type confidence: float

    :param inlier_threshold: Max. Euclidean distance of a point from the transformed point to be considered an inlier
    :type inlier_threshold: float

    :return: (homography, inliers, num_iterations)
        homography: The projective transformation matrix for homogeneous coordinates with shape (3, 3)
        inliers: Is True if the point at the index is an inlier. Boolean array with shape (n,)
        num_iterations: The number of iterations that were needed for the sample consensus
    :rtype: Tuple[np.ndarray, np.ndarray, int]
    """
    ######################################################
    n_points = source_points.shape[0]

    best_suggested_homography = np.eye(3)
    best_inliers = np.full(shape=len(target_points), fill_value=False, dtype=bool)
    num_iterations = 0
    
    m = 4
    epsilon = m / n_points

    f = (1 - epsilon**m)**num_iterations

    while f >= 1 - confidence:
        indices = np.random.choice(n_points, 4, replace=False)
        source_samples = source_points[indices]
        target_samples = target_points[indices]

        H = find_homography_leastsquares(source_samples, target_samples)

        source_h = np.hstack([source_points, np.ones((n_points,1))])
        projected = (H @ source_h.T).T

        projected /= projected[:, 2:3] # conversion in cartesian coordinate -> divide by w'

        distances = np.linalg.norm(projected[:, :2] - target_points, axis=1)

        inliers = distances < inlier_threshold

        if (inliers.sum() > best_inliers.sum()):
            best_inliers = inliers
            best_suggested_homography = H

            epsilon = best_inliers.sum() / n_points

        num_iterations += 1
        f = (1 - epsilon**m)**num_iterations

    best_suggested_homography = find_homography_leastsquares(
            source_points[best_inliers],
            target_points[best_inliers]
        )

    source_h = np.hstack([source_points, np.ones((n_points, 1))])
    projected = (best_suggested_homography @ source_h.T).T
    projected /= projected[:, 2:3]

    distances = np.linalg.norm(projected[:, :2] - target_points, axis=1)
    best_inliers = distances < inlier_threshold

    ######################################################
    return best_suggested_homography, best_inliers, num_iterations


def find_homography_leastsquares(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    """Return projective transformation matrix of source_points in the target image given matching points

    Return the projective transformation matrix for homogeneous coordinates. It uses the Least-Squares algorithm to
    minimize the back-projection error with all points provided. Requires at least 4 matching points.

    :param source_points: Array of points. Each row holds one point from the source image (object image) as [x, y]
    :type source_points: np.ndarray with shape (n, 2)

    :param target_points: Array of points. Each row holds one point from the target image (scene image) as [x, y].
    :type target_points: np.ndarray with shape (n, 2)

    :return: The projective transformation matrix for homogeneous coordinates with shape (3, 3)
    :rtype: np.ndarray with shape (3, 3)
    """
    ######################################################
    n = source_points.shape[0]
    A = []
    b = []

    for i in range(n):
        x, y = source_points[i]
        xp, yp = target_points[i]

        A.append([x, y, 1, 0, 0, 0, -xp * x, -xp * y])
        b.append(xp)

        A.append([0, 0, 0, x, y, 1, -yp * x, -yp * y])
        b.append(yp)

    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64)

    homography, _, _, _ = np.linalg.lstsq(A, b)

    homography = np.append(homography, 1).reshape(3, 3)

    return homography
