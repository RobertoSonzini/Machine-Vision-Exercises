#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Fit Plane in pointcloud

Author: Roberto Sonzini Gobbi
MatrNr: 12505287
"""

from typing import Tuple

import copy

import numpy as np
import open3d as o3d

def plane_from_points(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    n = np.cross(v1, v2)
    norm = np.linalg.norm(n)
    if norm < 1e-8:
        return None
    n /= norm
    a, b, c = n
    d = -np.dot(n, p1)
    return np.array([a, b, c, d])

def point_plane_distance(plane, pts):
    return np.abs(pts @ plane[:3] + plane[3])

def fit_plane(pcd: o3d.geometry.PointCloud,
              confidence: float,
              inlier_threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """ Find dominant plane in pointcloud with sample consensus.

    Detect a plane in the input pointcloud using sample consensus. The number of iterations is chosen adaptively.
    The concrete error function is given as an parameter.

    :param pcd: The (down-sampled) pointcloud in which to detect the dominant plane
    :type pcd: o3d.geometry.PointCloud (http://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html)

    :param confidence: Solution Confidence (in percent): Likelihood of all sampled points being inliers.
    :type confidence: float

    :param inlier_threshold: Max. distance of a point from the plane to be considered an inlier (in meters)
    :type inlier_threshold: float

    :return: (best_plane, best_inliers)
        best_plane: Array with the coefficients of the plane equation ax+by+cz+d=0 (shape = (4,))
        best_inliers: Boolean array with the same length as pcd.points. Is True if the point at the index is an inlier
    :rtype: tuple (np.ndarray[a,b,c,d], np.ndarray)
    """
    ######################################################
    points = np.asarray(pcd.points)
    N = points.shape[0]

    best_plane = np.array([0., 0., 1., 0.])
    best_inliers = np.full(N, False)
    num_iterations = 0

    s = 3
    epsilon = s / N
    f = (1 - epsilon**s)**num_iterations
    max_iterations = 10000

    while f >= 1 - confidence:
        idx = np.random.choice(N, s, replace=False)
        p1, p2, p3 = points[idx]

        plane = plane_from_points(p1, p2, p3)
        if plane is None:
            num_iterations += 1
            continue

        distances = point_plane_distance(plane, points)
        inliers = distances < inlier_threshold

        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_plane = plane

            epsilon = best_inliers.sum() / N

        num_iterations += 1
        f = (1 - epsilon**s)**num_iterations

    inlier_pts = points[best_inliers]
    A = inlier_pts
    b = np.ones(A.shape[0])
    normal, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, b_, c = normal
    d = -1.0
    refined_plane = np.array([a, b_, c, d])

    if refined_plane[2] < 0:
        refined_plane *= -1

    final_inliers = point_plane_distance(refined_plane, points) < inlier_threshold

    return refined_plane, final_inliers
