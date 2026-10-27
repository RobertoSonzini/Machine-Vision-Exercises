#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Find clusters of pointcloud

Author: FILL IN
MatrNr: FILL IN
"""

from typing import Tuple

import numpy as np
import open3d as o3d
from scipy.spatial import distance
from scipy.stats import anderson
import matplotlib.pyplot as plt
from helper_functions import plot_clustering_results, silhouette_score


def kmeans(points: np.ndarray,
           n_clusters: int,
           n_iterations: int,
           max_singlerun_iterations: int,
           centers_in: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """ Find clusters in the provided data coming from a pointcloud using the k-means algorithm.

    :param points: The (down-sampled) points of the pointcloud to be clustered.
    :type points: np.ndarray with shape=(n_points, 3)

    :param n_clusters: The number of clusters to form as well as the number of centroids to generate.
    :type n_clusters: int

    :param n_iterations: Number of time the k-means algorithm will be run with different centroid seeds.
        The final results will be the best output of n_iterations consecutive runs in terms of inertia.
    :type n_iterations: int

    :param max_singlerun_iterations: Maximum number of iterations of the k-means algorithm for a single run.
    :type max_singlerun_iterations: int

    :param centers_in: Start centers of the k-means algorithm.  If centers_in = None, the centers are randomly sampled
        from input data for each iteration.
    :type centers_in: np.ndarray with shape = (n_clusters, 3) or None

    :return: (best_centers, best_labels)
        best_centers: Array with the centers of the calculated clusters (shape = (n_clusters, 3) and dtype=np.float32)
        best_labels: Array with a different label for each cluster for each point (shape = (n_points),) and dtype=int)
            The label i corresponds with the center in best_centers[i] and therefore are in the range [0, n_clusters-1]
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    ######################################################
    N = points.shape[0]

    best_inertia = np.inf
    best_centers = None
    best_labels = None

    if centers_in is not None:
        n_iterations = 1

    for run in range(n_iterations):
        if centers_in is not None:
            centers = centers_in.copy()
        else:
            indices = np.random.choice(N, n_clusters, replace=False)
            centers = points[indices, :].copy()

        for iteration in range(max_singlerun_iterations):
            distances = distance.cdist(points, centers, metric='sqeuclidean')
            labels = np.argmin(distances, axis=1)

            new_centers = np.zeros_like(centers)
            for cluster_idx in range(n_clusters):
                cluster_points = points[labels == cluster_idx]
                if len(cluster_points) > 0:
                    new_centers[cluster_idx] = cluster_points.mean(axis=0)
                else:
                    new_centers[cluster_idx] = points[np.random.choice(N)]

            if np.allclose(new_centers, centers):
                break

            centers = new_centers

        inertia = np.sum((points - centers[labels])**2)

        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers.copy()
            best_labels = labels.copy()

    return best_centers, best_labels


def iterative_kmeans(points: np.ndarray,
                     max_n_clusters: int,
                     n_iterations: int,
                     max_singlerun_iterations: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Applies the k-means algorithm multiple times and returns the best result in terms of silhouette score.

    This algorithm runs the k-means algorithm for all number of clusters until max_n_clusters. The silhouette score is
    calculated for each solution. The clusters with the highest silhouette score are returned.

    :param points: The (down-sampled) points of the pointcloud that should be clustered
    :type points: np.ndarray with shape=(n_points, 3)

    :param max_n_clusters: The maximum number of clusters that is tested.
    :type max_n_clusters: int

    :param n_iterations: Number of time each k-means algorithm will be run with different centroid seeds.
        The final results will be the best output of n_iterations consecutive runs in terms of inertia.
    :type n_iterations: int

    :param max_singlerun_iterations: Maximum number of iterations of each k-means algorithm for a single run.
    :type max_singlerun_iterations: int

    :return: (best_centers, best_labels)
        best_centers: Array with the centers of the calculated clusters (shape = (n_clusters, 3) and dtype=np.float32)
        best_labels: Array with a different label for each cluster for each point (shape = (n_points),) and dtype=int)
            The label i corresponds with the center in best_centers[i] and therefore are in the range [0, n_clusters-1]
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    ######################################################
    best_score = -np.inf
    best_centers = None
    best_labels = None

    for k in range(2, max_n_clusters + 1):

        centers, labels = kmeans(points,
                                          k,
                                          n_iterations=n_iterations,
                                          max_singlerun_iterations=max_singlerun_iterations)

        score = silhouette_score(points, centers, labels)

        if score > best_score:
            best_score = score
            best_centers = centers.astype(np.float32)
            best_labels = labels.astype(int)

    return best_centers, best_labels

def gmeans(points: np.ndarray,
           tolerance: float,
           max_singlerun_iterations: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Find clusters in the provided data coming from a pointcloud using the g-means algorithm.

    The algorithm was proposed by Hamerly, Greg, and Charles Elkan. "Learning the k in k-means." Advances in neural
    information processing systems 16 (2003).

    :param points: The (down-sampled) points of the pointcloud to be clustered
    :type points: np.ndarray with shape=(n_points, 3)

    :param max_singlerun_iterations: Maximum number of iterations of the k-means algorithm for a single run.
    :type max_singlerun_iterations: int

    :return: (best_centers, best_labels)
        best_centers: Array with the centers of the calculated clusters (shape = (n_clusters, 3) and dtype=np.float32)
        best_labels: Array with a different label for each cluster for each point (shape = (n_points,) and dtype=int)
            The label i corresponds with the center in best_centers[i] and therefore are in the range [0, n_clusters-1]
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    MIN_CLUSTER_SIZE = 10
    ######################################################
    # Write your own code here

    # 1. partire da 1 cluster
    # 2. calcoliamo il centroide come media di tutti i punti.
    # centers = np.mean(points, axis=0).reshape(1,3)
    centers = np.mean(points, axis=0, keepdims=True)

    iterations = 0
    new_centers = True
    
    # Loop finchè non aggiungo più centers:
    while new_centers and iterations < max_singlerun_iterations:
        iterations += 1

        # calcoliamo il cluster con k-means.
        kmeans_centers, kmeans_labels = kmeans(
            points=points,
            n_clusters=centers.shape[0],
            n_iterations=1,
            max_singlerun_iterations=max_singlerun_iterations,
            centers_in=centers
        )

        # calcoliamo i children (numpy)
        updated_centers = []
        updated_labels = np.zeros(points.shape[0], dtype=int)
        new_center_idx = 0

        for i in range(kmeans_centers.shape[0]):
            cluster_mask = (kmeans_labels == i)
            cluster_points = points[cluster_mask]
            center = kmeans_centers[i]

            cov = np.cov(cluster_points.T)

            eigenvalues, eigenvectors = np.linalg.eig(cov)
            idx_max = np.argmax(eigenvalues)
            principal_vector = eigenvectors[:, idx_max]
            principal_vector /= np.linalg.norm(principal_vector)
            lambda_max = eigenvalues[idx_max]

            offset = principal_vector * np.sqrt(2 * lambda_max / np.pi)
            child1 = center + offset
            child2 = center - offset

            # Eseguiamo k-means
            # calcoliamo il cluster con k-means.
            child_centers, child_labels = kmeans(
                points=cluster_points,
                n_clusters=2,
                n_iterations=1,
                max_singlerun_iterations=max_singlerun_iterations,
                centers_in=np.vstack([child1, child2])
            )

            # proiezione
            v = child_centers[0] - child_centers[1]
            norm_v2 = np.dot(v, v)
            ''' x_projected = np.dot(cluster_points, v) / norm_v2

            # per ogni cluster facciamo il test statistico.
            # Perform the Anderson -Darling test for normality with the projected points
            estimation , critical , _ = anderson(x_projected)
            if estimation <= critical [-1] * tolerance:
                is_gaussian = True
            else:
                is_gaussian = False'''
            x_projected = np.dot(cluster_points, v) / norm_v2

            if (
                x_projected.shape[0] < MIN_CLUSTER_SIZE or
                    not np.isfinite(x_projected).all() or
                    np.std(x_projected) == 0):
                # Cluster non splittabile → trattalo come gaussiano
                is_gaussian = True
            else:
                estimation, critical, _ = anderson(x_projected)
                is_gaussian = estimation <= critical[-1] * tolerance

            if is_gaussian:
                updated_centers.append(center)
                updated_labels[cluster_mask] = new_center_idx
                new_center_idx += 1
            else:
                updated_centers.extend(child_centers)
                updated_labels[cluster_mask] = child_labels + new_center_idx
                new_center_idx += 2
                new_centers = True

        centers = np.array(updated_centers, dtype=np.float32)
        labels = updated_labels.copy()

    return centers, labels

def dbscan(points: np.ndarray,
           eps: float = 0.05,
           min_samples: int = 10) -> np.ndarray:
    """ Find clusters in the provided data coming from a pointcloud using the DBSCAN algorithm.

    The algorithm was proposed in Ester, Martin, et al. "A density-based algorithm for discovering clusters in large
    spatial databases with noise." kdd. Vol. 96. No. 34. 1996.

    :param points: The (down-sampled) points of the pointcloud to be clustered
    :type points: np.ndarray with shape=(n_points, 3)

    :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :type eps: float

    :param min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core
        point. This includes the point itself.
    :type min_samples: float

    :return: Labels array with a different label for each cluster for each point (shape = (n_points,) and dtype=int)
            The label -1 is assigned to points that are considered to be noise.
    :rtype: np.ndarray
    """
    ######################################################
    labels = np.zeros(points.shape[0])
    cluster_id = 0

    for i in range(points.shape[0]):
        if labels[i] != 0:
            continue

        point = points[i:i+1]
        distances = distance.cdist(point, points, metric='euclidean')[0]
        neighbors_idx = np.where(distances <= eps)[0]

        if len(neighbors_idx) < min_samples:
            labels[i] = -1
            continue

        cluster_id += 1
        labels[i] = cluster_id

        seeds = set(neighbors_idx)
        seeds.discard(i)

        while seeds:
            current = seeds.pop()
            
            if labels[current] == -1:
                labels[current] = cluster_id
            if labels[current] != 0:
                continue

            labels[current] = cluster_id

            current_point = points[current:current+1]
            current_distances = distance.cdist(current_point, points, metric='euclidean')[0]
            current_neighbors = np.where(current_distances <= eps)[0]

            if len(current_neighbors) >= min_samples:
                for idx in current_neighbors:
                    if labels[idx] == 0:
                        seeds.add(idx)

    return labels
