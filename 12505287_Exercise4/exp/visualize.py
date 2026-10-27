import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path

from fit_plane import fit_plane
from clustering import *
from helper_functions import plot_dominant_plane


def plot_clustering_subplots(points, labels_dict):
    fig = plt.figure(figsize=(15, 12))
    n = len(labels_dict)

    for i, (name, labels) in enumerate(labels_dict.items(), start=1):
        ax = fig.add_subplot(2, 2, i, projection="3d")

        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c=labels, s=2)
        ax.set_title(name)
        ax.set_axis_off()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    pointcloud_idx = 1

    use_voxel_downsampling = True
    voxel_size = 0.01
    uniform_every_k_points = 10
    confidence = 0.90
    inlier_threshold = 0.02
    kmeans_n_clusters = 6
    kmeans_iterations = 25
    max_singlerun_iterations = 100
    iterative_kmeans_max_clusters = 10
    gmeans_tolerance = 10

    dbscan_eps = 0.05
    dbscan_min_points = 15

    current_path = Path(__file__).parent
    file_path = current_path / f"pointclouds/image00{pointcloud_idx}.pcd"

    pcd = o3d.io.read_point_cloud(str(file_path),
                                  remove_nan_points=True,
                                  remove_infinite_points=True)

    if not pcd.has_points():
        raise FileNotFoundError(f"Impossibile caricare {file_path}")

    points_np = np.asarray(pcd.points)
    valid_idx = np.where(~np.all(points_np == 0, axis=1))[0]
    pcd = pcd.select_by_index(valid_idx)

    if use_voxel_downsampling:
        pcd_sampled = pcd.voxel_down_sample(voxel_size=voxel_size)
    else:
        pcd_sampled = pcd.uniform_down_sample(every_k_points=uniform_every_k_points)

    plane_model, best_inliers = fit_plane(
        pcd=pcd_sampled,
        confidence=confidence,
        inlier_threshold=inlier_threshold
    )

    inlier_indices = np.nonzero(best_inliers)[0]

    mask = np.full(len(pcd_sampled.points), False, dtype=bool)
    mask[inlier_indices] = True

    scene_pcd = pcd_sampled.select_by_index(inlier_indices, invert=True)

    points = np.asarray(scene_pcd.points, dtype=np.float32)

    centers_km, labels_km = kmeans(points,
                                   n_clusters=kmeans_n_clusters,
                                   n_iterations=kmeans_iterations,
                                   max_singlerun_iterations=max_singlerun_iterations)

    centers_ikm, labels_ikm = iterative_kmeans(points,
                                               max_n_clusters=iterative_kmeans_max_clusters,
                                               n_iterations=kmeans_iterations,
                                               max_singlerun_iterations=max_singlerun_iterations)

    centers_gm, labels_gm = gmeans(points,
                                   tolerance=gmeans_tolerance,
                                   max_singlerun_iterations=max_singlerun_iterations)

    labels_db = dbscan(points,
                       eps=dbscan_eps,
                       min_samples=dbscan_min_points)

    labels_dict = {
        "K-Means": labels_km,
        "Iterative K-Means": labels_ikm,
        "G-Means": labels_gm,
        "DBSCAN": labels_db
    }

    plot_clustering_subplots(points, labels_dict)