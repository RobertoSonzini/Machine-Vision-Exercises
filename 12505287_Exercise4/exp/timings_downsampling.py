import time
import numpy as np
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt

from fit_plane import fit_plane
from clustering import *

if __name__ == "__main__":

    pc_files = [
        "pointclouds/image000.pcd",
        "pointclouds/image009.pcd"
    ]

    # Two downsampling levels
    downsampling_levels = [
        {"name": "voxel_0.01", "voxel_size": 0.01},
        {"name": "voxel_0.02", "voxel_size": 0.02},
    ]

    confidence = 0.90
    inlier_threshold = 0.02

    kmeans_n_clusters = 6
    kmeans_iterations = 25
    max_singlerun_iterations = 100
    iterative_kmeans_max_clusters = 10
    gmeans_tolerance = 10
    dbscan_eps = 0.05
    dbscan_min_points = 15

    results = []

    for pc_idx, pc_file in enumerate(pc_files):
        print(f"Processing {pc_file} ...")

        # Load pointcloud
        pcd = o3d.io.read_point_cloud(pc_file, remove_nan_points=True, remove_infinite_points=True)
        points_np = np.asarray(pcd.points)
        valid_idx = np.where(~np.all(points_np == 0, axis=1))[0]
        pcd = pcd.select_by_index(valid_idx)

        for ds in downsampling_levels:
            row = {
                "PointCloud": f"PC{pc_idx}",
                "Downsampling": ds["name"]
            }

            pcd_sampled = pcd.voxel_down_sample(voxel_size=ds["voxel_size"])

            plane_model, best_inliers = fit_plane(
                pcd_sampled, confidence=confidence, inlier_threshold=inlier_threshold
            )
            inlier_indices = np.nonzero(best_inliers)[0]
            scene_pcd = pcd_sampled.select_by_index(inlier_indices, invert=True)
            points = np.asarray(scene_pcd.points, dtype=np.float32)

            start = time.perf_counter()
            kmeans(points, kmeans_n_clusters, kmeans_iterations, max_singlerun_iterations)
            row["K-Means"] = time.perf_counter() - start

            start = time.perf_counter()
            iterative_kmeans(points, iterative_kmeans_max_clusters,
                             kmeans_iterations, max_singlerun_iterations)
            row["Iterative K-Means"] = time.perf_counter() - start

            start = time.perf_counter()
            gmeans(points, gmeans_tolerance, max_singlerun_iterations)
            row["G-Means"] = time.perf_counter() - start

            start = time.perf_counter()
            dbscan(points, dbscan_eps, dbscan_min_points)
            row["DBSCAN"] = time.perf_counter() - start

            results.append(row)

    df = pd.DataFrame(results)
    print("\n=== Runtime Results ===")
    print(df.round(4))

    algorithms = ["K-Means", "Iterative K-Means", "G-Means", "DBSCAN"]

    df_grouped = (
        df
        .groupby("Downsampling")[algorithms]
        .mean()
        .T
    )

    df_grouped.plot(
        kind="bar",
        figsize=(10, 6)
    )

    plt.ylabel("Execution Time [s]")
    plt.title("Average runtime per algorithm for different downsampling levels")
    plt.xticks(rotation=0)
    plt.legend(title="Downsampling")
    plt.tight_layout()
    plt.show()



