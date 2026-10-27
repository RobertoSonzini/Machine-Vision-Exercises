import time
import numpy as np
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt

from fit_plane import fit_plane
from clustering import *
from helper_functions import plot_clustering_results, plot_dominant_plane

if __name__ == "__main__":
    pc_files = [
        "pointclouds/image000.pcd",
        "pointclouds/image009.pcd"
    ]

    downsampling_configs = [
        {"method": "voxel", "label": "voxel_0.01", "voxel_size": 0.01},
        {"method": "voxel", "label": "voxel_0.02", "voxel_size": 0.02},
        {"method": "uniform", "label": "uniform_k5", "every_k": 10},
        {"method": "uniform", "label": "uniform_k10", "every_k": 15},
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
    debug_output = True # show algoriithm output

    results = []

    for pc_idx, pc_file in enumerate(pc_files):
        print(f"Processing {pc_file} ...")

        pcd = o3d.io.read_point_cloud(
            pc_file,
            remove_nan_points=True,
            remove_infinite_points=True
        )

        points_np = np.asarray(pcd.points)
        valid_idx = np.where(~np.all(points_np == 0, axis=1))[0]
        pcd = pcd.select_by_index(valid_idx)

        for ds in downsampling_configs:
            row = {
                "PointCloud": f"PC{pc_idx}",
                "Downsampling": ds["label"],
                "Method": ds["method"]
            }

            if ds["method"] == "voxel":
                pcd_sampled = pcd.voxel_down_sample(voxel_size=ds["voxel_size"])
            else:
                pcd_sampled = pcd.uniform_down_sample(every_k_points=ds["every_k"])

            plane_model, best_inliers = fit_plane(
                pcd_sampled,
                confidence=confidence,
                inlier_threshold=inlier_threshold
            )

            inlier_indices = np.nonzero(best_inliers)[0]
            scene_pcd = pcd_sampled.select_by_index(inlier_indices, invert=True)
            points = np.asarray(scene_pcd.points, dtype=np.float32)

            start = time.perf_counter()
            centers, labels = kmeans(points,
                                     n_clusters=kmeans_n_clusters,
                                     n_iterations=kmeans_iterations,
                                     max_singlerun_iterations=max_singlerun_iterations)
            row["K-Means"] = time.perf_counter() - start
            if debug_output:
                plot_clustering_results(scene_pcd, labels, f"K-Means {ds['label']}")

            start = time.perf_counter()
            centers, labels = iterative_kmeans(points,
                                               max_n_clusters=iterative_kmeans_max_clusters,
                                               n_iterations=kmeans_iterations,
                                               max_singlerun_iterations=max_singlerun_iterations)
            row["Iterative K-Means"] = time.perf_counter() - start
            if debug_output:
                plot_clustering_results(scene_pcd, labels, f"Iterative K-Means {ds['label']}")

            start = time.perf_counter()
            centers, labels = gmeans(points,
                                     tolerance=gmeans_tolerance,
                                     max_singlerun_iterations=max_singlerun_iterations)
            row["G-Means"] = time.perf_counter() - start
            if debug_output:
                plot_clustering_results(scene_pcd, labels, f"G-Means {ds['label']}")

            start = time.perf_counter()
            labels = dbscan(points,
                            eps=dbscan_eps,
                            min_samples=dbscan_min_points)
            row["DBSCAN"] = time.perf_counter() - start
            if debug_output:
                plot_clustering_results(scene_pcd, labels, f"DBSCAN {ds['label']}")

            results.append(row)

    df = pd.DataFrame(results)
    print("\n=== Runtime Results ===")
    print(df.round(4))

    algorithms = ["K-Means", "Iterative K-Means", "G-Means", "DBSCAN"]

    df_mean = (
        df
        .groupby("Downsampling")[algorithms]
        .mean()
    )

    downsampling_order = ["voxel_0.01", "voxel_0.02", "uniform_k5", "uniform_k10"]

    df_mean = df_mean.reindex(downsampling_order)

    print("\n=== Average runtimes over point clouds ===")
    print(df_mean.round(4))

    df_mean.plot(
        kind="bar",
        figsize=(10, 6)
    )

    plt.yscale('log')
    plt.ylabel("Execution Time [s]")
    plt.title("Average runtime vs downsampling level")
    plt.xticks(rotation=0)
    plt.legend(title="Algorithm")
    plt.tight_layout()
    plt.show()
