import time
import numpy as np
import open3d as o3d
import pandas as pd

from fit_plane import fit_plane
from clustering import *

if __name__ == "__main__":
    pc_files = [
        "pointclouds/image000.pcd",
        "pointclouds/image009.pcd"
    ]

    confidence = 0.90
    inlier_threshold = 0.02
    use_voxel_downsampling = True
    voxel_size = 0.01
    uniform_every_k_points = 10

    kmeans_n_clusters = 6
    kmeans_iterations = 25
    max_singlerun_iterations = 100
    iterative_kmeans_max_clusters = 10
    gmeans_tolerance = 10
    dbscan_eps = 0.05
    dbscan_min_points = 15

    use_kmeans = True
    use_iterative_kmeans = True
    use_gmeans = True
    use_dbscan = True

    results = []
    for idx, pc_file in enumerate(pc_files):
        print(f"Processing {pc_file} ...")
        row = {"PointCloud": idx}

        pcd = o3d.io.read_point_cloud(pc_file, remove_nan_points=True, remove_infinite_points=True)
        if not pcd.has_points():
            print(f"WARNING: {pc_file} is empty, skipping")
            continue

        points_np = np.asarray(pcd.points)
        valid_idx = np.where(~np.all(points_np == 0, axis=1))[0]
        pcd = pcd.select_by_index(valid_idx)

        if use_voxel_downsampling:
            pcd_sampled = pcd.voxel_down_sample(voxel_size=voxel_size)
        else:
            pcd_sampled = pcd.uniform_down_sample(uniform_every_k_points)

        plane_model, best_inliers = fit_plane(pcd_sampled, confidence=confidence, inlier_threshold=inlier_threshold)
        inlier_indices = np.nonzero(best_inliers)[0]
        scene_pcd = pcd_sampled.select_by_index(inlier_indices, invert=True)
        points = np.asarray(scene_pcd.points, dtype=np.float32)

        if use_kmeans:
            start = time.perf_counter()
            centers, labels = kmeans(points, n_clusters=kmeans_n_clusters,
                                     n_iterations=kmeans_iterations,
                                     max_singlerun_iterations=max_singlerun_iterations)
            row["K-Means"] = time.perf_counter() - start

        if use_iterative_kmeans:
            start = time.perf_counter()
            centers, labels = iterative_kmeans(points,
                                               max_n_clusters=iterative_kmeans_max_clusters,
                                               n_iterations=kmeans_iterations,
                                               max_singlerun_iterations=max_singlerun_iterations)
            row["Iterative K-Means"] = time.perf_counter() - start

        if use_gmeans:
            start = time.perf_counter()
            centers, labels = gmeans(points,
                                     tolerance=gmeans_tolerance,
                                     max_singlerun_iterations=max_singlerun_iterations)
            row["G-Means"] = time.perf_counter() - start

        if use_dbscan:
            start = time.perf_counter()
            labels = dbscan(points,
                            eps=dbscan_eps,
                            min_samples=dbscan_min_points)
            row["DBSCAN"] = time.perf_counter() - start

        results.append(row)

    df = pd.DataFrame(results)

    mean_row = pd.DataFrame([{"PointCloud": "Average",
                              "K-Means": df["K-Means"].mean(),
                              "Iterative K-Means": df["Iterative K-Means"].mean(),
                              "G-Means": df["G-Means"].mean(),
                              "DBSCAN": df["DBSCAN"].mean()}])
    df = pd.concat([df, mean_row], ignore_index=True)

    pd.set_option("display.precision", 4)
    print("\n=== Clustering Execution Times (seconds) ===")
    print(df)

    df_plot = df[df["PointCloud"] != "Average"]

    algorithms = ["K-Means", "Iterative K-Means", "G-Means", "DBSCAN"]

    df_plot.set_index("PointCloud")[algorithms].plot(kind="bar", figsize=(10,6))
    plt.ylabel("Execution Time [s]")
    plt.title("Clustering Execution Times per PointCloud")
    plt.xticks(rotation=0)
    plt.legend(title="Algorithm")
    plt.tight_layout()
    plt.show()

