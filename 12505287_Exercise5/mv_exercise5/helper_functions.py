import numpy as np
import open3d as o3d       
import numpy as np         
import cv2                 
import os 
from typing import Tuple
import copy
import matplotlib.pyplot as plt

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

def plot_dominant_plane(pcd: o3d.geometry.PointCloud,
                        inliers: np.ndarray,
                        plane_eq: np.ndarray) -> None:
    """ Plot the inlier points in red and the rest of the pointcloud as is. A coordinate frame is drawn on the plane

    :param pcd: The (down-sampled) pointcloud in which to detect the dominant plane
    :type pcd: o3d.geometry.PointCloud

    :param inliers: Boolean array with the same size as pcd.points. Is True if the point at the index is an inlier
    :type inliers: np.array

    :param plane_eq: An array with the coefficients of the plane equation ax+by+cz+d=0
    :type plane_eq: np.array [a,b,c,d]

    :return: None
    """

    # Filter the inlier points and color them red
    inlier_indices = np.nonzero(inliers)[0]
    inlier_cloud = pcd.select_by_index(inlier_indices)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcd.select_by_index(inlier_indices, invert=True)

    # Create a rotation matrix according to the plane equation.
    # Detailed explanation of the approach can be found here: https://math.stackexchange.com/q/1957132
    normal_vector = -plane_eq[0:3] / np.linalg.norm(plane_eq[0:3])
    u1 = np.cross(normal_vector, [0, 0, 1])
    u2 = np.cross(normal_vector, u1)
    rot_mat = np.c_[u1, u2, normal_vector]

    # Create a coordinate frame and transform it to a point on the plane and with its z-axis in the same direction as
    # the normal vector of the plane
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    coordinate_frame.rotate(rot_mat, center=(0, 0, 0))
    if any(inlier_indices):
        coordinate_frame.translate(np.asarray(inlier_cloud.points)[-1])
        coordinate_frame.scale(0.3, np.asarray(inlier_cloud.points)[-1])

    geometries = [inlier_cloud, outlier_cloud, coordinate_frame]

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for p in geometries:
        vis.add_geometry(p)
    vc = vis.get_view_control()
    vc.set_front([-0.3, 0.32, -0.9])
    vc.set_lookat([-0.13, -0.15, 0.92])
    vc.set_up([0.22, -0.89, -0.39])
    vc.set_zoom(0.5)
    vis.run()
    vis.destroy_window()

def remove_ground_plane(pcd, confidence=0.95, inlier_threshold=0.02, plots = False):
    plane_model, best_inliers = fit_plane(pcd=pcd,
                                      confidence=confidence,
                                      inlier_threshold=inlier_threshold)
    
    inlier_indices = np.nonzero(best_inliers)[0]
    best_inliers_bool = np.full(len(pcd.points), False)
    best_inliers_bool[inlier_indices] = True
    best_inliers = best_inliers_bool

    if plots:
        plot_dominant_plane(pcd, best_inliers, plane_model)

    scene_pcd = pcd.select_by_index(np.where(best_inliers==False)[0])

    if plots:
        o3d.visualization.draw_geometries([scene_pcd])

    return scene_pcd

def plot_clustering_results(pcd: o3d.geometry.PointCloud,
                            labels: np.ndarray,
                            method_name: str,
                            cmap: str = "tab20"):
    """
    Plot the results of the clustering analysis. If a label is negative, it is considered noise.

    :param pcd: The pointcloud to be plotted.
    :type pcd: o3d.geometry.PointCloud

    :param labels: Labels array with a different label for each cluster for each point in the pointcloud
    :type labels: np.ndarray with shape = (n_points,) and dtype=int

    :param method_name: The name of the message. Will only be used in the print message.
    :type method_name: str

    :param cmap: The color map to be used. Check https://matplotlib.org/stable/users/explain/colors/colormaps.html
    :type cmap: str

    :return: None
    """

    unique_labels = np.unique(labels)
    if method_name is not None:
        if np.any(unique_labels < 0):
            print(f"{method_name}: Plotting {len(unique_labels)-1} clusters. Noise is shown in black.")
        else:
            print(f"{method_name}: Plotting {len(unique_labels)} clusters.")

    # Get different colors for each label. We only consider labels >= 0 since all negative labels will be shown as noise
    colors = plt.get_cmap(cmap)(labels / (sum(unique_labels>=0) if sum(unique_labels>=0) > 0 else 1))

    # Set the color of the noise points to black.
    colors[labels < 0] = 0

    # Assign colors to pointcloud and plot it
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd])

def clusters_to_2d_mask(pcd, labels,
                        fx, fy, cx, cy,
                        img_width=640, img_height=480,
                        radius=6,
                        closing_kernel=40,
                        plots=True):
    points = np.asarray(pcd.points)
    if len(points) != len(labels):
        raise ValueError(f"Mismatch tra points ({len(points)}) e labels ({len(labels)})")

    mask_2d = np.zeros((img_height, img_width), dtype=np.int32)
    z_buffer = np.full((img_height, img_width), np.inf)

    valid = (labels >= 0) & (points[:, 2] > 0)
    points = points[valid]
    labels = labels[valid]

    if len(points) == 0:
        return mask_2d

    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
    u = (fx * X / Z + cx).astype(np.int32)
    v = (fy * Y / Z + cy).astype(np.int32)

    for ui, vi, zi, lbl in zip(u, v, Z, labels):
        for du in range(-radius, radius + 1):
            for dv in range(-radius, radius + 1):
                uu, vv = ui + du, vi + dv
                if 0 <= uu < img_width and 0 <= vv < img_height:
                    if zi < z_buffer[vv, uu]:
                        z_buffer[vv, uu] = zi
                        mask_2d[vv, uu] = lbl + 1

    if closing_kernel > 1:
        fg_mask = (mask_2d > 0).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_kernel, closing_kernel))
        fg_mask_closed = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        mask_2d[fg_mask_closed == 0] = 0

    if plots:
        plt.figure(figsize=(6, 4))
        plt.imshow(mask_2d, cmap='jet')
        plt.colorbar(label='Cluster ID')
        plt.title("Cluster Mask 2D Dense Projection")
        plt.axis('off')
        plt.show()

    return mask_2d

def apply_mask(image_2d, mask_2d, cluster=-1, plots=True):
    if cluster == -1:
        mask_bool = mask_2d > 0
    else:
        mask_bool = mask_2d == cluster
        
    image_masked = np.zeros_like(image_2d, dtype=np.uint8)

    image_masked[mask_bool] = image_2d[mask_bool]

    if plots:
        plt.figure(figsize=(6, 4))
        plt.imshow(image_masked)
        title = f"Cluster {cluster}" if cluster >= 0 else "All clusters"
        plt.axis('off')
        plt.show()

    return image_masked

def predict_class_per_cluster(cluster_classes, threshold=None, plots=True):
    predicted_classes = {}

    for cluster_id, class_scores in cluster_classes.items():
        best_class, best_score = max(class_scores.items(), key=lambda x: x[1])

        if threshold is not None and best_score < threshold:
            predicted_classes[cluster_id] = "unknown"
        else:
            predicted_classes[cluster_id] = best_class

    if plots:
        print("\n=== Predicted Class per Cluster ===")
        for cluster_id, pred_class in predicted_classes.items():
            print(f"Cluster {cluster_id}: {pred_class}")

    return predicted_classes

def label_clusters_with_text(image_2D, mask_2d, predicted_classes, plots = True):
    labeled_image = image_2D.copy()

    for cluster_id, cls in predicted_classes.items():
        ys, xs = np.where(mask_2d == cluster_id)
        if len(xs) == 0 or len(ys) == 0:
            continue

        cx, cy = int(xs.mean()), int(ys.mean())

        cv2.putText(labeled_image,
                    cls,
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0,0,255),
                    thickness=1,
                    lineType=cv2.LINE_AA)
        
    if plots:
        cv2.imshow("Clusters labeled", labeled_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()