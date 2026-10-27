from pathlib import Path
import open3d as o3d       
import numpy as np         
import cv2                 
import os 
import matplotlib.pyplot as plt

from training import *

def compute_sift_keypoints_descriptors(image, mask=None):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, mask)

    return keypoints, descriptors

def match_sift_descriptors(desc_obj, desc_train, kp_obj, kp_train,
                           obj_img, train_img, visualize=False,
                           ratio_test=0.75):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(desc_obj, desc_train, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < ratio_test * n.distance:
            good_matches.append(m)

    if visualize:
        match_img = cv2.drawMatches(obj_img, kp_obj,
                                    train_img, kp_train,
                                    good_matches, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.figure(figsize=(12, 6))
        plt.imshow(match_img)
        plt.axis('off')
        plt.title(f"SIFT Matches ({len(good_matches)} good matches)")
        plt.show()

    return good_matches

def count_matches_in_training_per_cluster(training_images, training_labels, cluster_images, plots=True, visualize_matches=False):
    dict_object = {}

    for id, object_image in cluster_images.items():
        keypoints_test, descriptors_test = compute_sift_keypoints_descriptors(object_image)

        matches = {}

        for idx in range(len(training_images)):

            training_image = training_images[idx]
            training_class = training_labels[idx]

            keypoints_train, descriptors_train = compute_sift_keypoints_descriptors(training_image)

            good_matches = match_sift_descriptors(
                descriptors_test,
                descriptors_train,
                keypoints_test,
                keypoints_train,
                object_image,
                training_image,
                visualize_matches
            )

            matches_count_normalized = len(good_matches) / max(len(keypoints_test), 1)

            if training_class not in matches:
                matches[training_class] = []

            matches[training_class].append(matches_count_normalized)

        for training_class in matches:
            matches[training_class] = sum(matches[training_class]) / len(matches[training_class])

        dict_object[id] = matches

    if plots: 
        clusters = list(dict_object.keys())
        classes = sorted({cls for scores in dict_object.values() for cls in scores})
        data = np.zeros((len(clusters), len(classes)))

        for i, cluster_id in enumerate(clusters):
            for j, cls in enumerate(classes):
                data[i, j] = dict_object[cluster_id].get(cls, 0)

            
        plt.figure(figsize=(10, len(clusters)*0.5 + 2))
        plt.imshow(data, cmap="viridis", aspect="auto")
        plt.colorbar(label="Normalized SIFT match score")
        plt.xticks(ticks=np.arange(len(classes)), labels=classes, rotation=45, ha="right")
        plt.yticks(ticks=np.arange(len(clusters)), labels=clusters)
        plt.title("Heatmap of normalized SIFT matches per cluster")
        plt.xlabel("Training classes")
        plt.ylabel("Clusters")
        plt.tight_layout()
        plt.show()

    return dict_object

def sift_whole_image(training_images, training_labels, image, mask_2d, plots=True, visualize_matches=False):
    keypoints_scene, descriptors_scene = compute_sift_keypoints_descriptors(image)

    cluster_ids = np.unique(mask_2d)
    cluster_ids = cluster_ids[cluster_ids != 0]

    votes = {id: {} for id in cluster_ids}

    for img_train, label_train in zip(training_images, training_labels):
        kp_train, des_train = compute_sift_keypoints_descriptors(img_train)

        good_matches = match_sift_descriptors(
            descriptors_scene,
            des_train,
            keypoints_scene,
            kp_train,
            image,
            img_train,
            visualize=visualize_matches
        )

        match_per_cluster = {cid: 0 for cid in cluster_ids}
        for m in good_matches:
            x_scene, y_scene = keypoints_scene[m.queryIdx].pt
            cid = mask_2d[int(round(y_scene)), int(round(x_scene))]
            if cid in cluster_ids:
                match_per_cluster[cid] += 1

        best_cluster = max(match_per_cluster, key=match_per_cluster.get)
        if match_per_cluster[best_cluster] == 0:
            continue

        vote_value = match_per_cluster[best_cluster] / max(len(kp_train), 1)
        if label_train not in votes[best_cluster]:
            votes[best_cluster][label_train] = 0
        votes[best_cluster][label_train] += vote_value

    predicted_classes = {}
    for cid, vote_dict in votes.items():
        if vote_dict:
            predicted_classes[cid] = max(vote_dict, key=vote_dict.get)
        else:
            predicted_classes[cid] = "unknown"

    if plots:
        print("\n=== Predicted class per cluster ===")
        for cid, cls in predicted_classes.items():
            print(f"Cluster {cid}: {cls}")

    return predicted_classes