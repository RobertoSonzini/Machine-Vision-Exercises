import numpy as np
import matplotlib.pyplot as plt
from clustering import kmeans, iterative_kmeans, gmeans, dbscan


np.random.seed(0)

c1 = np.random.randn(200, 2) @ np.array([[2, 0], [0, 0.3]]) + np.array([0, 0])

c2 = np.random.randn(200, 2) @ np.array([[2, 0], [0, 0.3]]) + np.array([5, 1])

theta = np.random.rand(200) * 2 * np.pi
r = 0.8 + 0.1 * np.random.randn(200)
c3 = np.column_stack([r * np.cos(theta), r * np.sin(theta)]) + np.array([2.5, 4])

noise = np.random.uniform(low=-3, high=8, size=(80, 2))

points_2d = np.vstack([c1, c2, c3, noise]).astype(np.float32)

# K-Means
centers_km, labels_km = kmeans(
    points_2d,
    n_clusters=3,
    n_iterations=10,
    max_singlerun_iterations=100
)

# Iterative K-Means
centers_it, labels_it = iterative_kmeans(
    points_2d,
    max_n_clusters=6,
    n_iterations=10,
    max_singlerun_iterations=100
)

# G-Means
centers_gm, labels_gm = gmeans(
    points_2d,
    tolerance=10,
    max_singlerun_iterations=100
)

# DBSCAN
labels_db = dbscan(
    points_2d,
    eps=0.5,
    min_samples=10
)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
titles = ["K-Means", "Iterative K-Means", "G-Means", "DBSCAN"]
label_sets = [labels_km, labels_it, labels_gm, labels_db]

for ax, labels, title in zip(axes.flat, label_sets, titles):
    unique_labels = np.unique(labels)
    for lab in unique_labels:
        if lab == -1:
            ax.scatter(
                points_2d[labels == lab, 0],
                points_2d[labels == lab, 1],
                c="black",
                s=10,
                label="Noise"
            )
        else:
            ax.scatter(
                points_2d[labels == lab, 0],
                points_2d[labels == lab, 1],
                s=10
            )

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
