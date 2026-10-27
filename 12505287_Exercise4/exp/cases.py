import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons

from clustering import kmeans, iterative_kmeans, gmeans, dbscan

np.random.seed(42)

datasets = {}

X1, _ = make_blobs(n_samples=400, centers=3, cluster_std=0.4)
noise = np.random.uniform(low=-6, high=6, size=(50, 2))
datasets["Outliers"] = np.vstack([X1, noise])

datasets["K dependence"] = make_blobs(n_samples=400, centers=3, cluster_std=0.5)[0]

centers = [[0, 0], [0.8, 0]]
datasets["Close objects"] = make_blobs(
    n_samples=400, centers=centers, cluster_std=0.4
)[0]

X4a, _ = make_blobs(n_samples=200, centers=[[0, 0]], cluster_std=0.3)
X4b, _ = make_blobs(n_samples=200, centers=[[0, 1.2]], cluster_std=0.3)
datasets["Stacked objects"] = np.vstack([X4a, X4b])

algorithms = ["k-means", "iterative k-means", "g-means", "DBSCAN"]
fig, axs = plt.subplots(len(datasets), len(algorithms), figsize=(16, 12))

for row, (name, X) in enumerate(datasets.items()):

    centers, labels = kmeans(
        X, n_clusters=4, n_iterations=5, max_singlerun_iterations=50
    )
    axs[row, 0].scatter(X[:, 0], X[:, 1], c=labels, s=10)
    axs[row, 0].set_title("k-means (K=4)")

    centers, labels = iterative_kmeans(
        X, max_n_clusters=6, n_iterations=3, max_singlerun_iterations=50
    )
    axs[row, 1].scatter(X[:, 0], X[:, 1], c=labels, s=10)
    axs[row, 1].set_title("Iterative k-means")

    try:
        centers, labels = gmeans(X, tolerance=1.0, max_singlerun_iterations=50)
        axs[row, 2].scatter(X[:, 0], X[:, 1], c=labels, s=10)
        axs[row, 2].set_title("G-means")
    except Exception:
        axs[row, 2].text(0.5, 0.5, "Failed", ha="center", va="center")
        axs[row, 2].set_title("G-means")

    labels = dbscan(X, eps=0.5, min_samples=10)
    axs[row, 3].scatter(
        X[:, 0], X[:, 1],
        c=labels, s=10, cmap="tab10"
    )
    axs[row, 3].set_title("DBSCAN")

    for col in range(len(algorithms)):
        axs[row, col].axis("equal")
        axs[row, col].set_xticks([])
        axs[row, col].set_yticks([])

    axs[row, 0].set_ylabel(name, fontsize=12)

plt.suptitle("2D Clustering Failure Cases and Algorithm Behavior", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()
