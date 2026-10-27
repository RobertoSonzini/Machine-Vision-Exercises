import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from clustering import kmeans, iterative_kmeans

np.random.seed(42)

X, _ = make_blobs(
    n_samples=500,
    centers=4,
    cluster_std=0.5,
    random_state=42
)

X2, _ = make_blobs(
    n_samples=500,
    centers=10,
    cluster_std=0.5,
    random_state=42
)

fig, axs = plt.subplots(2, 3, figsize=(15, 8))

centers, labels = kmeans(X, n_clusters=4, n_iterations=5, max_singlerun_iterations=50)
axs[0, 0].scatter(X[:, 0], X[:, 1], c=labels, s=10)
axs[0, 0].set_title("K-means: K=4 (correct)")

centers, labels = kmeans(X, n_clusters=5, n_iterations=5, max_singlerun_iterations=50)
axs[0, 1].scatter(X[:, 0], X[:, 1], c=labels, s=10)
axs[0, 1].set_title("K-means: K=5 (wrong)")

centers, labels = kmeans(X2, n_clusters=5, n_iterations=5, max_singlerun_iterations=50)
axs[0, 2].scatter(X2[:, 0], X2[:, 1], c=labels, s=10)
axs[0, 2].set_title("K-means: K=5 (wrong)")

centers, labels = iterative_kmeans(X, max_n_clusters=5, n_iterations=5, max_singlerun_iterations=50)
axs[1, 0].scatter(X[:, 0], X[:, 1], c=labels, s=10)
axs[1, 0].set_title("Iter. k-means: max K=5")

centers, labels = iterative_kmeans(X, max_n_clusters=5, n_iterations=5, max_singlerun_iterations=50)
axs[1, 1].scatter(X[:, 0], X[:, 1], c=labels, s=10)
axs[1, 1].set_title("Iter. k-means: max K=5")

centers, labels = iterative_kmeans(X2, max_n_clusters=5, n_iterations=5, max_singlerun_iterations=50)
axs[1, 2].scatter(X2[:, 0], X2[:, 1], c=labels, s=10)
axs[1, 2].set_title("Iter. k-means: max K=5")

for i in range(2):
    for j in range(3):
        axs[i, j].axis("equal")
        axs[i, j].set_xticks([])
        axs[i, j].set_yticks([])

axs[0, 0].set_ylabel("k-means", fontsize=12)
axs[1, 0].set_ylabel("Iterative k-means", fontsize=12)

plt.suptitle("Dependence on K: k-means vs Iterative k-means", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()
