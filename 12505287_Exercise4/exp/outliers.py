import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from clustering import kmeans, iterative_kmeans, gmeans, dbscan

np.random.seed(42)

X_base, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=42)
outliers1 = np.array([[6, 6], [-5, -4], [7, -3]])
X_case1 = np.vstack([X_base, outliers1])

cluster1 = np.random.randn(150, 2) * 0.2 + np.array([0, 0])
cluster2 = np.random.randn(150, 2) * 0.2 + np.array([2, 0]) 

eps_value_fake = 0.3

distance_between_clusters = np.linalg.norm([2-0, 0-0])
n_outliers = int(np.ceil(distance_between_clusters / eps_value_fake)) - 1

outliers = np.zeros((n_outliers, 2))
for i in range(n_outliers):
    t = (i + 1) / (n_outliers + 1)
    outliers[i] = np.array([0, 0]) * (1-t) + np.array([3, 0]) * t

X_case2 = np.vstack([cluster1, cluster2, outliers])

cases = [("Few distant outliers", X_case1), ("Two close clusters + outliers", X_case2)]

algorithms = ["k-means", "iterative k-means", "DBSCAN"]

fig, axs = plt.subplots(len(cases), len(algorithms), figsize=(16, 8))

for row, (case_name, X) in enumerate(cases):

    centers, labels = kmeans(X, n_clusters=3, n_iterations=5, max_singlerun_iterations=50)
    axs[row, 0].scatter(X[:, 0], X[:, 1], c=labels, s=10)
    axs[row, 0].set_title("k-means")

    centers, labels = iterative_kmeans(X, max_n_clusters=6, n_iterations=5, max_singlerun_iterations=50)
    axs[row, 1].scatter(X[:, 0], X[:, 1], c=labels, s=10)
    axs[row, 1].set_title("Iter. k-means")

    eps_value = 0.7
    labels = dbscan(X, eps=eps_value, min_samples=5)
    axs[row, 2].scatter(X[:, 0], X[:, 1], c=labels, s=10, cmap="tab10")
    axs[row, 2].set_title(f"DBSCAN (eps={eps_value})")

    for col in range(len(algorithms)):
        axs[row, col].axis("equal")
        axs[row, col].set_xticks([])
        axs[row, col].set_yticks([])

    axs[row, 0].set_ylabel(case_name, fontsize=12)

plt.suptitle("Clustering behavior with outliers (2D)", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()
