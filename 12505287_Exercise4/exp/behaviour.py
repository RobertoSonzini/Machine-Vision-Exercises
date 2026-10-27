import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons

# resources: https://chatgpt.com/share/693d49a5-364c-800e-8204-5d020b8dda57 (for cluster generation)

np.random.seed(42)

fig, axs = plt.subplots(2, 3, figsize=(12, 8))
axs = axs.flatten()

X1, _ = make_blobs(n_samples=500, centers=3, cluster_std=0.5)
axs[0].scatter(X1[:, 0], X1[:, 1], s=10)
axs[0].set_title("Spherical Clusters")
axs[0].axis("equal")

X2, _ = make_blobs(n_samples=500, centers=2, cluster_std=[0.2, 1.5])
X2[:, 0] *= 3
axs[1].scatter(X2[:, 0], X2[:, 1], s=10)
axs[1].set_title("Elongated Clusters")
axs[1].axis("equal")

X3, _ = make_moons(n_samples=500, noise=0.05)
axs[2].scatter(X3[:, 0], X3[:, 1], s=10)
axs[2].set_title("Non-convex Clusters (Moons)")
axs[2].axis("equal")

X4, _ = make_blobs(n_samples=450, centers=3, cluster_std=0.4)
noise = np.random.uniform(low=-6, high=6, size=(50, 2))
X4 = np.vstack([X4, noise])
axs[3].scatter(X4[:, 0], X4[:, 1], s=10)
axs[3].set_title("Clusters with Noise / Outliers")
axs[3].axis("equal")

X5a, _ = make_blobs(n_samples=300, centers=[[0, 0]], cluster_std=0.2)
X5b, _ = make_blobs(n_samples=100, centers=[[3, 3]], cluster_std=0.8)
X5 = np.vstack([X5a, X5b])
axs[4].scatter(X5[:, 0], X5[:, 1], s=10)
axs[4].set_title("Clusters with Different Densities")
axs[4].axis("equal")

axs[5].axis("off")

plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.show()
