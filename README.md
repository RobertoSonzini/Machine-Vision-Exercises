# Machine Vision Exercises

Solutions to the exercises of **Machine Vision (376.081)** at TU Wien (Automation & Control Institute).

| # | Topic | Main entry point |
|---|-------|------------------|
| 1 | Canny edge detector (Gaussian blur, Sobel, non-max suppression, hysteresis) | `12505287_Exercise1/mv_exercise1/main.py` |
| 2 | Interest points & descriptors: Harris, SIFT, RANSAC homography, image stitching | `12505287_Exercise2/mv_exercise2/main_*.py` |
| 3 | Image classification on CIFAR-10 with PyTorch (linear, MLP, CNN) | `12505287_Exercise3/12505287_mv_ex3_2025.ipynb` |
| 4 | Point cloud processing: plane fitting and clustering (k-means, DBSCAN) | `12505287_Exercise4/mv_exercise4/main.py` |
| 5 | Object recognition on RGB-D data (point cloud + SIFT + neural network) | `12505287_Exercise5/mv_exercise5/main.py` |

Each folder contains the submitted code (`mv_exerciseX/`), extra experiments (`exp/`, `mv_exp/`, `experiments/`) and a PDF report.

## Setup

```bash
conda env create -f 12505287_Exercise5/mv_exercise5/new_libs.yml
conda activate machinevision
```

Python 3.12 · NumPy · OpenCV · Matplotlib · scikit-learn · Open3D · PyTorch.
Exercise 3 is meant to run on Google Colab (GPU).

## Run

```bash
cd 12505287_Exercise1/mv_exercise1
python main.py
```

Scripts load data via relative paths, so run them from inside their folder.
