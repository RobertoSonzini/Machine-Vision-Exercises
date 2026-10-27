#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Machine Vision (376.081)
Exercise 1: Canny Edge Detector
Matthias Hirschmanner 2025
Automation & Control Institute, TU Wien

Tutors: machinevision@acin.tuwien.ac.at

sources for some plots: https://chatgpt.com/share/68f0c71b-0588-800e-aa58-3c0a931b6637
"""
from pathlib import Path

import cv2
import numpy as np

from blur_gauss import blur_gauss
from helper_functions import *
from hyst_auto import hyst_thresh_auto
from hyst_thresh import hyst_thresh
from non_max import non_max
from sobel import sobel

def add_gaussian_noise_clip_info(img: np.array, mean: float = 0.0, sigma: float = 0.1) -> np.array:
    noisy_img = img.copy()
    if noisy_img.dtype != np.float32:
        noisy_img.astype(np.float32)
        if np.max(noisy_img) > 1.0:
            noisy_img = noisy_img/255.

    noise = sigma * np.random.randn(*noisy_img.shape) + mean
    noisy_img += noise
    cv2.normalize(noisy_img, noisy_img, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

    num_clipped = np.sum(noisy_img < 0.) + np.sum(noisy_img > 1.)

    # Sometimes values are outside of limits due to rounding errors. Cut those values:
    noisy_img[noisy_img < 0.] = 0.
    noisy_img[noisy_img > 1.] = 1.
    return noisy_img, num_clipped

def get_orientations(orientation):
    orientation_copy = orientation.copy()
    # I convert the orientations radiants in degrees (module 180)
    orientation_deg = np.rad2deg(orientation_copy) % 180

    # I observe the orientations as vertical, horizontal and diagonal
    orientation_horizontal = (orientation_deg <= 22.5) | (orientation_deg > 157.5)
    orientation_diag = (orientation_deg > 22.5) & (orientation_deg <= 67.5)
    orientation_vertical = (orientation_deg > 67.5) & (orientation_deg <= 112.5)
    orientation_rev_diag = (orientation_deg > 112.5) & (orientation_deg <= 157.5)

    return orientation_vertical, orientation_horizontal, orientation_diag, orientation_rev_diag

def plot(blurs, noises, variances_list, labels=None, ncols=2, xlabel="", ylabel=""):
    n_plots = len(variances_list)
    nrows = (n_plots + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(8*ncols, 4*nrows))
    axes = np.array(axes).reshape(-1)
    
    for i, variances in enumerate(variances_list):
        label = labels[i] if labels is not None and i < len(labels) else f"Serie {i+1}"

        for j, var_sigma in enumerate(variances):
            axes[i].plot(noises, var_sigma, 'o-', linewidth=2, markersize=6, label=f"{labels[i]} (blur: {blurs[j]})")

        axes[i].set_xlabel(xlabel, fontsize=12)
        axes[i].set_ylabel(ylabel, fontsize=12)
        axes[i].grid(True, linestyle='--', alpha=0.6)
        axes[i].legend()
    
    for j in range(n_plots, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

def get_differences():
    noises = np.arange(0, 0.5, 0.01)
    sigmas = np.arange(2, 12, 2)

    difference_orientations = []

    img_gray = cv2.imread(str(current_path.joinpath("image/rubens.jpg")), cv2.IMREAD_GRAYSCALE)

    for sigma in sigmas:
        dif_sigma_or = []

        for noise in noises:
            img_noised = add_gaussian_noise(img_gray, 0, noise)

            img_blur_noised = blur_gauss(img_noised, sigma)
            img_blur = blur_gauss(img_gray, sigma)

            gradients_noised, orientations_noised = sobel(img_blur_noised)
            gradients, orientations = sobel(img_blur)

            orientation_vertical_orig, orientation_horizontal_orig, orientation_diag_orig, orientation_rev_diag_orig = get_orientations(orientations)
            orientation_vertical_noised, orientation_horizontal_noised, orientation_diag_noised, orientation_rev_diag_noised = get_orientations(orientations_noised)
            
            # Logic XOR
            dif_orientations_vert = orientation_vertical_orig ^ orientation_vertical_noised
            dif_orientations_horiz = orientation_horizontal_orig ^ orientation_horizontal_noised
            dif_orientations_diag = orientation_diag_orig ^ orientation_diag_noised
            dif_orientations_revdiag = orientation_rev_diag_orig ^ orientation_rev_diag_noised

            num_vert = np.sum(dif_orientations_vert)
            num_horiz = np.sum(dif_orientations_horiz)
            num_diag = np.sum(dif_orientations_diag)
            num_revdiag = np.sum(dif_orientations_revdiag)

            total_pixels_changed = num_vert + num_horiz + num_diag + num_revdiag

            dif_sigma_or.append(total_pixels_changed)

        difference_orientations.append(dif_sigma_or)

    plot(sigmas, noises, [difference_orientations], ["Distance orientations"], ncols=1, xlabel="Noise std", ylabel="Distance from no noised result")

def get_variances():
    noises = np.arange(0, 0.5, 0.01)
    sigmas = np.arange(1, 6, 1)

    variances_image = []
    variance_img_blur = []
    variances_gradients = []
    variances_orientations = []

    clipped = []

    for sigma in sigmas:
        var_sigma_image = []
        var_sigma_blur = []
        var_sigma_grad = []
        var_sigma_or = []

        num_clipped_per_noise = []

        for noise in noises:
            img_gray = cv2.imread(str(current_path.joinpath("image/rubens.jpg")), cv2.IMREAD_GRAYSCALE)

            img_noised, num_clipped = add_gaussian_noise_clip_info(img_gray, 0, noise)
            var_sigma_image.append(np.var(img_noised))
            num_clipped_per_noise.append(num_clipped)

            img_blur = blur_gauss(img_noised, sigma)
            var_sigma_blur.append(np.var(img_blur))

            gradients, orientations = sobel(img_blur)
            var_sigma_grad.append(np.var(gradients))
            var_sigma_or.append(np.var(orientations))

        variances_image.append(var_sigma_image)
        variance_img_blur.append(var_sigma_blur)
        variances_gradients.append(var_sigma_grad)
        variances_orientations.append(var_sigma_or)

        clipped.append(num_clipped_per_noise)

    clipped_mean = np.mean(np.array(clipped), axis=0)

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(noises, clipped_mean, 'o-', linewidth=2)
    ax.set_xlabel("Noise Standard Deviation")
    ax.set_ylabel("Mean Clipped Pixels")
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    plot(sigmas, noises, [variances_image, variance_img_blur, variances_gradients, variances_orientations], ["Img_variance", "Blur_Img_variance", "gradients_variance", "orintations_variance"], xlabel="Noise std", ylabel="Variance")

if __name__ == '__main__':

    # Define behavior of the show_image function. You can change these variables if necessary
    save_image = False
    matplotlib_plotting = False

    # Read image. You can change the image you want to process here.
    current_path = Path(__file__).parent

    get_variances()
    get_differences()

    # Destroy all OpenCV windows in case we have any open
    cv2.destroyAllWindows()
