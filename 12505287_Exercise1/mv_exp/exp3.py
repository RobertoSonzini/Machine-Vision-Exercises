#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Machine Vision (376.081)
Exercise 1: Canny Edge Detector
Matthias Hirschmanner 2025
Automation & Control Institute, TU Wien

Tutors: machinevision@acin.tuwien.ac.at
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

def plot_variances(sigmas, variances_list, labels=None, ncols=2):
    n_plots = len(variances_list)
    nrows = (n_plots + ncols - 1) // ncols  # numero di righe necessario
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = np.array(axes).reshape(-1)  # appiattisce la griglia in array 1D
    
    for i, variances in enumerate(variances_list):
        label = labels[i] if labels is not None and i < len(labels) else f"Serie {i+1}"
        axes[i].plot(sigmas, variances, 'o-', linewidth=2, markersize=6, label=label)
        axes[i].set_xlabel("Noise Standard Deviation", fontsize=12)
        axes[i].set_ylabel("Variance", fontsize=12)
        axes[i].grid(True, linestyle='--', alpha=0.6)
        axes[i].legend()
    
    for j in range(n_plots, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

def get_variances():
    noises = np.arange(0.01, 0.5, 0.1)

    sigma = 5

    variances_image = []
    variance_img_blur = []
    variances_gradients = []
    variances_orientations = []

    for noise in noises:
        img_gray = cv2.imread(str(current_path.joinpath("image/rubens.jpg")), cv2.IMREAD_GRAYSCALE)

        img_noised = add_gaussian_noise(img_gray, 0, noise)
        variances_image.append(np.var(img_noised))

        img_blur = blur_gauss(img_noised, sigma)
        variance_img_blur.append(np.var(img_blur))

        gradients, orientations = sobel(img_blur)
        variances_gradients.append(np.var(gradients))
        variances_orientations.append(np.var(orientations))

    plot_variances(noises, [variances_image, variance_img_blur, variances_gradients, variances_orientations], ["Img_variance", "Blur_Img_variance", "gradients_variance", "orintations_variance"])

if __name__ == '__main__':
    # Define behavior of the show_image function. You can change these variables if necessary
    save_image = False
    matplotlib_plotting = False

    # Read image. You can change the image you want to process here.
    current_path = Path(__file__).parent
    img_gray = cv2.imread(str(current_path.joinpath("image/rubens.jpg")), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError("Couldn't load image in " + str(current_path))

    # Before we start working with the image, we convert it from uint8 with range [0,255] to float32 with range [0,1]
    img_gray = img_gray.astype(np.float32) / 255.
    show_image(img_gray, "Original Image", save_image=save_image, use_matplotlib=matplotlib_plotting)

    noises = [0.01, 0.1, 0.3, 0.5]
    for noise_sd in noises:
        # Adding gaussian noise
        img_noised = add_gaussian_noise(img_gray, 0, noise_sd)

        show_image(img_noised, "Noisy Image", save_image=save_image, use_matplotlib=matplotlib_plotting)

        # 1. Blur Image
        sigma = 5  # Change this value
        img_blur = blur_gauss(img_noised, sigma)
        show_image(img_blur, "Blurred Image", save_image=save_image, use_matplotlib=matplotlib_plotting)

        # 2. Edge Detection
        gradients, orientations = sobel(img_blur)

        orientations_color = cv2.applyColorMap(np.uint8((orientations.copy() + np.pi) / (2 * np.pi) * 255),
                                           cv2.COLORMAP_RAINBOW)
        orientations_color = orientations_color.astype(np.float32) / 255.
        show_image(orientations_color, "Orientation color map", save_image=save_image, use_matplotlib=matplotlib_plotting)
        gradient_img = np.append(cv2.cvtColor(gradients, cv2.COLOR_GRAY2BGR), orientations_color, axis=1)
        show_image(gradient_img, "Gradients", save_image=save_image, use_matplotlib=matplotlib_plotting)

        # 3. Non-Maxima Suppression
        edges = non_max(gradients, orientations)

        # 4. Hysteresis Thresholding
        hyst_method_auto = True

        if hyst_method_auto:
            canny_edges = hyst_thresh_auto(edges, 0.9, 0.035)
        else:
            canny_edges = hyst_thresh(edges, 0.04, 0.1)
        show_image(canny_edges, "Canny Edges", save_image=save_image, use_matplotlib=matplotlib_plotting)

        # Overlay the found edges in red over the original image
        img_gray_overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        img_gray_overlay[canny_edges == 1.0] = (0., 0., 1.)
        show_image(img_gray_overlay, "Overlay", save_image=save_image, use_matplotlib=matplotlib_plotting)

    get_variances()

    # Destroy all OpenCV windows in case we have any open
    cv2.destroyAllWindows()
