#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Roberto Sonzini Gobbi
MatrNr: 12505287
"""
from pathlib import Path

import cv2
import numpy as np

from blur_gauss import blur_gauss
from math import ceil
from helper_functions import *
from hyst_auto import hyst_thresh_auto
from hyst_thresh import hyst_thresh
from non_max import non_max
from sobel import sobel

def kernel_width(sigma):
    return 2 * ceil(3 * sigma) + 1

def plot_width_function():
    sigma_values = np.linspace(0.1, 10, 100)

    width_values = [kernel_width(s) for s in sigma_values]

    plt.plot(sigma_values, width_values, marker='o')
    plt.xlabel('Sigma')
    plt.ylabel('Kernel Width')
    plt.title('Kernel Width')
    plt.grid(True)
    plt.show()
    return 

def plot_full_image_intensities(img: np.array, skip: int = 1, title: str = "Intensities") -> None:
    plt.figure(figsize=(8,6))
    
    for row in range(0, img.shape[0], skip):
        plt.plot(np.arange(img.shape[1]), img[row, :], alpha=0.5)
    
    plt.xlabel("Column")
    plt.ylabel("Intensity")
    plt.title(title)
    plt.grid(True)
    plt.show()

if __name__ == '__main__':

    # Define behavior of the show_image function. You can change these variables if necessary
    save_image = True
    matplotlib_plotting = False

    # Read image. You can change the image you want to process here.
    current_path = Path(__file__).parent
    img_gray = cv2.imread(str(current_path.joinpath("image/rubens.jpg")), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError("Couldn't load image in " + str(current_path))

    # Before we start working with the image, we convert it from uint8 with range [0,255] to float32 with range [0,1]
    img_gray = img_gray.astype(np.float32) / 255.
    show_image(img_gray, "Original Image", save_image=save_image, use_matplotlib=matplotlib_plotting)

    # show the kernel_width function
    plot_width_function()

    # 1. Blur Image
    sigmas = [1, 3, 4, 5, 10]
    for sigma in sigmas:
        img_blur = blur_gauss(img_gray, sigma)

        show_image(img_blur, "Blurred Image", save_image=save_image, use_matplotlib=matplotlib_plotting)

    sigmas = [1, 5, 10]
    for sigma in sigmas:
        img_blur = blur_gauss(img_gray, sigma)
        plot_full_image_intensities(img_blur, 50)

    # Destroy all OpenCV windows in case we have any open
    cv2.destroyAllWindows()