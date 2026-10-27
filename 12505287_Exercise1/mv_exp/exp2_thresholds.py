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

def hyst_thresh_studio(edges_in: np.array, low: float, high: float) -> np.array:
    """ Apply hysteresis thresholding.

    Apply hysteresis thresholding to return the edges as a binary image. All connected pixels with value > low are
    considered a valid edge if at least one pixel has a value > high.

    :param edges_in: Edge strength of the image in range [0.,1.]
    :type edges_in: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param low: Value below all edges are filtered out
    :type low: float in range [0., 1.]

    :param high: Value which a connected element has to contain to not be filtered out
    :type high: float in range [0., 1.]

    :return: Binary edge image
    :rtype: np.array with shape (height, width) with dtype = np.float32 and values either 0 or 1
    """
    ######################################################
    num_edges_low = 0
    num_edges_high = 0

    # I create the mask to have the edges where the value > low
    low_mask = np.where(edges_in > low, 1, 0).astype(np.uint8)

    # To not do many conversions of the matrix edges_in, i can operate direcly on the mask.
    num_edges_low, labels = cv2.connectedComponents(low_mask)

    # for each object we see if the condition is satisfied and if the can mantain the border.
    mask = np.zeros(edges_in.shape).astype(bool)
    for label in np.unique(labels):
        if label != 0: 
            label_mask = (labels == label)
            values = edges_in[label_mask]

            if (values > high).any():
                num_edges_high = num_edges_high + 1
                mask = mask | label_mask

    bitwise_img = mask.astype(np.float32)

    ######################################################
    return bitwise_img, num_edges_low, num_edges_high

if __name__ == '__main__':
    images = ['beardman.jpg']

    for img_name in images:
        # Define behavior of the show_image function. You can change these variables if necessary
        save_image = False
        matplotlib_plotting = False

        # Read image. You can change the image you want to process here.
        current_path = Path(__file__).parent
        img_gray = cv2.imread(str(current_path.joinpath("image/" + img_name)), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise FileNotFoundError("Couldn't load image in " + str(current_path))

        # Before we start working with the image, we convert it from uint8 with range [0,255] to float32 with range [0,1]
        img_gray = img_gray.astype(np.float32) / 255.
        show_image(img_gray, "Original Image", save_image=save_image, use_matplotlib=matplotlib_plotting)

        # 1. Blur Image
        sigma = 3
        img_blur = blur_gauss(img_gray, sigma)

        # 2. Edge Detection
        gradients, orientations = sobel(img_blur)

        # 3. Non-Maxima Suppression
        edges = non_max(gradients, orientations)

        # 4. Hysteresis Thresholding
        hyst_method_auto = False
        threshold_test = False

        low_thresh = np.arange(0, 1, 0.1)
        high_thresh = np.arange(0, 1, 0.1)

        Low, High = np.meshgrid(low_thresh, high_thresh)

        num_edges_low_matrix = np.zeros((len(low_thresh), len(high_thresh)))
        num_edges_high_matrix = np.zeros((len(low_thresh), len(high_thresh)))

        for i, low in enumerate(low_thresh):
            for j, high in enumerate(high_thresh):
                if high < low:
                    num_edges_low_matrix[j, i] = np.nan
                    num_edges_high_matrix[j, i] = np.nan
                    continue

                _, num_low, num_high = hyst_thresh_studio(edges, low, high)
                num_edges_low_matrix[j, i] = num_low
                num_edges_high_matrix[j, i] = num_high

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(Low, High, num_edges_high_matrix, cmap='viridis', edgecolor='none')
        ax.set_xlabel('High threshold')
        ax.set_ylabel('Low threshold')
        ax.set_zlabel('Number of edges')
        ax.set_title("Number of detected edges")
        plt.show()

    # Destroy all OpenCV windows in case we have any open
    cv2.destroyAllWindows()