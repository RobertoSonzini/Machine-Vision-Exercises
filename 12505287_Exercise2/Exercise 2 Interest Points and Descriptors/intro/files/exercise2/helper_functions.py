#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Machine Vision and Cognitive Robotics (376.054)
Exercise 2: Harris Corners
Clara Haider, Matthias Hirschmanner 2020
Automation & Control Institute, TU Wien

Tutors: machinevision@acin.tuwien.ac.at
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2


def show_corners(img: np.ndarray,
                 i_xx: np.ndarray,
                 i_yy: np.ndarray,
                 i_xy: np.ndarray,
                 g_xx: np.ndarray,
                 g_yy: np.ndarray,
                 g_xy: np.ndarray,
                 h_dense: np.ndarray,
                 h_nonmax: np.ndarray,
                 corners: np.ndarray,
                 debug_corners: bool = False,
                 save_image: bool = False) -> None:
    """ Plot the corners in an image and optionally the intermediate steps

    :param img: Grayscale input image
    :param i_xx: squared input image filtered with derivative of gaussian in x-direction
    :param i_yy: squared input image filtered with derivative of gaussian in y-direction
    :param i_xy: Multiplication of input image filtered with derivative of gaussian in x- and y-direction
    :param g_xx: i_xx filtered by larger gaussian
    :param g_yy: i_yy filtered by larger gaussian
    :param g_xy: i_xy filtered by larger gaussian
    :param h_dense: Result of harris calculation for every pixel. Array of same size as input image.
            Values normalized to 0-1
    :param h_nonmax: Binary mask of non-maxima suppression. Array of same size as input image.
            1 where values are NOT suppressed, 0 where they are.
    :param corners: n x 3 matrix containing all detected corners after thresholding and non-maxima suppression.
            Every row vector represents a corner with the elements [y, x, d]
            (d is the result of the harris calculation)
    :param debug_corners: If True will also plot all intermediate images
    :param save_image: If True will save the figure
    :return: None
    """

    if not debug_corners:
        plt.imshow(img, cmap='gray')

        # plot red X at Corners
        plt.scatter(x=corners[:, 1], y=corners[:, 0], c='red', marker='x', lw=0.3)
        if save_image:
            plt.savefig('corners.png', dpi=300)
        plt.show()

    else:
        i_xx = i_xx / np.amax(i_xx)
        i_yy = i_yy / np.amax(i_yy)
        i_xy = i_xy / np.amax(i_xy)
        g_xx = g_xx / np.amax(g_xx)
        g_yy = g_yy / np.amax(g_yy)
        g_xy = g_xy / np.amax(g_xy)
        h_nonmax = h_nonmax / np.amax(h_nonmax)
        h_dense = h_dense / np.amax(h_dense)

        fig, ax = plt.subplots(3, 3, figsize=(20, 10))
        plt.subplot(3, 3, 1)
        plt.title("I")
        plt.imshow(img, cmap='gray')
        plt.scatter(x=corners[:, 1], y=corners[:, 0], c='red', marker='x', s=10, lw=0.3)

        plt.subplot(3, 3, 2)
        plt.title("Ixx")
        plt.imshow(i_xx, cmap='gray')

        plt.subplot(3, 3, 3)
        plt.title("Iyy")
        plt.imshow(i_yy, cmap='gray')

        plt.subplot(3, 3, 4)
        plt.title("Ixy")
        plt.imshow(i_xy, cmap='gray', vmin=0, vmax=1)

        plt.subplot(3, 3, 5)
        plt.title("Gxx")
        plt.imshow(g_xx, cmap='gray')

        plt.subplot(3, 3, 6)
        plt.title("Gyy")
        plt.imshow(g_yy, cmap='gray')

        plt.subplot(3, 3, 7)
        plt.title("Gxy")
        plt.imshow(g_xy, cmap='gray', vmin=0, vmax=1)

        plt.subplot(3, 3, 8)
        plt.title("Hdense")
        plt.imshow(h_dense, cmap='gray')

        plt.subplot(3, 3, 9)
        plt.title("Hnonmax")
        plt.imshow(h_nonmax, cmap='gray')
        if save_image:
            plt.savefig('debug_image.png', dpi=300)
        plt.show()


def show_matches(img_1: np.ndarray,
                 img_2: np.ndarray,
                 corners_1: np.ndarray,
                 corners_2: np.ndarray,
                 matches: np.ndarray,
                 save_image: bool = False):
    """ Plot the matches between two images

    :param img_1: First grayscale image
    :type img_1: np.ndarray with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param img_2: Second grayscale image
    :type img_2: np.ndarray with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param corners_1: Array containing the coordinates of all detected corners in the first image
    :type corners_1: np.ndarray with shape (n, 2) where each row consists of the [y, x] coordinates of an corner

    :param corners_2: Array containing the coordinates of all detected corners in the second image
    :type corners_2: np.ndarray with shape (n, 2) where each row consists of the [y, x] coordinates of an corner

    :param matches: Array representing the successful matches. Each row contains the indices of the matches descriptors
    :param save_image: np.ndarray with shape (k, 2) with k being the number of matches.
        Each row contains the index of the corner in img_1 and img_2 respectively: [idx_1, idx_2]

    :return: None
    """
    height_a, width_a = img_1.shape
    height_b, width_b = img_2.shape

    if height_a < height_b:
        img_1 = np.r_[img_1, np.zeros((height_b - height_a, width_a))]
    elif height_b < height_a:
        img_2 = np.r_[img_2, np.zeros((height_a - height_b, width_b))]

    img = np.c_[img_1, img_2]

    plt.figure(2)
    plt.imshow(img, cmap='gray')

    # plot red X at Corners
    plt.scatter(x=corners_1[:, 1], y=corners_1[:, 0], c='red', marker='x', lw=0.3)
    plt.scatter(x=corners_2[:, 1] + width_a, y=corners_2[:, 0], c='red', marker='x', lw=0.3)

    for i in range(0, matches.shape[0]):
        y = np.c_[corners_1[int(matches[i, 0]), 0], corners_2[int(matches[i, 1]), 0]].flatten()
        x = np.c_[corners_1[int(matches[i, 0]), 1], corners_2[int(matches[i, 1]), 1] + width_a].flatten()
        color = np.random.uniform(0, 1, 3)

        plt.plot(x, y, c=color, lw=0.5)

    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if save_image:
        plt.savefig('matches.png', dpi=300)
    plt.show()


def circle_mask(dsize: int) -> np.ndarray:
    """ Returns a circular Boolean mask

    :param dsize: Size of the desired mask
    :type dsize: int

    :return: Boolean Mask with True Values in a circle of the array
        e.g. for a 4x4 array:   |FTTF|
                                |TTTT|
                                |TTTT|
                                |FTTF|
    :rtype: np.ndarray with shape (dsize, dsize)
    """
    radius = dsize/2
    row, col = np.mgrid[1:dsize+1, 1:dsize+1]
    mask = np.power(row - radius - 0.5, 2) + np.power(col - radius - 0.5, 2) <= np.power(radius, 2)
    return mask


# Adapted from https://github.com/jrosebr1/imutils/blob/master/imutils/convenience.py#L41
def rotate_bound(img: np.ndarray, angle: float) -> np.ndarray:
    """ Rotate an image by the angle and return it with the additional pixels filled with replicated border

    :param img: Grayscale input image
    :type img: np.ndarray with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param angle: The angle the image will be rotated in degree
    :type angle: float

    :return: Resulting image with the rotated original image and filled borders
    :rtype: np.ndarray with shape (new_height, new_width) with dtype np.float32 an values in range [0., 1.]
    """
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(img, M, (nW, nH), borderMode=cv2.BORDER_REPLICATE)
