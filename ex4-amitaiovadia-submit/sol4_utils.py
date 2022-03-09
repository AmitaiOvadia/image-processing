import numpy as np
import scipy
from skimage.color import rgb2gray
import matplotlib.image as mpimg
from scipy.ndimage.filters import convolve
from scipy.signal import convolve2d
import os
import matplotlib.pyplot as plt


GRAY_TXT = 'gray'
BASE_ARRAY = np.array([1, 1])
MIN_DIMENSIONS = 16
ROWS_NUM = 0
COL_NUM = 1
ADD_BELOW = 0
ADD_TO_THE_RIGHT = 1

GRAY_SCALE = 1
RGB = 2

RGB_DIM = 3
GRAY_SCALE_DIM = 2
COLOR_AXIS = 2
HIGHEST_COLOR_VALUE = 255
NUMBER_OF_COLORS = 256

def image_dimensions(image):
    """
    :param image: image rgb or grayscale
    :return: int: number of dimensions
    """
    return len(image.shape)


def read_image(filename, representation):
    """
    A function which reads an image file and converts it into a given
    representation.
    :param filename: <string> the filename of an image in [0,1] intensities (grayscale or RGB).
    :param representation: <int> gray scale image (1) or an RGB image (2)
    :return: an image, a matrix of type np.float64 with intensities with
             intensities normalized to the range [0, 1].
    """
    img = mpimg.imread(filename)  # reading image
    img_float64 = img.astype(np.float64)  # converting from float32 to float64
    if img_float64.max() > 1:
        img_float64 /= HIGHEST_COLOR_VALUE  # normalize
    if image_dimensions(img) == GRAY_SCALE_DIM and representation == GRAY_SCALE:
        return img_float64  # checks if image is already gray scale
    if representation == GRAY_SCALE:   # image must be rgb
        return rgb2gray(img_float64)   # convert from rgb to gray scale
    if representation == RGB:
        return img_float64             # no need to convert
    return None



def create_filter_vec(filter_size):
    """
    :param filter_size: the size of the desired Gaussian filter
    :return: a filter_size numpy array of the n'th roe binomial coefficients
    """
    filter_vec = np.poly1d(BASE_ARRAY)**(filter_size-1)
    filter_vec = np.array([filter_vec.coef]).astype(np.float64)
    filter_sum = np.sum(filter_vec)
    filter_vec /= filter_sum  # normalize filter
    return filter_vec

def image_big_enough(im):
    """
    checks if image is big enough to expand pyramid
    :param im: 2d image
    :return: boolean
    """
    N, M = im.shape
    if N >= MIN_DIMENSIONS*2 and M >= MIN_DIMENSIONS*2: return True
    return False

def reduce(img, filter_vec):
    """
    creating an image of size M/2 * N/2
    :param img: 2d image
    :param filter_vec: np.array(1, filter_size) for blurring
    :return: image of size M/2 * N/2
    """
    img = convolve(img, filter_vec)  # blur rows
    img = convolve(img, filter_vec.transpose())  # blur columns
    img = img[::2]  # take every second pixel in row
    img = (img.transpose()[::2]).transpose()  # take every second pixel in columns
    return img

def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter (filter_size = 3 you should get [0.25, 0.5, 0.25]).
                        filter size will be >=2.
    :return: pyr: a python list with maximum length of max_levels, where each element of the array is a grayscale image.
             filter_vec: row vector of shape (1, filter_size) used
                    for the pyramid construction. This filter should be built using a consequent filter_size - 1
                    1D convolutions of [1 1] with itself in order to derive a row of the binomial coefficients
                    The filter_vec is normalized.
    """
    im = im.astype(np.float64)
    filter_vec = create_filter_vec(filter_size)  # creating binomial coefficients vector
    pyr = [im]
    while image_big_enough(pyr[-1]) and len(pyr) < max_levels:  # while x,y of image >= 16 and pyramid < max_levels
        last_image_in_pyramid = pyr[-1]
        reduced = reduce(last_image_in_pyramid, filter_vec)  # reducing the last image in the pyramid using the filter_vec to blur
        pyr.append(reduced)
    return pyr, filter_vec

from scipy.signal import convolve2d
import numpy as np


def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img



