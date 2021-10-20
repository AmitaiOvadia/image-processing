import numpy as np
from skimage.color import rgb2gray
from imageio import imread, imwrite
import matplotlib as matpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

x = np.hstack([np.repeat(np.arange(0,50,2),10)[None,:], np.array([255]*6)[None,:]])
grad = np.tile(x,(256,1))
GRAY_SCALE = 1
RGB = 2
RGB_to_YIQ = np.array([[0.299, 0.587, 0.114],
                       [0.596, -0.275, -0.231],
                       [0.212, -0.523, 0.311]])
RGB_DIM = 3
GRAY_SCALE_DIM = 2

def read_image(filename, representation):
    """
    A function which reads an image file and converts it into a given
    representation.
    :param filename: <string> the filename of an image (grayscale or RGB).
    :param representation: <int> gray scale image (1) or an RGB image (2)
    :return: an image, a matrix of type np.float64 with intensities with
             intensities normalized to the range [0, 1].
    """
    img = mpimg.imread(filename)
    img_float64 = img.astype(np.float64)
    if representation == GRAY_SCALE: return rgb2gray(img_float64)
    if representation == RGB: return img_float64
    return None

def imdisplay(filename, representation):
    """
    displaying an image given it's representation
    :param filename: <string> the filename of an image (grayscale or RGB).
    :param representation: <int> gray scale image (1) or an RGB image (2)
    :return: nothing
    """
    image = read_image(filename, representation)
    if representation == GRAY_SCALE:
        plt.imshow(image, cmap='gray')
        plt.show()
    if representation == RGB:
        plt.imshow(image)
        plt.show()

def rgb2yiq(imRGB):
    """
    converting image from RGB representation to YIQ by multiplying each color
    vector in the image'e indexes with the convertion matrix
    :param imRGB: an RGB image
    :return: a YIQ image
    """
    yiq_image = np.tensordot(imRGB, RGB_to_YIQ, axes=([2], [0]))
    # multiplying each color vector (the 2th axis) with the conversion matrix
    return yiq_image

def yiq2rgb(imYIQ):
    """
    converting image from YIQ representation to RGB by multiplying each color
    vector in the image'e indexes with the conversion matrix inv(RGB_to_YIQ)
    :param imYIQ: a YIQ image
    :return: a RGB image
    """
    YIQ_to_RGB = np.linalg.inv(RGB_to_YIQ)  # the inverse of the RGB_to_YIQ mat
    rgb_image = np.tensordot(imYIQ, YIQ_to_RGB, axes=([2], [0]))
    # multiplying each color vector (the 2th axis) with the conversion matrix
    return rgb_image

def histogram_equalize(im_orig):
    """
    Performs histogram equalization
    :param im_orig: gray scale or RGB float64 image with values in [0, 1].
    :return: a list of [im_eq, hist_orig, hist_eq]
    im_eq = equalized image same characteristics as the im_orig
    hist_orig = original histogram
    hist_eq = the equalized histogram
    """
    im = None
    if len(im_orig.shape) == GRAY_SCALE_DIM:
        im = im_orig
    if len(im_orig.shape) == RGB_DIM:
        # convert to YIQ, work only on Y axis
        im = rgb2yiq(im_orig)[:, :, 0]
    plt.tick_params(labelsize=10)
    plt.hist(im.flatten(), bins=128)
    plt.show()