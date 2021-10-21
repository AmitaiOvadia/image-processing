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
COLOR_AXIS = 2

def image_dimensions(image):
    return len(image.shape)

def read_image(filename, representation):
    """
    A function which reads an image file and converts it into a given
    representation.
    :param filename: <string> the filename of an image (grayscale or RGB).
    :param representation: <int> gray scale image (1) or an RGB image (2)
    :return: an image, a matrix of type np.float64 with intensities with
             intensities normalized to the range [0, 1].
    """
    img = mpimg.imread(filename)  # reading image
    img_float64 = img.astype(np.float64)  # converting from float32 to float64
    if image_dimensions(img) == GRAY_SCALE_DIM and representation == GRAY_SCALE:
        return img_float64  # checks if image is already gray scale
    if representation == GRAY_SCALE:
        return rgb2gray(img_float64)   # convert from rgb to gray scale
    if representation == RGB:
        return img_float64             # no need to convert
    return None


def imdisplay(filename, representation):
    """
    displaying an image given it's representation
    :param filename: <string> the filename of an image (grayscale or RGB).
    :param representation: the representation in which to display the image
           <int> gray scale image (1) or an RGB image (2)
    :return: nothing
    """
    image = read_image(filename, representation)
    if representation == GRAY_SCALE:
        if image_dimensions(image) == GRAY_SCALE_DIM:
            plt.imshow(image, cmap='gray')
            plt.show()
        elif image_dimensions(image) == RGB_DIM:
            # if image is RGB and representation is gray scale: convert to gray
            image = rgb2gray(image)  # convert to gray scale
            plt.imshow(image, cmap='gray')
            plt.show()
    elif representation == RGB:
        if image_dimensions(image) == RGB_DIM:
            plt.imshow(image)
            plt.show()


def rgb2yiq(imRGB):
    """
    converting image from RGB representation to YIQ by multiplying each color
    vector in the image'e indexes with the convertion matrix
    :param imRGB: an RGB image
    :return: a YIQ image
    """
    # multiplying each color vector (the 2th axis) with the conversion matrix
    yiq_image = np.tensordot(imRGB, RGB_to_YIQ, axes=([COLOR_AXIS], [0]))
    return yiq_image

def yiq2rgb(imYIQ):
    """
    converting image from YIQ representation to RGB by multiplying each color
    vector in the image'e indexes with the conversion matrix inv(RGB_to_YIQ)
    :param imYIQ: a YIQ image
    :return: a RGB image
    """
    YIQ_to_RGB = np.linalg.inv(RGB_to_YIQ)  # the inverse of the RGB_to_YIQ mat
    # multiplying each color vector (the 2th axis) with the conversion matrix
    rgb_image = np.tensordot(imYIQ, YIQ_to_RGB, axes=([COLOR_AXIS], [0]))
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
    img = None
    if image_dimensions(im_orig) == GRAY_SCALE_DIM:
        img = im_orig
    if image_dimensions(im_orig) == RGB_DIM:
        # convert to YIQ, im will represent only the Y axis
        img = rgb2yiq(im_orig)[:, :, 0]
    # now img is gray scale image
    img = img * 265
    img = img.astype(np.int32)
    orig_hist, bounds = np.histogram(img, 256, (0, 255))
    # print(orig_hist)
    # orig_hist = np.divide(orig_hist, orig_hist.max)
    print(orig_hist)
    # plt.tick_params(labelsize=10)
    # plt.hist(img.flatten(), bins=128)
    # plt.show()


    # todo if the original image was rgb: convert back to rgb