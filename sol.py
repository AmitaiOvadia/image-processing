import numpy as np
from skimage.color import rgb2gray
from skimage.color import rgb2yiq
from imageio import imread, imwrite
import matplotlib as matpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg





GRAY_SCALE = 1
RGB = 2
RGB_to_YIQ = np.array([[0.299, 0.587, 0.114],
                       [0.596, -0.275, -0.231],
                       [0.212, -0.523, 0.311]])
RGB_DIM = 3
GRAY_SCALE_DIM = 2
COLOR_AXIS = 2
HIGHEST_COLOR_VALUE = 255

# private methods

def image_dimensions(image):
    return len(image.shape)


def plot_histogram(cumulative_norm):
    x_axis = np.arange(256)
    plt.plot(x_axis, cumulative_norm)
    plt.show()


def get_histogram(image):
    return np.histogram(image, 256, (0, HIGHEST_COLOR_VALUE))[0]


def get_cumulative_histogram(histogram):
    return np.cumsum(histogram)


def display_equalization_results(hist_eq, hist_orig, im_eq, im_orig):
    cumulative_equalized_histogram = get_cumulative_histogram(hist_eq)
    cumulative_norm = get_cumulative_norm(hist_orig)
    plt.imshow(im_orig, cmap='gray')
    plt.show()
    plot_histogram(hist_orig)
    plot_histogram(cumulative_norm)
    plt.imshow(rgb2gray(im_eq), cmap='gray')
    plt.show()
    plot_histogram(hist_eq)
    plot_histogram(cumulative_equalized_histogram)


def get_equalized_img(cumulative_norm, img):
    first_nonzero_inx = np.where(cumulative_norm != 0)[0][0]  # first nonzero
    last_value = cumulative_norm[len(cumulative_norm) - 1]
    # look-out table: T(k) = (C(k) - C(m))/(c(last) - C(m)))*255
    # and turning back to int32
    T = (((cumulative_norm - first_nonzero_inx) /
          (last_value - first_nonzero_inx)) * HIGHEST_COLOR_VALUE).astype(
        np.int32)
    # each pixel with intensity k will be mapped to T[k]
    im_eq = T[img]
    return im_eq


def get_cumulative_norm(hist_orig):
    orig_cumulative = get_cumulative_histogram(
        hist_orig)  # cumulative histogram
    orig_cumulative_float = orig_cumulative.astype(
        np.float64)  # cumulative as float
    max_val = np.amax(orig_cumulative_float)  # max value
    cumulative_norm = ((
                                   orig_cumulative_float / max_val) *  # normalize cumulative histogram
                       HIGHEST_COLOR_VALUE).astype(np.int32)  # back to int
    return cumulative_norm


def get_workable_img(im_orig, im_orig_yiq):
    img = None
    if image_dimensions(im_orig) == GRAY_SCALE_DIM:
        img = im_orig
    if image_dimensions(im_orig) == RGB_DIM:
        # convert to YIQ, img will represent only the Y axis
        img = im_orig_yiq[:, :, 0]
    # now img is gray scale image
    img = (img * HIGHEST_COLOR_VALUE / np.amax(img)).astype(
        np.int32)  # make sure it's normalized to 255
    return img


# API


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
    vector in the image's indexes with the conversion matrix
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
    *i means the number of line in algorithem in ex1
    """
    if image_dimensions(im_orig) == RGB_DIM:
        # convert to YIQ, img will represent only the Y axis
        im_orig_yiq = rgb2yiq(im_orig)
        img = get_workable_img(im_orig, im_orig_yiq)
    elif image_dimensions(im_orig) == GRAY_SCALE_DIM:
        img = get_workable_img(im_orig, 0)
    hist_orig = get_histogram(img)   # computing histogram (*1)
    cumulative_norm = get_cumulative_norm(hist_orig)  # (*2 + *3 + *4)
    im_eq = get_equalized_img(cumulative_norm, img)  # (*5 + *6 + *7)
    hist_eq = get_histogram(im_eq)
    if image_dimensions(im_orig) == RGB_DIM:
        # assigning the y axis of the original imagE in the YIQ space with the
        # new equaliazed image
        im_orig_yiq[:, :, 0] = im_eq/HIGHEST_COLOR_VALUE
        # transforming back to rgb and assigning to im_eq
        im_eq = yiq2rgb(im_orig_yiq)
    im_eq = im_eq/HIGHEST_COLOR_VALUE  # normalize again to [0,1] values
    # display_equalization_results(hist_eq, hist_orig, im_eq, im_orig)
    return [im_eq, hist_orig, hist_eq]

def quantize(im_orig, n_quant, n_iter):
    """
    performs optimal quantization of a given grayscale or RGB image.
    If an RGB image is given, the quantization procedure operates on the Y channel of the
    corresponding YIQ image and then convert back from YIQ to RGB
    solves an optimization problem: min()
    :param im_orig: is the input grayscale or RGB image to be quantized (float64 image with values in [0, 1]).
    :param n_quant: is the number of intensities your output im_quant image should have.
    :param n_iter: is the maximum number of iterations of the optimization procedure (may converge earlier.)
    :return: output is a list [im_quant, error]
    im_quant - is the quantized output image. (float64 image with values in [0, 1]).
    error - is an array with shape (n_iter,) (or less) of the total intensities error for each iteration of the
            quantization procedure.
    """