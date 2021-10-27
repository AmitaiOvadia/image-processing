import numpy as np
from scipy.misc import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

COLOR_GRAYSCALE = 1
COLOR_RGB = 2
MAX_COLOR_RANGE = 255
COLOR_COUNT = MAX_COLOR_RANGE + 1
RGB_TO_YIQ = np.array([[0.299, 0.587, 0.114],
                       [0.596, -0.275, -0.321],
                       [0.212, -0.523, 0.311]])
YIQ_TO_RGB = np.linalg.inv(RGB_TO_YIQ)
RGB_SHAPE = 3


def read_image(filename, representation):
    """
    Reads an image file and converts it into a given representation.
    :param filename: the filename of an image.
    :param representation: - representation code, either 1 or 2 defining whether
           the output should be a grayscale image (1) or an RGB image (2)
    :return: An image.
    """
    im = imread(filename).astype(np.float64)
    if representation == COLOR_GRAYSCALE:
        im = rgb2gray(im)
    if im.max() > 1:
        im /= MAX_COLOR_RANGE
    return im


def matrix_display(mat, representation):
    """
    display an image represented as a 2D array.
    :param mat: 2D np array
    :param representation: The representation of the image.
    """
    if representation == COLOR_GRAYSCALE:
        plt.imshow(mat, cmap=plt.cm.gray)
    else:
        plt.imshow(mat)
    plt.show()


def imdisplay(filename, representation):
    """
    display an image in a given representation
    :param filename: the filename of an image.
    :param representation: - representation code, either 1 or 2 defining whether
           the output should be a grayscale image (1) or an RGB image (2)
    """
    matrix_display(read_image(filename, representation), representation)


def rgb2yiq(imRGB):
    """
    RGB image to YIQ.
    :param imRGB: RGB image, height×width×3 np.float64 matrix
    :return: YIQ image
    """
    return imRGB.dot(RGB_TO_YIQ.T)


def yiq2rgb(imYIQ):
    """
    YIQ image to RGB.
    :param imYIQ: height×width×3 np.float64 matrix
    :return: RGB image.
    """
    return imYIQ.dot(YIQ_TO_RGB.T)


def histogram_equalize_helper(im_orig):
    """
    A helper function for the equalization that equalizes an one color image.
    :param im_orig: The image with one color.
    :return: a list [im_eq, hist_orig, hist_eq] where
         im_eq - is the equalized image. grayscale or RGB float64
         image with values in [0, 1].
         hist_orig - is a 256 bin histogram of the original image
         (array with shape (256,) ).
         hist_eq - is a 256 bin histogram of the equalized image
         (array with shape (256,) ).
    """
    im = (im_orig * MAX_COLOR_RANGE).round().astype(np.uint8)
    hist_orig, bins = np.histogram(im, COLOR_COUNT, [0, MAX_COLOR_RANGE])
    width, height = im_orig.shape
    hist_cum = np.cumsum(hist_orig).astype(np.float64)
    hist_cum /= (width * height)
    # now hist_cum is equalized
    hist_cum *= MAX_COLOR_RANGE
    hist_eq_cum = hist_cum.copy()

    if hist_cum.min() != 0 or hist_cum.max() != MAX_COLOR_RANGE:
        first_nonzero_index = hist_cum.nonzero()[0][0]
        hist_eq_cum -= hist_cum[first_nonzero_index]
        hist_eq_cum /= (hist_cum[MAX_COLOR_RANGE] - hist_cum[first_nonzero_index])
        hist_eq_cum *= MAX_COLOR_RANGE

    hist_eq_cum = np.round(hist_eq_cum).astype(np.uint8)
    im_eq = hist_eq_cum[im].astype(np.float64)
    hist_eq, bins = np.histogram(im_eq, COLOR_COUNT, [0, MAX_COLOR_RANGE])
    im_eq /= MAX_COLOR_RANGE
    return [im_eq, hist_orig, hist_eq]


def histogram_equalize_rgb(im_orig):
    """
    Equalize an rgb image.
    :param im_orig: is the input RGB float64 image with values in [0, 1]
    :return: a list [im_eq, hist_orig, hist_eq] where
             im_eq - is the equalized image. grayscale or RGB float64
             image with values in [0, 1].
             hist_orig - is a 256 bin histogram of the original image
             (array with shape (256,) ).
             hist_eq - is a 256 bin histogram of the equalized image
             (array with shape (256,) ).
    """
    im_yiq = rgb2yiq(im_orig)
    im_eq_y, hist_orig, hist_eq = histogram_equalize_helper(im_yiq[:, :, 0])
    im_yiq[:, :, 0] = im_eq_y
    im_eq = yiq2rgb(im_yiq)
    return [np.clip(im_eq, 0, 1), hist_orig, hist_eq]


def histogram_equalize(im_orig):
    """
    Performs histogram equalization of a given grayscale or RGB image
    :param im_orig: is the input grayscale or RGB float64 image with
                    values in [0, 1]
    :return: a list [im_eq, hist_orig, hist_eq] where
             im_eq - is the equalized image. grayscale or RGB float64
             image with values in [0, 1].
             hist_orig - is a 256 bin histogram of the original image
             (array with shape (256,) ).
             hist_eq - is a 256 bin histogram of the equalized image
             (array with shape (256,) ).
    """
    if len(im_orig.shape) == RGB_SHAPE:
        return histogram_equalize_rgb(im_orig)
    return histogram_equalize_helper(im_orig)


def create_bin_to_q_map(q, z):
    """
    Create a mapping of bins to q values.
    :param q: q values
    :param z: The segmentation of the colors.
    :return: The mapping.
    """
    # Mapping the right border of each segment to the segment to its right.
    # The last element will be added to the last segment.
    return np.concatenate([np.repeat(q[i], z[i + 1] - z[i]) for i in range(len(q))] + [np.array([q[-1]])])


def get_quantization_error(z, q, hist):
    """
    Get the quantization error.
    :param z: The segmentation of the colors.
    :param q: q values
    :param hist: The histogram of the image.
    :return: The error.
    """
    return ((create_bin_to_q_map(q.round(), z) - np.arange(COLOR_COUNT))**2 * hist).sum()


def get_initial_segment_division(n_quant, hist, pixel_count):
    """
    Get an initial value
    :param n_quant: is the number of intensities your output im_quant
                    image should have.
    :param hist: The histogram of the image.
    :param pixel_count: The number of pixels in the image
    :return:
    """
    hist_cum = np.cumsum(hist)
    z = [0]
    desired_segment_pixels = pixel_count / n_quant
    for i in range(n_quant - 1):
        indices = np.where(hist_cum < desired_segment_pixels * (i + 1))[0]
        z.append(indices[-1])
    z.append(MAX_COLOR_RANGE)
    return np.array(z, dtype=np.uint8)


def compute_z(q):
    """
    Compute the optimal z values based on the algorithm.
    :param q: q values
    :return: z values (segmentation).
    """
    return np.ceil(np.concatenate([[0], (q[:-1] + q[1:]) / 2,
                                   [MAX_COLOR_RANGE]])).astype(np.uint8)


def compute_q(z, hist, n_quant):
    """
    Compute the optimal q values based on the algorithm.
    :param z: z values (segmentation).
    :param hist: The histogram of the image.
    :param n_quant: is the number of intensities your output im_quant
                    image should have.
    :return: q values
    """
    q = []
    for i in range(n_quant):
        bins = np.arange(z[i], z[i + 1])
        q.append((bins * hist[bins]).sum().astype(np.float64) / hist[bins].sum())
    return np.array(q, dtype=np.float64)


def quantize_helper(im_orig_axis, n_quant, n_iter):
    """
    helps performing optimal quantization of a given one color image.
    :param im_orig_axis: The image with one color.
    :param n_quant: is the number of intensities your output im_quant
                    image should have.
    :param n_iter: is the number of intensities your output im_quant image
                    should have.
    :return: a list [im_quant, error] where
             im_quant - is the quantized output image, one color.
             error - is an array with shape (n_iter,) (or less) of the total
             intensities error for each iteration of the quantization procedure
    """
    im = (im_orig_axis * MAX_COLOR_RANGE).round().astype(np.uint8)
    hist, bins = np.histogram(im, COLOR_COUNT, [0, MAX_COLOR_RANGE])
    width, height = im.shape
    z = get_initial_segment_division(n_quant, hist, width * height)
    old_z = z
    errors = []
    q = None
    for i in range(n_iter):
        q = compute_q(z, hist, n_quant)
        z = compute_z(q)
        errors.append(get_quantization_error(z, q, hist))
        if (z == old_z).all():
            break
        old_z = z

    im_quant = create_bin_to_q_map(q, z)[im].astype(np.float64) / MAX_COLOR_RANGE
    return [im_quant, np.array(errors, dtype=np.float64)]


def quantize(im_orig, n_quant, n_iter):
    """
    Performs optimal quantization of a given grayscale or RGB image
    :param im_orig: is the input grayscale or RGB image to be quantized
                    (float64 image with values in [0, 1]).
    :param n_quant: is the number of intensities your output im_quant
                    image should have.
    :param n_iter: is the number of intensities your output im_quant image
                    should have.
    :return: a list [im_quant, error] where
             im_quant - is the quantized output image.
             error - is an array with shape (n_iter,) (or less) of the total
             intensities error for each iteration of the quantization procedure
    """
    if len(im_orig.shape) == RGB_SHAPE:
        im_yiq = rgb2yiq(im_orig)
        im_quant_y, errors = quantize_helper(im_yiq[:, :, 0], n_quant, n_iter)
        im_yiq[:, :, 0] = im_quant_y
        im_quant = yiq2rgb(im_yiq)
        return [im_quant, errors]
    return quantize_helper(im_orig, n_quant, n_iter)
