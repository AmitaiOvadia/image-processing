import numpy as np
from skimage.color import rgb2gray
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
NUMBER_OF_COLORS = 256
GRAY_TXT = 'gray'

# private methods

def image_dimensions(image):
    """
    :param image: image rgb or grayscale
    :return: int: number of dimensions
    """
    return len(image.shape)


def plot_histogram(histogram):
    """
    plot histogram
    :param histogram: np.array(256,)
    :return: None
    """
    x_axis = np.arange(NUMBER_OF_COLORS)
    plt.plot(x_axis, histogram)
    plt.show()


def get_histogram(image):
    """
    :param image: 2D image
    :return: histogram: np.array(256,)
    """
    return np.histogram(image, NUMBER_OF_COLORS, (0, HIGHEST_COLOR_VALUE))[0]


def get_cumulative_histogram(histogram):
    """
    :param histogram: np.array(256,)
    :return: cumulative histogram: np.array(256,)
    """
    return np.cumsum(histogram)


def get_equalized_img(cumulative_norm, img):
    """
    :param cumulative_norm: cumulative histogram np.array(256,)
    :param img: 2D image [0,255]
    :return: im_eq: 2D image [0,255] after equalization
    """
    first_nonzero_inx = np.where(cumulative_norm != 0)[0][0]  # first nonzero
    last_value = cumulative_norm[len(cumulative_norm) - 1]
    # look-out table: T(k) = (C(k) - C(m))/(c(last) - C(m)))*255
    # and turning back to int32
    T = (((cumulative_norm - first_nonzero_inx) /  # stretching the axes to 0 and 255
          (last_value - first_nonzero_inx)) * HIGHEST_COLOR_VALUE).astype(np.int32)
    # each pixel with intensity k will be mapped to T[k]
    im_eq = T[img]
    return im_eq


def get_cumulative_norm(hist_orig):
    """
    :param hist_orig: original histogram np.array(256,)
    :return: a cumulative histogram normalized to 255 np.array(256,)
    """
    orig_cumulative = get_cumulative_histogram(hist_orig)  # cumulative histogram
    orig_cumulative_float = orig_cumulative.astype(np.float64)  # cumulative as float
    max_val = np.amax(orig_cumulative_float)  # max value
    cumulative_norm = ((orig_cumulative_float / max_val) *  # normalize cumulative histogram
                       HIGHEST_COLOR_VALUE).astype(np.int32)  # back to int
    return cumulative_norm


def get_workable_img(im_orig, im_orig_yiq):
    """
    :param im_orig: 3D rgb image or 2D grayscale image
    :param im_orig_yiq: 3D yiq version of im_orig
    :return: 2D [0,255] int image
    """
    img = None
    if image_dimensions(im_orig) == GRAY_SCALE_DIM:
        img = im_orig
    if image_dimensions(im_orig) == RGB_DIM:
        # convert to YIQ, img will represent only the Y axis
        img = im_orig_yiq[:, :, 0]
    # now img is gray scale image
    img = (img * HIGHEST_COLOR_VALUE / np.amax(img)).astype(np.int32)  # make sure it's normalized to 255
    return img

def back_to_rgb(img, im_orig):
    """
    :param img: 2D image
    :param im_orig: 3D rgb image
    :return: a merge between img and im_orig via YIQ
    """
    # assigning the y axis of the original imagE in the YIQ space with the
    # new equaliazed image
    im_orig_yiq = rgb2yiq(im_orig)
    im_orig_yiq[:, :, 0] = img   # replacing the y axes with img
    # transforming back to rgb and assigning to ing
    img = yiq2rgb(im_orig_yiq)
    return img


def get_int_img_histograms(im_orig):
    """
    :param im_orig: original image as rgb or grayscale
    :return: cumulative_norm: cumulative histogram normalized to 256
             hist_orig: the original histogram
             img: 2D image at the range [0,255]
    """
    if image_dimensions(im_orig) == RGB_DIM:
        # convert to YIQ, img will represent only the Y axis
        im_orig_yiq = rgb2yiq(im_orig)
        img = get_workable_img(im_orig, im_orig_yiq)
    elif image_dimensions(im_orig) == GRAY_SCALE_DIM:
        img = get_workable_img(im_orig, 0)
    hist_orig = get_histogram(img)  # computing histogram (*1)
    cumulative_norm = get_cumulative_norm(hist_orig)  # (*2 + *3 + *4)
    return cumulative_norm, hist_orig, img



def quantize_helper(im_orig, n_iter, n_quant):
    """
    :param im_orig: 2D image in float64
    :param n_iter: max number of optimization iterations
    :param n_quant: number of different colors
    :return: im_quant: quantized image
             error_arr: list of the errors in every iteration
    """
    img = (im_orig*HIGHEST_COLOR_VALUE).astype(np.int32)
    hist_orig = get_histogram(img)
    cumulative_norm = get_cumulative_norm(hist_orig)
    # cumulative_norm, hist_orig, img = get_int_img_histograms(im_orig)
    z_arr, q_arr = z_q_initialize(cumulative_norm, n_quant)
    error_arr = []
    for i in range(n_iter):  # optimization
        error = calculate_quantization_error(z_arr, q_arr, hist_orig)
        if (len(error_arr) >= 2 and error_arr[-1] == error_arr[-2]): break
        error_arr.append(error)
        q_arr = update_q_arr(z_arr, hist_orig, n_quant)
        z_arr = update_z_arr(q_arr, z_arr)
    im_quant = quant_by_q_z(img, q_arr, z_arr)
    im_quant = normalize_img(im_quant)  # normalize again to [0,1] values
    return im_quant, error_arr


def normalize_img(img):
    """
    :param img: [0,255] int image
    :return: [0,1] np.float64 image
    """
    return img.astype(np.float64)/HIGHEST_COLOR_VALUE

def update_q_arr(z_arr, hist_orig, n_quant):
    """
    updating the target colors in each optimization iteration according to the formula
    :param z_arr: list of the color boundaries for example: [  0  22  79 139 255]
    :param hist_orig: the original histogram
    :param n_quant: number of desired colors: for example 4
    :return: q_arr: the desired colors after 1 optimization step for example: [11, 46, 107, 162]
    """
    q_arr = []
    for i in range(n_quant):
        z_bins = np.arange(z_arr[i], z_arr[i+1])
        histogram_chunk_of_z_bins = hist_orig[z_bins]   # (z_bins,) vector
        mone = z_bins * histogram_chunk_of_z_bins  # z(j)*h(j)   (z_bins,) vector
        mechane = histogram_chunk_of_z_bins  # (z_bins,) vector
        if mechane.sum() == 0:
            q_i = 0
        else:
            q_i = int(mone.sum() // mechane.sum())
        q_arr.append(q_i)
    return q_arr


def update_z_arr(q_arr, z_arr):
    """
    :param q_arr: the desired colors for example: [11, 46, 107, 162]
    :param z_arr: list of the color boundaries for example: [  0  28  76 134 255]
    :return: updated z_arr by new q_arr
    """
    for i in range(1, len(q_arr)):
        z_arr[i] = (q_arr[i-1] + q_arr[i]) // 2
    return z_arr


def calculate_quantization_error(z_arr, q_arr, hist_orig):
    """
    calculating the quantization error for each iteration
    :param z_arr: list of the color boundaries for example: [  0  28  76 134 255]
    :param q_arr: the desired colors for example: [11, 46, 107, 162]
    :param hist_orig: original histogram np.array(256,)
    :return: the error calculated for example 6147088
    """
    qi = get_quantizised_histogram(q_arr, z_arr)  # array (256,) of all the qi's
    zi = np.arange(NUMBER_OF_COLORS)  # all the different z's (colors)
    # calculate for each z, what is the distance^2 from each color to it's new location
    dist_of_zi_to_qi_squared = (qi - zi)**2   # (256,) vector
    error_for_each_z = dist_of_zi_to_qi_squared*hist_orig  # (256,) vector, multiply x(i)*y(i)
    total_error = error_for_each_z.sum()  # scalar
    return total_error


def quant_by_q_z(img, q_arr, z_arr):
    """
    :param img: 2D [0,255] image
    :param q_arr: the desired colors for example: [11, 46, 107, 162]
    :param z_arr: list of the color boundaries for example: [  0  28  76 134 255]
    :return:
    """
    # the histogram that reflects the mapping of the z's to the right q's
    hist_quantized = get_quantizised_histogram(q_arr, z_arr)
    img = hist_quantized[img]    # each pixel with intensity k will be mapped to hist_quantized[k]
    return img


def get_quantizised_histogram(q_arr, z_arr):
    """
    assembling an array that represents the mapping of the colors bound by z_arr to their respective q_arr values
    :param q_arr: the desired colors for example: [11, 46, 107, 162]
    :param z_arr: list of the color boundaries for example: [  0  28  76 134 255]
    :return: hist_quantized: np.array(256,)
    """
    hist_quantized = np.array([]).astype(np.int32)  # empty array of ints
    for i in range(len(q_arr)):
        occurrences = z_arr[i + 1] - z_arr[i]
        zi = np.repeat(q_arr[i], occurrences)
        hist_quantized = np.concatenate((hist_quantized, zi))
    hist_quantized = np.concatenate((hist_quantized, np.array([q_arr[-1]])))  # add last element to last section
    return hist_quantized


def z_q_initialize(cumulative_norm, n_quant):
    """
    :param cumulative_norm: cumulative histogram of the original image
    :param n_quant: number of quants
    :return: q_arr: the values to which each of the segments’ intensities will map. q is also a one
                dimensional array, containing n_quant elements.
             z_arr: the borders which divide the histograms into segments. z is an array with shape
                (n_quant+1,). The first and last elements are 0 and 255 respectively.
    """
    buffer = int(HIGHEST_COLOR_VALUE/n_quant)
    z_arr = np.zeros(n_quant + 1).astype(np.int32)
    for i in range(1, n_quant):
        x = np.argmax(cumulative_norm > i*buffer)
        z_arr[i] = x
    z_arr[-1] = HIGHEST_COLOR_VALUE
    q_arr = np.zeros(n_quant).astype(np.int32)
    for i in range(len(q_arr)):
        q_arr[i] = (z_arr[i] + z_arr[i+1])//2
    return z_arr, q_arr

def quantize_rgb_img(im_orig, n_iter, n_quant):
    """
    quantize rgb image
    :param im_orig: 3D rgb [0-1] image
    :param n_iter: number of optimization iterations
    :param n_quant: number of desired colors
    :return  im_quant: quantized image
             error_arr: list of the errors in every iteration
    """
    img_yiq = rgb2yiq(im_orig)
    img = img_yiq[:, :, 0]  # getting the y axes
    im_quant_y, error_arr = quantize_helper(img, n_iter, n_quant)
    img_yiq[:, :, 0] = im_quant_y  # replacing the y axis with im_quant_y
    im_quant = yiq2rgb(img_yiq)  # transferring back to rgb
    return [im_quant, error_arr]


# API specified in ex1


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
            plt.imshow(image, cmap=GRAY_TXT)
            plt.show()
        elif image_dimensions(image) == RGB_DIM:
            # if image is RGB and representation is gray scale: convert to gray
            image = rgb2gray(image)  # convert to gray scale
            plt.imshow(image, cmap=GRAY_TXT)
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
    # multiplying each color vector (the 2th axis) with the conversion matrix (transposed)
    # If a is an N - D array and b is an M - D array(where M >= 2), it is a sum product over the *last* axis of a and
    # the second - to - last axis of b: (from numpy)
    yiq_image = imRGB.dot(RGB_to_YIQ.T)
    return yiq_image


def yiq2rgb(imYIQ):
    """
    converting image from YIQ representation to RGB by multiplying each color
    vector in the image'e indexes with the conversion matrix inv(RGB_to_YIQ)
    :param imYIQ: a YIQ image
    :return: a RGB image
    """
    YIQ_to_RGB = np.linalg.inv(RGB_to_YIQ)  # the inverse of the RGB_to_YIQ mat
    # multiplying each color vector (the 2th axis) with the conversion matrix (transposed)
    # If a is an N - D array and b is an M - D array(where M >= 2), it is a sum product over the *last* axis of a and
    # the second - to - last axis of b: (from numpy)
    rgb_image = imYIQ.dot(YIQ_to_RGB.T)
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
    #  get histogram and cumulative histogram, and 2D image converted to int[0,255]
    cumulative_norm, hist_orig, img = get_int_img_histograms(im_orig)
    im_eq = get_equalized_img(cumulative_norm, img)  # equalize
    hist_eq = get_histogram(im_eq)
    # if im_orig was rgb, convert back to yiq and replace the y axes with im_eq
    im_eq = normalize_img(im_eq)
    if image_dimensions(im_orig) == RGB_DIM:
        im_eq = back_to_rgb(im_eq, im_orig)
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
    if image_dimensions(im_orig) == RGB_DIM:   # if image is rgb:
        return quantize_rgb_img(im_orig, n_iter, n_quant)
    return quantize_helper(im_orig, n_iter, n_quant)




