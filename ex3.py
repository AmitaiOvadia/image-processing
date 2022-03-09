import numpy as np
import scipy
from skimage.color import rgb2gray
import matplotlib.image as mpimg
from scipy.ndimage.filters import convolve
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

def image_dimensions(image):
    """
    :param image: image rgb or grayscale
    :return: int: number of dimensions
    """
    return len(image.shape)


def image_big_enough(im):
    """
    checks if image is big enough to expand pyramid
    :param im: 2d image
    :return: boolean
    """
    N, M = im.shape
    if N >= MIN_DIMENSIONS*2 and M >= MIN_DIMENSIONS*2: return True
    return False


def expand(im, filter_vec):
    """
    expanding image by adding zeros in odd places and blurring with gaussian filter normalized to 2
    :param im: 2d image
    :param filter_vec: (1, size) filter vector
    :param img_shape: image dimensions
    :return: expanded 2d image (*2)
    """
    expanded = np.zeros((2 * im.shape[0], 2 * im.shape[1]), np.float64)  # a matrix of zeros of shape 2M*2N
    expanded[::2, ::2] = im  # in all the even indexes take the original image's pixels
    expanded = convolve(expanded, filter_vec)  # blur rows with the filter_vec (normalized to 2)
    expanded = convolve(expanded, filter_vec.T)  # blur columns
    return expanded


def stretch_values_to_01(image):
    """
    stretching values from 0 to 1
    :param image: 2d image
    :return: normalized image [0,1]
    """
    min = np.min(image)
    image -= min  # now image is from zero to i_max
    i_max = np.max(image)
    if i_max != 0: image = np.divide(image, i_max)  # divide with the max value
    return image


# API

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


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    :param im: 2d image
    :param max_levels: max number of levels
    :param filter_size: size of filter for blurring (in reducing and expanding)
    :return: laplacian_pyramid: list of 2d images
             filter_vec: a filter vector for blurring
    """
    gausian_pyramid, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    laplacian_pyramid = []
    # for every image of the laplacian pyramid, take the next image (the smaller one),
    # expand it to the size of the current image, subtract it from the current image
    # and add the result to the laplacian pyramid
    for i in range(len(gausian_pyramid)-1):
        img = gausian_pyramid[i]
        smaller_image = gausian_pyramid[i + 1]
        expanded_smaller_img = expand(smaller_image, filter_vec*2)
        laplacian_image_i = img - expanded_smaller_img
        laplacian_pyramid.append(laplacian_image_i)
    # the last image in the laplacian pyramid is the last image in the gaussian
    laplacian_pyramid.append(gausian_pyramid[-1])
    return laplacian_pyramid, filter_vec


def expand_all_laplacians(lpyr, filter_vec):
    """
    expands all laplacians in lpyr list to the size of the original image
    :param lpyr: laplacian pyramid
    :return: python list of expanded laplacians
    """
    expanded_laplacians = []
    first_image = lpyr[0]
    target_shape = first_image.shape
    for i in range(len(lpyr)):
        im = lpyr[i]
        while im.shape != target_shape:  # keep expanding each until target dimensions are achieved
            im = expand(im, filter_vec)
        expanded_laplacians.append(im)
    return expanded_laplacians


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    constructs an image from a laplacian pyramid (sum all levels)
    :param lpyr: laplacian pyramid
    :param filter_vec: the filter vector
    :param coeff: python lst (len= num of levels in lpyr). used for lpyr[i]*coeff[i]
    :return: reconstructed image
    """
    n = len(lpyr)
    expanded_filter_vec = 2 * filter_vec  # now blurring vector is normalized to 2
    expanded_lpyr = expand_all_laplacians(lpyr, expanded_filter_vec)
    # multiply each of the laplacian pyramid levels with the assigned coefficient between [0,1]
    weighted_lpyr = [expanded_lpyr[i] * coeff[i] for i in range(n)]
    reconstructed_image = sum(weighted_lpyr)  # summing up all the expanded and weighted laplacian pyramid's components
    return reconstructed_image


def render_pyramid(pyr, levels):
    """
    :param pyr: a list of 2d images of different sizes (the first is the largest)
    :param levels: number of levels in the pyramid
    :return: an image that contain all pyramid's images one next to the other
    """
    rendered = stretch_values_to_01(pyr[0])
    Y = rendered.shape[ROWS_NUM]
    for i in range(1, min(levels, len(pyr))):
        next = pyr[i]
        next = stretch_values_to_01(next)  # stretched
        next_x = next.shape[COL_NUM]
        next_y = next.shape[ROWS_NUM]
        black_add = np.zeros((Y-next_y, next_x))  # prepare the added black patch (added to the bottom of the image)
        # add black pad below the next (smaller) image
        next = np.concatenate((next, black_add), axis=ADD_BELOW)
        # patch next image (with black down-padding) to the right of the rendered image
        rendered = np.concatenate((rendered, next), axis=ADD_TO_THE_RIGHT)
    return rendered



def display_pyramid(pyr, levels):
    """
    should use render_pyramid to internally render and then display
    the stacked pyramid image using plt.imshow().
    :param pyr:
    :param levels:
    :return:
    """
    image = render_pyramid(pyr, levels)
    plt.imshow(image, cmap=GRAY_TXT)
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    the algorighm:
    create L1 and L2: laplacian pyramids of im1 and im2
    create G_m: a Gaussian pyramid of mask
    create L_out: a laplacian pyramid that is a blend of L1, L2 and G_m by
    for every level k:
        L_out[k] = G_m[k]*L1[k] + (1-G_m[k])*L2[k]  (pixel-wise multiplication)
    reconstruct blended_image using L_out
    im1, im2 and mask should all have the same dimensions
    :param im1: grayscale image to be blended.
    :param im2:  grayscale image to be blended.
    :param mask: – is a boolean (i.e. dtype == np.bool) mask containing
                True (1) and False (0) representing which parts
                of im1 and im2 should appear in the resulting im_blend.
    :param max_levels: max level used for generating pyramids
    :param filter_size_im: – is the size of the Gaussian filter (an odd scalar
            that represents a squared filter) which
            defining the filter used in the construction of the Laplacian
            pyramids of im1 and im2.
    :param filter_size_mask: – is the size of the Gaussian filter(an odd scalar
            that represents a squared filter) which
            defining the filter used in the construction of the
            Gaussian pyramid of mask.
    :return: a blended image of im1 and im2 using mask
    """
    L_1, filter_vec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    L_2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)[0]
    G_m = build_gaussian_pyramid(mask, max_levels, filter_size_mask)[0]
    num_of_levels = len(G_m)
    L_out = [G_m[i] * L_1[i] + (1 - G_m[i]) * L_2[i] for i in range(num_of_levels)]
    coeff = [1 for i in range(len(L_out))]
    blended_image = laplacian_to_image(L_out, filter_vec, coeff)
    return blended_image


def blend_rgb_images(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    :param im1: rgb image
    :param im2: rgb image
    :param mask: grascale boolean image
    :param max_levels: max number of levels of the pyramid
    :param filter_size_im: filter size of images
    :param filter_size_mask: filter size for mask
    :return: bleneded rgb image
    """
    R_1, G_1, B_1 = im1[:, :, 0], im1[:, :, 1], im1[:, :, 2]  # separate different color channels of im1
    R_2, G_2, B_2 = im2[:, :, 0], im2[:, :, 1], im2[:, :, 2]  # separate different color channels of im2
    red_blend = pyramid_blending(R_1, R_2, mask, max_levels, filter_size_im, filter_size_mask)  # blend red channel
    green_blend = pyramid_blending(G_1, G_2, mask, max_levels, filter_size_im, filter_size_mask)  # blend green channel
    blue_blend = pyramid_blending(B_1, B_2, mask, max_levels, filter_size_im, filter_size_mask)  # blend blue channel
    blended_image = np.dstack((red_blend, green_blend, blue_blend))  # regroup all color channels
    blended_image = np.where(blended_image > 1, 1, blended_image)  # clip all value above 1 to 1
    return blended_image


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def plot_example(im1, im2, mask, blend_im, name):
    """
    plotting the 2 images, mask and the blended image
    :param im1: rgb image
    :param im2: rgb image
    :param mask: 2d grayscale images (1's and 0's only)
    :param blend_im: the result
    :param name: name of the blended image
    :return: None
    """
    f = plt.figure()
    f.add_subplot(2, 2, 1)
    plt.title("Image 1")
    plt.imshow(im1)

    f.add_subplot(2, 2, 2)
    plt.title("Image 2")
    plt.imshow(im2)

    f.add_subplot(2, 2, 3)
    plt.title("Mask")
    plt.imshow(mask, cmap=GRAY_TXT)

    f.add_subplot(2, 2, 4)
    plt.title(name)
    plt.imshow(blend_im)
    plt.show()


def blending_example1():
    """
    bibi and trump
    :return:
    """
    im2 = read_image(relpath('example1/trump.jpg'), 2).astype(np.float64)
    im1 = read_image(relpath('example1/bibi_and_trump.jpg'), 2).astype(np.float64)
    mask = (read_image(relpath('example1/bibi_and_trump_grayscale_mask.jpg'), 1)).astype(np.bool)
    max_levels = 6
    filter_size_im = 15
    filter_size_mask = 7
    blended_image = blend_rgb_images(im1, im2, mask, max_levels, filter_size_im, filter_size_mask)
    plot_example(im1, im2, mask, blended_image, "Trampiyahu")
    return im1, im2, mask, blended_image


def blending_example2():
    """
    queen Elizabeth and berlad
    :return:
    """
    im2 = read_image(relpath('example2/queen.jpg'), 2).astype(np.float64)
    im1 = read_image(relpath('example2/queen_and_berlad.jpg'), 2).astype(
        np.float64)
    mask = (read_image(relpath('example2/mask_queen_berlad.jpg'), 1)).astype(np.bool)
    max_levels = 6
    filter_size_im = 15
    filter_size_mask = 7
    blended_image = blend_rgb_images(im1, im2, mask, max_levels,
                                     filter_size_im, filter_size_mask)
    plot_example(im1, im2, mask, blended_image, "Queen Berlad")
    return im1, im2, mask, blended_image
