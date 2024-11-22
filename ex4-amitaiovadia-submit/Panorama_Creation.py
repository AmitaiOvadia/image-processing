# Initial code for ex4.
# You may change this code, but keep the functions' signatures
# You can also split the code to multiple files as long as this file's API is unchanged
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass
import shutil
from scipy.ndimage import map_coordinates
from imageio import imwrite
import sol4_utils

X_DERIVATIVE_KERNEL = np.asarray([[1, 0, -1]])
Y_DERIVATIVE_KERNEL = X_DERIVATIVE_KERNEL.T
INLINERS_COLOR = 'y'
OUTLINERS_COLOR = 'b'
K = 0.04


def get_x_derivative(im):
    return convolve(im, X_DERIVATIVE_KERNEL)


def get_y_derivative(im):
    return convolve(im, Y_DERIVATIVE_KERNEL)


def get_M_components(im):
    Ix = get_x_derivative(im)
    Iy = get_y_derivative(im)
    Ix2 = sol4_utils.blur_spatial(np.multiply(Ix, Ix), 3)
    Iy2 = sol4_utils.blur_spatial(np.multiply(Iy, Iy), 3)
    IxIy = sol4_utils.blur_spatial(np.multiply(Ix, Iy), 3)
    return Ix2, IxIy, Iy2


# API

def harris_corner_detector(im):
    """
    Detects harris corners.
    use non_maximum_suppression to find the (returns binary image of the strongest corners)
    corners = np.array(2,N)
    traverse the binary image, for every pixel add (x,y) to corners (need to flip the values though)
    :param im: A 2D array representing an image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    Ix2, IxIy, Iy2 = get_M_components(im)
    M_determinate = np.multiply(Ix2, Iy2) - np.multiply(IxIy, IxIy)  # the determinant of the M matrix
    M_trace = Ix2 + Iy2  # the trace of M
    R_response_map = M_determinate - K * (M_trace * M_trace)  # R is the response image
    binary_maximized_R = non_maximum_suppression(R_response_map)
    all_corners = np.argwhere(binary_maximized_R)  # get all non 0 indexes (now the it's in (y,x) form)
    all_corners = np.flip(all_corners, axis=1)  # flip (x,y) to (y,x)
    return all_corners


def sample_descriptor(im, pos, desc_rad):
    """
    Samples descriptors at the given corners.
    :param im: A 2D array representing an image,
            in our case the 3rd level pyramid image.
    :param pos: An array with shape (N,2), where pos[i,:] are the [x,y]
            coordinates of the ith corner point. could be a fraction
    :param desc_rad: "Radius" of descriptors to compute. means the number of
            pixels left and right of the main one 7 on 7 is radius 3
    :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
    """
    k = 1 + 2 * desc_rad
    N = pos.shape[0]  # check
    descriptors = np.zeros(shape=(N, k, k))
    for i in range(N):
        x, y = pos[i]  # x,y of point in the pyramid[2]
        grid = np.array([(y + j, x + k)  # an array of the corresponding indexes in image
                         for j in range(-desc_rad, desc_rad + 1)
                         for k in range(-desc_rad, desc_rad + 1)]).T
        # a 2d sample of the pixels in the radius around (x,y), mapped from image
        sample = (map_coordinates(im, (grid), order=1, prefilter=False)).reshape(k, k)
        descriptor = sample - np.mean(sample)  # d - m (m is mean of the matrix)
        norm_factor = np.linalg.norm(descriptor)  # |d-m|
        if norm_factor != 0: descriptor /= norm_factor  # (d-m)/|d-m| if |d-m| is not 0
        descriptors[i::] = descriptor
    return descriptors


def find_features(pyr):
    """
    Detects and extracts feature points from a pyramid.
    calls spread_out_corners on pyr[0]
    for each corner found in pyr[0]: get their pyr[2] index and extract descriptor.
    :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
    :return: A list containing:
                1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                   These coordinates are provided at the pyramid level pyr[0].
                2) A feature descriptor array with shape (N,K,K)
    """
    im = pyr[0]  # the original image is level 0 of the pyramid
    corners_in_pyr0 = spread_out_corners(im, 4, 4, 5)  # find corners in image
    # get the corners corresponding coordinates in the level 2 of the pyramid:
    corners_in_pyr3 = corners_in_pyr0.astype(np.float64) / 4.0  # 2**(0-2) level 0 to level 2
    descriptors = sample_descriptor(pyr[2], corners_in_pyr3, 3)  # get descriptors for all the corners
    return [corners_in_pyr0, descriptors]


def match_features(desc1, desc2, min_score):
    """
    Return indices of matching descriptors.
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param desc2: A feature descriptor array with shape (N2,K,K).
    :param min_score: Minimal match score.
    :return: A list containing:
                1) An array with shape (M,) and dtype int of matching indices in desc1.
                2) An array with shape (M,) and dtype int of matching indices in desc2.
    """
    N = desc1.shape[0]  # N the number of descriptors in desc1,
    k = desc1.shape[1]  # k*k the size of a single descriptor
    M = desc2.shape[0]  # M the number of descriptors in desc2
    # reshape descriptors to a vector of N/M cells of k*k features
    D1 = desc1.reshape((N, k * k))
    D2 = desc2.reshape((M, k * k))
    # all scores is a matrix that in M[i,j] = dot product of D1[i] feature with D2[j] feature
    # in short: the score of the match of the descriptors D1[i] with D2[j]
    score = np.einsum('ij, kj->ik', D1, D2)  # multiply every cell of D1 with every cell of D2 ((49,) (,49))
    rows_2nd_max = np.sort(score, axis=1)[:, -2].reshape(N, 1)  # take 2nd max of rows
    cols_2nd_max = np.sort(score.T, axis=1)[:, -2].reshape(1, M)  # take 2nd max of columns
    # take score[i][j] only of larger then 2nd best in row, and 2nd best in column and the min_score
    matched_indexes = (score >= cols_2nd_max) & \
                      (score >= rows_2nd_max) & \
                      (score >= min_score)
    y_x_pairs = np.argwhere(matched_indexes)  # take only indexes that apply to the terms ()
    x_y_pairs = np.flip(y_x_pairs, axis=1)  # flip to (x,y) representation
    desc1_matches = x_y_pairs[:, 1]  # only the desc1 matched indexes
    desc2_matches = x_y_pairs[:, 0]  # only the desc2 matched indexes
    return [desc1_matches, desc2_matches]




def apply_homography(pos1, H12):
    """
    Apply homography to inhomogenous points.
    add an extra coordinate in pos1: [x,y] -> [x,y,1]
    pos1_new = H12 * (every vector in pos1)
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates
             obtained from transforming pos1 using H12.
    """
    N = pos1.shape[0]  # number of points
    hom_coordinate = np.ones((N, 1))  # the added homogeneous coordinate vector to add
    pos1_hom = np.concatenate((pos1, hom_coordinate), axis=1)  # add hom coordinate
    # apply homography to each point by multiplying from the left H12*hom_point
    # pos_new_hom = np.einsum('ij, kj->ki', H12, pos1_hom)
    pos_new_hom = (H12 @ pos1_hom.T).T
    # pos_new_hom = np.apply_along_axis(lambda point: H12.dot(point), 1, pos1_hom)
    x_axis, y_axis, z_axis = pos_new_hom[:, 0], pos_new_hom[:, 1], pos_new_hom[:, 2]
    returned_points = np.vstack((x_axis, y_axis))  # join x and y : [[x1, x2 .. xn], [y1, y2 .. yn]]
    returned_points = np.divide(returned_points, z_axis)  # divide by z axis element-wise
    return returned_points.T




def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
    """
    Computes homography between two sets of points using RANSAC.
    :param pos1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
    :param pos2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
    :param num_iter: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :param translation_only: see estimate rigid transform
    :return: A list containing:
                1) A 3x3 normalized homography matrix.
                2) An Array with shape (S,) where S is the number of inliers,
                    containing the indices in pos1/pos2 of the maximal set of inlier matches found.
    """
    max_inliers_idx = np.zeros((0, 0))  # the returned array of inliers indexes
    N = points1.shape[0]  # number of points matched
    max_inliers_count = 0  # highest number of inliers yet
    for i in range(num_iter):
        idx1, idx2 = np.random.choice(N, size=2)  # choose randomly 2 indexes matched points
        pair_of_points1 = np.array([points1[idx1], points1[idx2]])  # [[x1,y1] in image 1, [x2,y2] in image 2]]
        pair_of_points2 = np.array([points2[idx1], points2[idx2]])  # [[x1,y1] in image 1, [x2,y2] in image 2]]
        H12 = estimate_rigid_transform(pair_of_points1, pair_of_points2, translation_only)  # get H12
        p1_transformed = apply_homography(points1, H12)  # apply H12 to all points found in image 1
        # compute square error of H12: (how far)^2 from points2 had all points1 landed after applying H12
        E = np.apply_along_axis(lambda p: p[0] * p[0] + p[1] * p[1], 1, p1_transformed - points2)
        E = np.where(E < inlier_tol, 0, 1)  # all indexes that landed closer then inlier_tol value marked as 0:
        inliers_count = np.count_nonzero(E == 0)  # count all the inliers
        if inliers_count > max_inliers_count:  # if this iteration was better the the best iteration so far:
            indexes = np.argwhere(E == 0)  # get inliers' indexes
            max_inliers_count = inliers_count  # update max inliers count
            max_inliers_idx = indexes.reshape(inliers_count,)
    # get all the points from points1 and points2 that are certainly a part of a good match
    matched_points_1 = points1[max_inliers_idx, :]
    matched_points_2 = points2[max_inliers_idx, :]
    # estimate for the last time the transform H12 using all the good matches and least squares for higher accuracy
    chosen_H12 = estimate_rigid_transform(matched_points_1, matched_points_2, translation_only)
    return [chosen_H12, max_inliers_idx]


def plot_transformation(color, in_out_liners, pos1, pos2, width_im1):
    """
    :param color: color of line
    :param in_out_liners:  ndarray(N,) of indexses in pos1, pos2 of points draw
    :param pos1: ndarray(N,2) of points in image
    :param pos2: ndarray(N,2) of points in image
    :param width_im1: the width of image1, need to shift the x values of pos2
    :return: None. plot the shift without showing
    """
    # get x values
    x_vals_im1 = pos1[:, 0][in_out_liners]
    x_vals_im2 = width_im1 + pos2[:, 0][in_out_liners]  # values are shifted
    # get y vals
    y_vals_im1 = pos1[:, 1][in_out_liners]
    y_vals_im2 = pos2[:, 1][in_out_liners]
    x1_x2_vals = [x_vals_im1, x_vals_im2]
    y1_y2_vals = [y_vals_im1, y_vals_im2]
    plt.plot(x1_x2_vals, y1_y2_vals, mfc='r', c=color, lw=.5, ms=3, marker='o')


def display_matches(im1, im2, points1, points2, inliers):
    """
    Display matching points between 2 images
    :param im1: A grayscale image.
    :param im2: A grayscale image.
    :parma pos1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
    :param pos2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    To use the function plt.plot to draw a single thin blue line between two wide red points (x[0],y[0])
    and (x[1],y[1]), you can execute plt.plot(x, y, mfc=’r’, c=’b’, lw=.4, ms=10, marker=’o’).
    """
    combined_im = np.hstack((im1, im2))  # combine 2 images
    plt.imshow(combined_im, 'gray')
    width_im1 = im1.shape[1]  # get x shift of im2 in the display
    all_indexes = np.arange(0, points1.shape[0])  # an array of all the points indexes
    outliers = np.setdiff1d(all_indexes, inliers)  # all indexes that are not inliers
    plot_transformation(OUTLINERS_COLOR, outliers, points1, points2, width_im1)  # plot all outliers shiftd
    plot_transformation(INLINERS_COLOR, inliers, points1, points2, width_im1)  # plot all inliers shiftd
    plt.show()


def accumulate_homographies(H_succesive, m):
    """
    Convert a list of succesive homographies to a
    list of homographies to a common reference frame.
    :param H_successive: A list of M-1 3x3 homography
      matrices where H_successive[i] is a homography which transforms points
      from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system towards which we would like to
      accumulate the given homographies.
    :return: A list of M 3x3 homography matrices,
      where H2m[i] transforms points from coordinate system i to coordinate system m
    """
    N = len(H_succesive)
    H2m = np.empty((N + 1, 3, 3))
    H2m[m] = np.eye(3)
    for i in range(m-1, -1, -1):
        H2m[i] = np.dot(H2m[i + 1], H_succesive[i])
        norm_factor = H2m[i][2, 2]
        H2m[i] /= norm_factor
    for i in range(m + 1, N + 1):
        inv_of_previous = np.linalg.inv(H_succesive[i - 1])
        H2m[i] = np.dot(H2m[i - 1], inv_of_previous)
        norm_factor = H2m[i][2, 2]
        H2m[i] /= norm_factor
    return list(H2m)



def compute_bounding_box(homography, w, h):
    """
    computes bounding box of warped image under homography, without actually warping the image
    :param homography: homography
    :param w: width of the image
    :param h: height of the image
    :return: 2x2 array, where the first row is [x,y] of the top left corner,
     and the second row is the [x,y] of the bottom right corner
    """
    bottom_left = [0, h - 1]  # bottom left corner - h-1 is the last row index
    bottom_right = [w - 1, h - 1]  # bottom right corner
    top_left = [0, 0]  # top left corner
    top_right = [w - 1, 0]  # top right corner
    all_corners = np.array([top_left, top_right, bottom_left, bottom_right])
    new_corners_cords = apply_homography(all_corners, homography)  # apply homography to all the corners
    # get the top right corner of the box by taking the min_x and min_y of the corners after the homography
    new_x_of_corners = new_corners_cords[:, 0]  # all the new x'es
    new_y_of_corners = new_corners_cords[:, 1]  # all the new y'es
    min_x = np.min(new_x_of_corners)
    min_y = np.min(new_y_of_corners)
    max_x = np.max(new_x_of_corners)
    max_y = np.max(new_y_of_corners)
    # the box is defined by the most extreme coordinates of each axes
    return np.array([[min_x, min_y], [max_x, max_y]]).astype(np.int)  # top right and bottom left of the box


def get_warped_coordinate(homography, image):
    h, w = image.shape
    box = compute_bounding_box(homography, w, h)
    x_0, y_0 = box[0]
    x_m, y_m = box[1]
    x_range = np.arange(x_0, x_m + 1)
    y_range = np.arange(y_0, y_m + 1)
    x_i, y_i = np.meshgrid(x_range, y_range)
    # all indexes of the warped image in the coordinate system of the old omage
    coords = np.stack((x_i, y_i), axis=-1)  # (x_dim,y_dim,2)
    box_width = coords.shape[0]
    box_hight = coords.shape[1]
    coords = coords.reshape(box_width * box_hight, 2)  # (N,2)
    return coords, box_hight, box_width


def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    convert image to homogeneous coordinates
    create a new image at
    :param image: a 2D image.
    :param homography: homograhpy.
    :return: A 2d warped image.
    """
    coords, box_hight, box_width = get_warped_coordinate(homography, image)  # get all the coordinate of the warped box
    inv_hom = np.linalg.inv(homography)  # inverse homography
    # ask every index in warped image where he came from
    # (in case of negative indexes and out of range indexes the image will be black)
    warp_indexes = (apply_homography(coords, inv_hom)).T  # (2,N) ordered as [[y],[x]]
    # each coordinate in the warped image has now a corresponding coordinate in image
    warp_indexes[[0, 1]] = warp_indexes[[1, 0]]  # (2,N) ordered as [[x],[y]]
    # mape coordinates of image to their corresponding warp indexses
    warp_im = (map_coordinates(image, warp_indexes, order=1, prefilter=False))
    return warp_im.reshape((box_width, box_hight))  # back to original image shape





def warp_image(image, homography):
    """
    Warps an RGB image with a given homography.
    :param image: an RGB image.
    :param homography: homograhpy.
    :return: A warped image.
    """
    return np.dstack([warp_channel(image[..., channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
    """
    Filters rigid transformations encoded as homographies by the amount of translation from left to right.
    :param homographies: homograhpies to filter.
    :param minimum_right_translation: amount of translation below which the transformation is discarded.
    :return: filtered homographies..
    """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if homographies[i][0, -1] - last > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
    """
    Computes rigid transforming points1 towards points2, using least squares method.
    points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
    :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
    :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
    :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
    :return: A 3x3 array with the computed homography.
    """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2, :2] = rotation
    H[:2, 2] = translation
    return H


def non_maximum_suppression(image):
    """
    Finds local maximas of an image.
    :param image: A 2D array representing an image.
    :return: A boolean array with the same shape as the input image, where True indicates local maximum.
    """
    # Find local maximas.
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max[image < (image.max() * 0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num) + 1)
    centers = np.stack(centers).round().astype(np.int)
    ret = np.zeros_like(image, dtype=np.bool)
    ret[centers[:, 0], centers[:, 1]] = True

    return ret


def spread_out_corners(im, m, n, radius):
    """
    Splits the image im to m by n rectangles and uses harris_corner_detector on each.
    :param im: A 2D array representing an image.
    :param m: Vertical number of rectangles.
    :param n: Horizontal number of rectangles.
    :param radius: Minimal distance of corner points from the boundary of the image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    corners = [np.empty((0, 2), dtype=np.int)]
    x_bound = np.linspace(0, im.shape[1], n + 1, dtype=np.int)
    y_bound = np.linspace(0, im.shape[0], m + 1, dtype=np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j + 1], x_bound[i]:x_bound[i + 1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:, 0] > radius) & (corners[:, 0] < im.shape[1] - radius) &
             (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0] - radius))
    ret = corners[legit, :]
    return ret


class PanoramicVideoGenerator:
    """
    Generates panorama from a set of images.
    """

    def __init__(self, data_dir, file_prefix, num_images, bonus=False):
        """
        The naming convention for a sequence of images is file_prefixN.jpg,
        where N is a running number 001, 002, 003...
        :param data_dir: path to input images.
        :param file_prefix: see above.
        :param num_images: number of images to produce the panoramas with.
        """
        self.bonus = bonus
        self.file_prefix = file_prefix
        self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
        self.files = list(filter(os.path.exists, self.files))
        self.panoramas = None
        self.homographies = None
        print('found %d images' % len(self.files))

    def align_images(self, translation_only=False):
        """
        compute homographies between all images to a common coordinate system
        :param translation_only: see estimte_rigid_transform
        """
        # Extract feature point locations and descriptors.
        points_and_descriptors = []
        for file in self.files:
            image = sol4_utils.read_image(file, 1)
            self.h, self.w = image.shape
            pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
            points_and_descriptors.append(find_features(pyramid))

        # Compute homographies between successive pairs of images.
        Hs = []
        for i in range(len(points_and_descriptors) - 1):
            points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
            desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

            # Find matching feature points.
            ind1, ind2 = match_features(desc1, desc2, .7)
            points1, points2 = points1[ind1, :], points2[ind2, :]

            # Compute homography using RANSAC.
            H12, inliers = ransac_homography(points1, points2, 100, 6, translation_only)

            # Uncomment for debugging: display inliers and outliers among matching points.
            # In the submitted code this function should be commented out!
            # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

            Hs.append(H12)

        # Compute composite homographies from the central coordinate system.
        accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
        self.homographies = np.stack(accumulated_homographies)
        self.frames_for_panoramas = filter_homographies_with_translation(self.homographies, minimum_right_translation=5)
        self.homographies = self.homographies[self.frames_for_panoramas]

    def generate_panoramic_images(self, number_of_panoramas):
        """
        combine slices from input images to panoramas.
        :param number_of_panoramas: how many different slices to take from each input image
        """
        if self.bonus:
            self.generate_panoramic_images_bonus(number_of_panoramas)
        else:
            self.generate_panoramic_images_normal(number_of_panoramas)

    def generate_panoramic_images_normal(self, number_of_panoramas):
        """
        combine slices from input images to panoramas.
        :param number_of_panoramas: how many different slices to take from each input image
        """
        assert self.homographies is not None

        # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
        self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
        for i in range(self.frames_for_panoramas.size):
            self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

        # change our reference coordinate system to the panoramas
        # all panoramas share the same coordinate system
        global_offset = np.min(self.bounding_boxes, axis=(0, 1))
        self.bounding_boxes -= global_offset

        slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int)[1:-1]
        warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
        # every slice is a different panorama, it indicates the slices of the input images from which the panorama
        # will be concatenated
        for i in range(slice_centers.size):
            slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
            # homography warps the slice center to the coordinate system of the middle image
            warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
            # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
            warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

        panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

        # boundary between input images in the panorama
        x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
        x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                      x_strip_boundary,
                                      np.ones((number_of_panoramas, 1)) * panorama_size[0]])
        x_strip_boundary = x_strip_boundary.round().astype(np.int)

        self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
        for i, frame_index in enumerate(self.frames_for_panoramas):
            # warp every input image once, and populate all panoramas
            image = sol4_utils.read_image(self.files[frame_index], 2)
            warped_image = warp_image(image, self.homographies[i])
            x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
            y_bottom = y_offset + warped_image.shape[0]

            for panorama_index in range(number_of_panoramas):
                # take strip of warped image and paste to current panorama
                boundaries = x_strip_boundary[panorama_index, i:i + 2]
                image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
                x_end = boundaries[0] + image_strip.shape[1]
                self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

        # crop out areas not recorded from enough angles
        # assert will fail if there is overlap in field of view between the left most image and the right most image
        crop_left = int(self.bounding_boxes[0][1, 0])
        crop_right = int(self.bounding_boxes[-1][0, 0])
        assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
        print(crop_left, crop_right)
        self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

    def generate_panoramic_images_bonus(self, number_of_panoramas):
        """
        The bonus
        :param number_of_panoramas: how many different slices to take from each input image
        """
        pass

    def save_panoramas_to_video(self):
        assert self.panoramas is not None
        out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
        try:
            shutil.rmtree(out_folder)
        except:
            print('could not remove folder')
            pass
        os.makedirs(out_folder)
        # save individual panorama images to 'tmp_folder_for_panoramic_frames'
        for i, panorama in enumerate(self.panoramas):
            imwrite('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
        if os.path.exists('%s.mp4' % self.file_prefix):
            os.remove('%s.mp4' % self.file_prefix)
        # write output video to current folder
        os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
                  (out_folder, self.file_prefix))

    def show_panorama(self, panorama_index, figsize=(20, 20)):
        assert self.panoramas is not None
        plt.figure(figsize=figsize)
        plt.imshow(self.panoramas[panorama_index].clip(0, 1))
        plt.show()
