import numpy as np
from skimage.color import rgb2gray
from skimage.color import rgb2yiq
from skimage.color import yiq2rgb
from imageio import imread, imwrite
import matplotlib as matpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sol
import math

x = np.hstack([np.repeat(np.arange(0, 50, 2), 10)[None, :],
               np.array([255] * 6)[None, :]])
grad = np.tile(x, (256, 1))

def print_gray_gradient():
    x = np.hstack([np.repeat(np.arange(0, 50, 2), 10)[None, :],
                   np.array([255] * 6)[None, :]])
    grad = np.tile(x, (256, 1))
    grad = np.log(grad + 1)
    plt.imshow(grad)
    plt.show()

def display_negative():
    im = sol.imdisplay(r"image.png", 1)


def main():
    # sol.imdisplay(r"image3.png", 2)
    # print_gray_gradient()
    im = sol.read_image(r"image.png", 1)
    # # im = grad
    sol.histogram_equalize(grad)
    # print_gray_gradient()

if __name__ == "__main__":
    main()