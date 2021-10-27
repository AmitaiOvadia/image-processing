import numpy as np
import matplotlib.pyplot as plt
import sol1
import sol_ariel

x = np.hstack([np.repeat(np.arange(0, 50, 2), 10)[None, :],
                   np.array([255] * 6)[None, :]])
grad = np.tile(x, (256, 1))

def print_gray_gradient():
    x = np.hstack([np.repeat(np.arange(0, 50, 2), 10)[None, :],
                   np.array([255] * 6)[None, :]])
    grad = np.tile(x, (256, 1))
    grad = np.log(grad + 1)
    plt.imshow(grad, cmap='gray')
    plt.show()

def display_negative():
    im = sol1.imdisplay(r"image.png", 1)


def main():
    im1 = sol1.read_image(r"image.png", 1)
    im2 = sol1.read_image(r"image.png", 2)
    L1 = sol1.histogram_equalize(im1)
    L2 = sol1.histogram_equalize(im2)
    L3 = sol1.histogram_equalize(grad)
    M1 = sol1.quantize(L1[0], 4,4)
    M2 = sol1.quantize(L2[0], 4,4)
    M3 = sol1.quantize(L3[0], 4,4)
    plt.imshow(L1[0], cmap='gray')
    plt.show()
    plt.imshow(M1[0], cmap='gray')
    plt.show()

    plt.imshow(L2[0], cmap='gray')
    plt.show()
    plt.imshow(M2[0], cmap='gray')
    plt.show()

    plt.imshow(L3[0], cmap='gray')
    plt.show()
    plt.imshow(M3[0], cmap='gray')
    plt.show()

if __name__ == "__main__":
    main()