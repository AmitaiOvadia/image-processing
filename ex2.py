import numpy as np
from skimage.color import rgb2gray
import matplotlib.image as mpimg
from scipy.io.wavfile import read
from scipy.io.wavfile import write
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates
# todo erase
# sign for the exponent argument
IDFT_SIGN = 1
DFT_SIGN = -1

RGB_DIM = 3  # shape of an rgb image has 3 elemnts (rows,cols,3)
MAX_GREYSCALE = 255
TO_GREYSCALE = 1
TO_RGB = 2


FOURIER_TRANSFORM = -1
INVERSE_FOURIER_TRANSFORM = 1
GRAY_SCALE = 1
RGB = 2
RGB_DIM = 3
GRAY_SCALE_DIM = 2
HIGHEST_COLOR_VALUE = 255
GRAY_TXT = 'gray'
CHANGE_RATE_FILE_NAME = "change_rate.wav"
CHANGE_SAMPLES_FILE_NAME = "change_samples.wav"

# supplied

def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    num_timesteps = int(spec.shape[1] / ratio)
    time_steps = np.arange(num_timesteps) * ratio

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec


# private

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


# 1.1.1
def DFT(signal):
    """
    Discrete Fourier Transform
    :param signal: array of dtype float64 with shape (N,) or (N,1)
    :return: complex Fourier signal.
    """
    x = np.arange(len(signal))
    u = x.reshape((len(signal), 1))
    e = np.exp((-2j * np.pi * x * u) / len(signal))
    dft = signal.T @ e
    return dft.reshape(signal.shape)


# 1.1.2
def IDFT(fourier_signal):
    """
    Inverse Discrete Fourier Transform
    :param fourier_signal: array of dtype complex128 with shape (N,) or (N,1)
    :return: complex signal.
    """
    x = np.arange(len(fourier_signal))
    u = x.reshape((len(fourier_signal), 1))
    e = np.exp((2j * np.pi * x * u) / len(fourier_signal))
    idft = fourier_signal.T @ e
    return idft.reshape(fourier_signal.shape) / len(fourier_signal)


def DFT2(image):
    """
    :param image: greyscale image of dtype float64, shape (M,N,1) or (M, N)
    :return: a fourier image of dtype complex128 and the same dimensions
    """
    shape = image.shape
    rows, columns = shape[0], shape[1]  # rows, columns
    dft2D = np.ndarray(image.shape, dtype=np.complex128)
    for y in range(rows):  # dft rows
        dft2D[y, :] = DFT(image[y, :])
    for x in range(columns):  # dft columns
        dft2D[:, x] = DFT(dft2D[:, x])
    return dft2D


def IDFT2(fourier_image):
    """
    :param fourier_image: 2D array of type complex128, shape (M,N,1)
    :return: original image:
    """
    shape = fourier_image.shape
    rows, columns = shape[0], shape[1]  # rows, columns
    idft2D = np.ndarray(fourier_image.shape, dtype=np.complex128)
    for y in range(rows):  # idft rows
        idft2D[y, :] = IDFT(fourier_image[y, :])
    for x in range(columns):  # idft columns
        idft2D[:, x] = IDFT(idft2D[:, x])
    return idft2D


def change_rate(filename, ratio):
    """
    changing the rate of sampling only by changing the file header
    :param filename: string, path of file to change
    :param ratio: a positive float64 representing the duration change 0.25 < ratio < 4.
    :return: None
    """
    sampling_rate, data = read(filename)
    write(CHANGE_RATE_FILE_NAME, int(sampling_rate*ratio), data)


def change_samples(filename, ratio):
    """
    function gets audio file and resizing using resize(data, ratio)
    :param filename: string, path of file to change
    :param ratio: a positive float64 representing the duration change 0.25 < ratio < 4.
    :return:
    """
    sampling_rate, data = read(filename)
    data = np.real(resize(data.astype(np.float64), ratio))
    write(CHANGE_SAMPLES_FILE_NAME, sampling_rate, data)
    return data


def resize(data, ratio):
    """
    resizing the data:
        doing ft in 1D to the data
        fft shift : shifting the 0 frequency to the middle
        if r > 1:
            clipping the high frequencies
        if r > 1:
            add r/2 amount of zeros at the high and low frequencies
            (if r is odd add 1 more to one of the sides)
        transform back the data using ift
    :param data: 1D ndarray of dtype float64 or complex128(*)
                  representing the original sample points
    :param ratio: 0.25 < ratio < 4
    :return: D ndarray of the dtype of data representing the new sample points
    """
    if ratio == 1 or len(data) == 0:  # do nothing
        return data
    fourier_signal = DFT(data)
    resized = np.fft.fftshift(fourier_signal)  #
    N = len(fourier_signal)
    new_N = int(N / ratio)
    size_diff = np.abs(int(N - new_N))
    right_diff = int(size_diff / 2)
    left_diff = size_diff - right_diff
    if ratio < 1:
        resized = np.pad(resized, (left_diff, right_diff), 'constant')
    elif ratio > 1:
        resized = resized[left_diff:new_N + left_diff]
    if len(resized) == 0:
        return resized.astype(data.dtype.type)
    if data.dtype.type == np.complex128:
        return IDFT(np.fft.ifftshift(resized))
    else:
        return np.real(IDFT(np.fft.ifftshift(resized))).astype(data.dtype.type)


def resize_spectrogram(data, ratio):
    """
    speeds up a WAV file, without changing the pitch, using spectrogram scaling. This
    is done by computing the spectrogram, changing the number of spectrogram columns, and creating back
    the audio.
    :param data: is a 1D ndarray of dtype float64 representing the original sample points
    :param ratio: is a positive float64 representing the rate change of the WAV file
    :return: the new sample points according to ratio with the same datatype as data.
    """
    data_type = data.dtype.type
    stft_spectogram = stft(data)
    rows_count, cols_count = stft_spectogram.shape
    resized_spectoram = np.ndarray(shape=(rows_count, int(cols_count / ratio)),
                              dtype=np.complex128)
    for i in range(len(stft_spectogram)):
        line = stft_spectogram[i]
        resized_spectoram[i] = resize(line, ratio)
    resized_data = istft(resized_spectoram)
    return resized_data.astype(data_type)


def resize_vocoder(data, ratio):
    """
    function that speedups a WAV file by phase vocoding its spectrogram.
    :param data: 1D ndarray of dtype float64 representing the original sample points
    :param ratio: is a positive float64 representing the rate change of the WAV file
    :return:
    """
    phase_vocoded_data = istft(phase_vocoder(stft(data), ratio))
    return phase_vocoded_data.astype(data.dtype.type)


def conv_der(im):
    """
    derivative approximation using convolution
    :param im: 2D grayscale image
    :return: 2D grayscale image of the derivatives magnitude in abs
    """
    dx_conv = np.array([[-0.5, 0, 0.5]])
    dy_conv = dx_conv.transpose()
    dx = signal.convolve2d(im, dx_conv, mode='same').astype(np.float64)
    dy = signal.convolve2d(im, dy_conv, mode='same').astype(np.float64)
    magnitude = np.sqrt(np.abs(dx)**2 + np.abs(dy)**2)
    return magnitude


def fourier_der(im):
    """
    fdx =*pi*i/N)*ft({for evey u (u*ft(u))
    :param im: 2D float64 grayscale
    :return: magnitude: 2D float64 grayscale the magnitude in abs of the derivative
    """
    # upply 2D dft and shift 0 to center
    fourier_signal = np.fft.fftshift(DFT2(im))
    M, N = fourier_signal.shape  # M = num of columns (y's), N = columns (x's)
    derivative_coefficient_x = 2 * np.pi * 1j / N
    derivative_coefficient_y = 2 * np.pi * 1j / M
    # create 2 matrixes N*N (rows increasing) and M*M (columns increasing),
    # from -k/2 to k/2
    dy_coef, dx_coef = np.meshgrid(np.arange(int(-N / 2), np.ceil(N / 2)),
                                   np.arange(int(-M / 2), np.ceil(M / 2)))
    # for each u/v multiply by F(u/v) and shift
    u_mul_Fu = np.fft.ifftshift(dx_coef * fourier_signal)
    v_mul_Fv = np.fft.ifftshift(dy_coef * fourier_signal)
    dx = derivative_coefficient_x * IDFT2(u_mul_Fu)
    dy = derivative_coefficient_y * IDFT2(v_mul_Fv)
    magnitude = np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)
    return magnitude
