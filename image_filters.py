import os
import cv2
from PIL import Image
import math
import numpy as np


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to grayscale.

    Args:
        image: a numpy array

    Returns:
        a numpy array
    """

    gray_image = np.zeros((image.shape[0], image.shape[1]))

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # https://en.wikipedia.org/wiki/Grayscale#Colorimetric_(perceptual_luminance-preserving)_conversion_to_grayscale
            gray_image[i][j] = 0.299 * image[i][j][0] + 0.587 * image[i][j][1] + 0.114 * image[i][j][2]

    return gray_image


def conv(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve an image with a kernel.

    Args:
        image: a numpy array
        kernel: a numpy array

    Returns:
        a numpy array
    """

    if len(kernel.shape) == 3:
        conv_image_shape = (image.shape[0] - kernel.shape[0] + 1, image.shape[1] - kernel.shape[1] + 1, image.shape[2])
    elif len(kernel.shape) == 2:
        conv_image_shape = (image.shape[0] - kernel.shape[0] + 1, image.shape[1] - kernel.shape[1] + 1)
    else:
        raise ValueError('Invalid kernel shape')

    conv_image = np.zeros(conv_image_shape)
    for i in range(conv_image_shape[0]):
        for j in range(conv_image_shape[1]):
            conv_image[i][j] = conv_pixel(image, kernel, i, j)

    return conv_image


def conv_pixel(image: np.ndarray, kernel: np.ndarray, x: int, y: int) -> np.ndarray:
    """
    Convolve an image with a kernel at a specific pixel.

    Args:
        image: a numpy array
        kernel: a numpy array
        x: the x coordinate of the pixel
        y: the y coordinate of the pixel

    Returns:
        a numpy array
    """

    if len(kernel.shape) == 3:
        pixel = np.zeros(image.shape[2])
    elif len(kernel.shape) == 2:
        pixel = 0
    else:
        raise ValueError('Invalid kernel shape')

    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            pixel += image[x + i][y + j] * kernel[i][j]

    return pixel


def get_box_filter(shape: tuple) -> np.ndarray:
    """
    Return a box filter.

    Args:
        shape: the shape of the filter

    Returns:
        a numpy array
    """

    filter = np.zeros(shape)

    for c in range(shape[2]):
        for i in range(shape[0]):
            for j in range(shape[1]):
                filter[i][j][c] = 1 / (shape[0] * shape[1])

    return filter


def get_gaussian_filter(shape: tuple, sigma: float) -> np.ndarray:
    """
    Return a Gaussian filter.

    Args:
        shape: the shape of the filter
        sigma: the standard deviation of the Gaussian distribution

    Returns:
        a numpy array
    """

    filter = np.zeros(shape)

    for c in range(shape[2]):
        for i in range(shape[0]):
            for j in range(shape[1]):
                filter[i][j][c] = gaussian(i, j, sigma)

    normalize(filter, shape[2])
    return filter


def gaussian(x: int, y: int, sigma: float) -> float:
    """
    Return the value of a Gaussian distribution at (x, y).

    Args:
        x: the x coordinate
        y: the y coordinate
        sigma: the standard deviation of the Gaussian distribution

    Returns:
        a float
    """

    return 1 / (2 * math.pi * sigma ** 2) * math.e ** (-(x ** 2 + y ** 2) / (2 * sigma ** 2))


def get_sobel_edge_filter(size: int, orientation: int) -> np.ndarray:
    """
    Return an edge filter.

    Args:
        shape: the shape of the filter

    Returns:
        a numpy array
    """

    if size == 3:
        if orientation == 0:
            return np.array([
                [1, 2, 1],
                [0, 0, 0],
                [-1, -2, 1]
            ])
        elif orientation == 1:
            return np.array([
                [1, 0, -1],
                [2, 0, -2],
                [1, 0, -1]
            ])
        else:
            raise ValueError('Invalid orientation')
    elif size == 5:
        if orientation == 0:
            return np.array([
                [2, 2, 4, 2, 2],
                [1, 1, 2, 1, 1],
                [0, 0, 0, 0, 0],
                [-1, -1, -2, -1, -1],
                [-2, -2, -4, -2, -2]
            ])
        elif orientation == 1:
            return np.array([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ])
        else:
            raise ValueError('Invalid orientation')


def get_rectangle_filter() -> np.ndarray:
    """
    Return a rectangle filter.

    Returns:
        a numpy array
    """

    rectangle_filter = np.array([
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0]
    ])

    filter = np.zeros((5, 5, 3))

    for c in range(filter.shape[2]):
        filter[:, :, c] = rectangle_filter
    
    normalize(filter, filter.shape[2])
    return filter


def normalize(filter: np.ndarray, channels: int):
    if channels == 1:
        filter /= np.sum(filter)
    else:
        for c in range(channels):
            filter[:, :, c] /= np.sum(filter[:, :, c])


def filter_on_color(image: Image, lower: list, upper: list) -> np.ndarray:
    hsv_image = image.convert('HSV')

    lower_h = lower[0]
    upper_h = upper[0]

    np_lower = np.array(lower, dtype=np.uint8)
    np_upper = np.array(upper, dtype=np.uint8)

    if lower_h < upper_h:
        mask = np.all(np.logical_and(hsv_image >= np_lower, hsv_image <= np_upper), axis=-1)
    else:
        mask = np.all(np.logical_or(hsv_image >= np_lower, hsv_image <= np_upper), axis=-1)

    curated_image = np.array(image)
    curated_image[~mask] = [0, 0, 0]

    return curated_image


def main():
    resources_path = './resources'
    results_path = './results'

    # Box & Gaussian & Edge filters
    for filename in os.listdir(resources_path):
        image = cv2.imread(os.path.join(resources_path, filename))
        filter_size = 5

        # Box filter
        box_filter = get_box_filter((filter_size, filter_size, image.shape[2]))
        box_result = conv(image, box_filter)
        cv2.imwrite(os.path.join(results_path, f'box_{filter_size}/{filename}'), box_result)

        # Gaussian filter
        gaussian_filter = get_gaussian_filter((filter_size, filter_size, image.shape[2]), 1)
        gaussian_result = conv(image, gaussian_filter)
        cv2.imwrite(os.path.join(results_path, f'gaussian_{filter_size}/{filename}'), gaussian_result)

        # Edge filter
        gray_image = convert_to_grayscale(image)
        edge_x_filter = get_sobel_edge_filter(filter_size, 0)
        edge_y_filter = get_sobel_edge_filter(filter_size, 1)
        edge_result = conv(conv(gray_image, edge_x_filter), edge_y_filter)
        cv2.imwrite(os.path.join(results_path, f'edge_{filter_size}/{filename}'), edge_result)


    # Object detection
    for filename in os.listdir(resources_path):
        image = Image.open(f'{resources_path}/{filename}')

        # Blue pool
        blue_pool = filter_on_color(image, [125, 50, 50], [130, 255, 255])
        blue_pool_image = Image.fromarray(blue_pool)
        blue_pool_image.save(f'{results_path}/blue_pool/{filename}')

        # Orange building
        orange_building = filter_on_color(image, [0, 50, 50], [15, 255, 255])
        orange_building_image = Image.fromarray(orange_building)
        orange_building_image.save(f'{results_path}/orange_building/{filename}')

        # Red H
        red_h = filter_on_color(image, [240, 50, 50], [5, 255, 255])
        red_h_image = Image.fromarray(red_h)
        red_h_image.save(f'{results_path}/red_h/{filename}')


if __name__ == '__main__':
    main()
