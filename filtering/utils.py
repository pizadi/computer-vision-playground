import numpy as np
from typing import Iterable
from matplotlib import pyplot as plt
import cv2 as cv

def gaussianNoise(image : np.ndarray, sigma : float) -> np.ndarray:
    assert isinstance(image, np.ndarray) and image.dtype == np.float64 and image.ndim == 2, 'Parameter \'image\' should be a valid np.ndarray image with dtype=np.float64.'
    assert (isinstance(sigma, float) or isinstance(sigma, int)) and sigma >= 0, 'Parameter \'sigma\' should be a non-negative number.'
    noise = np.float64(np.random.normal(0, sigma, size=image.shape))
    output = image + noise
    output = np.clip(output, 0., 1.)
    return output


def lowPassFilter(image_shape) -> np.ndarray:
    assert isinstance(image_shape, Iterable) and len(image_shape) == 2 and \
    all(isinstance(s, int) and s > 0 for s in image_shape), 'Parameter \'image_shape\' should be an iterable of two integers.'

    h, w = image_shape
    l_h = np.linspace(-h/2, h/2, h).reshape(h, 1) / h * 2
    l_w = np.linspace(-w/2, w/2, w).reshape(1, w) / w * 2
    filter = -(l_h**2 + l_w**2) / .07**2 / 2
    filter = np.exp(filter)
    filter = np.complex128(filter)
    return filter

def sharpeningFilter(image_shape) -> np.ndarray:
    assert isinstance(image_shape, Iterable) and len(image_shape) == 2 and \
    all(isinstance(s, int) and s > 0 for s in image_shape), 'Parameter \'image_shape\' should be an iterable of two integers.'

    h, w = image_shape
    l_h = np.linspace(-h/2, h/2, h).reshape(h, 1)
    l_w = np.linspace(-w/2, w/2, w).reshape(1, w)
    filter = -(l_h**2 + l_w**2) / 50**2 / 2
    filter = 2 - np.exp(filter)
    filter = np.complex128(filter)
    return filter
    
def paddingDemo(image : np.ndarray) -> None:
    assert isinstance(image, np.ndarray) and image.ndim == 2 and image.dtype == np.float64 \
    and np.max(image) <= 1. and np.min(image) >= 0., 'Parameter \'image\' should be a valid np.ndarray image with dtype=np.float64.'

    h, w = image.shape
    
    image_zero_pad = np.zeros((h*2, w*2), dtype=image.dtype)
    image_zero_pad[:h,:w] = image
    
    image_mirror_pad = np.copy(image_zero_pad)
    image_mirror_pad[h:,:w] = np.flip(image, axis=0)
    image_mirror_pad[:h,w:] = np.flip(image, axis=1)
    image_mirror_pad[h:,w:] = np.flip(image, axis=[0, 1])

    image_nearest_pad = np.copy(image_zero_pad)
    image_nearest_pad[h:h + h//2,:w] = image[-1:,:]
    image_nearest_pad[h + h//2:,:w] = image[:1,:]
    image_nearest_pad[h:,w:w + w//2] = image[:,-1:]
    image_nearest_pad[h:,w + w//2:] = image[:,:1]
    image_nearest_pad[h:h + h//2, w:w + w//2] = image[-1,-1]
    image_nearest_pad[h:h + h//2, w + w//2:] = image[-1,0]
    image_nearest_pad[h + h//2:, w:w + w//2] = image[0,-1]
    image_nearest_pad[h + h//2:, w + w//2:] = image[0,0]

    filter_pad = lowPassFilter(image_mirror_pad.shape)
    filter_no_pad = lowPassFilter(image.shape)

    filter = lambda img, filt : np.real(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(img)) * filt)))
    
    image_no_pad_filtered = filter(image, filter_no_pad)
    image_zero_pad_filtered = filter(image_zero_pad, filter_pad)[:h,:w]
    image_mirror_pad_filtered = filter(image_mirror_pad, filter_pad)[:h,:w]
    image_nearest_pad_filtered = filter(image_nearest_pad, filter_pad)[:h,:w]

    _ = plt.set_cmap('Greys_r')
    _ = plt.figure(figsize=(15, 15))
    _ = plt.subplot(2, 2, 1), plt.imshow(image_no_pad_filtered, vmin=0, vmax=1), plt.axis('off'), plt.title('No Padding')
    _ = plt.subplot(2, 2, 2), plt.imshow(image_zero_pad_filtered, vmin=0, vmax=1), plt.axis('off'), plt.title('Zero Padding')
    _ = plt.subplot(2, 2, 3), plt.imshow(image_mirror_pad_filtered, vmin=0, vmax=1), plt.axis('off'), plt.title('Mirror Padding')
    _ = plt.subplot(2, 2, 4), plt.imshow(image_nearest_pad_filtered, vmin=0, vmax=1), plt.axis('off'), plt.title('Nearest Padding')

def spatialKernelDemo(image_shape : Iterable[int]) -> None:
    assert isinstance(image_shape, Iterable) and len(image_shape) == 2 and \
    all(isinstance(s, int) and s > 0 for s in image_shape), 'Parameter \'image_shape\' should be an iterable of two integers.'
    
    mean_kernel = np.zeros(image_shape, dtype=np.float64)
    mean_kernel[:3,:3] = 1 / 9

    gaussian_kernel = np.zeros(image_shape, dtype=np.float64)
    gaussian_kernel[:3,:3] = cv.getGaussianKernel(3, 1) @ cv.getGaussianKernel(3, 1).T

    prewitt_horizontal_kernel = np.zeros(image_shape, dtype=np.float64)
    prewitt_horizontal_kernel[:3,0] = -1 / 3
    prewitt_horizontal_kernel[:3,2] = 1 / 3

    prewitt_vertical_kernel = np.zeros(image_shape, dtype=np.float64)
    prewitt_vertical_kernel[0,:3] = -1 / 3
    prewitt_vertical_kernel[2,:3] = 1 / 3

    laplacian_kernel = np.zeros(image_shape, dtype=np.float64)
    laplacian_kernel[0,1] = laplacian_kernel[1,0] = laplacian_kernel[2,1] = laplacian_kernel[1,2] = 1
    laplacian_kernel[1,1] = -4

    unsharp_kernel = np.zeros(image_shape, dtype=np.float64)
    unsharp_kernel[0,1] = unsharp_kernel[1,0] = unsharp_kernel[2,1] = unsharp_kernel[1,2] = -1
    unsharp_kernel[1,1] = 5

    mean_f = np.fft.fftshift(np.fft.fft2(mean_kernel))
    gaussian_f = np.fft.fftshift(np.fft.fft2(gaussian_kernel))
    prewitt_horizontal_f = np.fft.fftshift(np.fft.fft2(prewitt_horizontal_kernel))
    prewitt_vertical_f = np.fft.fftshift(np.fft.fft2(prewitt_vertical_kernel))
    laplacian_f = np.fft.fftshift(np.fft.fft2(laplacian_kernel))
    unsharp_f = np.fft.fftshift(np.fft.fft2(unsharp_kernel))

    _ = plt.set_cmap('Greys_r')
    _ = plt.figure(figsize=(18, 12))
    _ = plt.subplot(2, 3, 1), plt.imshow(np.abs(mean_f), vmin=0), plt.axis('off'), plt.title('Mean Filter')
    _ = plt.subplot(2, 3, 2), plt.imshow(np.abs(gaussian_f), vmin=0), plt.axis('off'), plt.title('Gaussian Filter')
    _ = plt.subplot(2, 3, 3), plt.imshow(np.abs(laplacian_f), vmin=0), plt.axis('off'), plt.title('Laplacian Filter')
    _ = plt.subplot(2, 3, 4), plt.imshow(np.abs(prewitt_horizontal_f), vmin=0), plt.axis('off'), plt.title('Horizonal Prewitt Filter')
    _ = plt.subplot(2, 3, 5), plt.imshow(np.abs(prewitt_vertical_f), vmin=0), plt.axis('off'), plt.title('Vertical Prewitt Filter')
    _ = plt.subplot(2, 3, 6), plt.imshow(np.abs(unsharp_f), vmin=0), plt.axis('off'), plt.title('Unsharp Filter')

def drawGaussianCrossSection(f : float, w : float, resolution : int) -> None:
    assert (isinstance(f, float) or isinstance(f, int)) and f > 0, 'Parameter \'f\' should be a positive number.'
    assert (isinstance(w, float) or isinstance(w, int)) and w > 0 and f > w/2, 'Parameter \'w\' should be a positive number smaller that 2f.'
    assert isinstance(resolution, int) and resolution > 1, 'Parameter \'resolution\' should be a positive integer.'

    pos = np.linspace(0, 2 * np.pi, resolution * 4)

    gaussian_low_pass = np.exp(- pos**2 / (f - w)**2 )
    gaussian_high_pass = 1 - np.exp(- pos**2 / (f + w)**2)

    _ = plt.figure(figsize=(10, 5))
    plt.axis((0, 2, -0.01, 1))
    _ = plt.plot(pos / np.pi, gaussian_low_pass + gaussian_high_pass, 'r-')
    plt.title(f'Cross section of a Gaussian BRF with f={f/np.pi:.4f}*pi and W={w/np.pi:.4f}*pi')
    
