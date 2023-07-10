import numpy as np
import cv2 as cv
from typing import Iterable

def meanKernelRef(kernel_size : Iterable[int]) -> np.ndarray:
    """
    Parameters:
    - kernel_size : Iterable[int, int]
        An Iterable of two integers, which determines the size of the kernel.
    Returns:
    - kernel : np.ndarray
        A mean kernel in np.ndarray format, with the specified size,
        and with dtype=np.float64.
    """
    assert isinstance(kernel_size, Iterable) and len(kernel_size) == 2 and \
    all(isinstance(s, int) and s > 0 for s in kernel_size), 'Parameter \'kernel_size\' should be an Iterable of two positive integers.'
    
    kernel = np.ndarray(kernel_size)
    kernel = kernel / np.sum(kernel)
    return kernel

def gaussianKernelRef(kernel_size : Iterable[int], sigma : float) -> np.ndarray:
    """
    Parameters:
    - kernel_size : Iterable[int]
        An Iterable of two integers, which determines the size of the kernel.
    - sigma : float
        The sigma parameter in the gaussian distribution.
    Returns:
    - kernel : np.ndarray
        A gaussian kernel in np.ndarray format, with the specified size,
        and with dtype=np.float64.
    """
    assert isinstance(kernel_size, Iterable) and len(kernel_size) == 2 and \
    all(isinstance(s, int) and s > 0 for s in kernel_size), 'Parameter \'kernel_size\' should be an Iterable of two positive integers.'
    assert (isinstance(sigma, int) or isinstance(sigma, float)) and sigma > 0, 'Parameter \'sigma\' should be a positive number.'
        
    h, w = kernel_size
    kernel_i, kernel_j = cv.getGaussianKernel(h, sigma), cv.getGaussianKernel(w, sigma).T
    return kernel_i @ kernel_j

def derivateKernelRef(direction : str, mode : str) -> np.ndarray:
    """
    Parameters:
    - direction : ['h' | 'v']
        Determines the axis along which derivation takes place. 'h' for horizontal
        and 'v' for vertical differentiation.
    - mode : ['c' | 'f']
        Determines the type of differentiation. 'c' for central difference, and 'f'
        for forward difference.
    Returns:
    - kernel : np.ndarray
        A derivative kernel in np.ndarray format, with the specified 
        characteristics and with dtype=np.float64.
    """
    assert direction == 'h' or direction == 'v', 'Parameter \'direction\' should be in [\'h\' | \'v\'].'
    assert mode == 'c' or mode == 'f', 'Parameter \'mode\' should be in [\'c\' | \'f\'].'

    if mode == 'c':
        kernel = np.array([[-1, 0, 1]])
    else:
        kernel = np.array([[-1, 1]])

    if direction == 'v':
        kernel = kernel.T
    
    return kernel

def gradientMapRef(image : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters:
    - image : np.ndarray
        An image in np.ndarray format, with dtype=np.float64.
        
    Returns
    - gradient_magnitude : np.ndarray
        An np.ndarray representation of gradient magnitudes, with dtype=np.float64.
    - gradient_orientation : np.ndarray
        An np.ndarray representation of gradient orientations. Values can range from
        -pi to +pi. Should also have dtype=np.float64.
    
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.float64, 'Parameter \'image\' should be an np.ndarray with dtype=np.float64.'
        
    dx_kernel = derivateKernelRef('h', 'f')
    dy_kernel = derivateKernelRef('v', 'f')
    dx = cv.filter2D(image, cv.CV_64F, dx_kernel)
    dy = cv.filter2D(image, cv.CV_64F, dy_kernel)
    gradient_magnitude = np.sqrt(dx**2 + dy**2)
    gradient_orientation = np.arctan2(dy, dx)
    return gradient_magnitude, gradient_orientation

def smoothDerivativeKernelRef(direction : str, kernel_type : str, radius : int, sigma : float = None) -> np.ndarray:
    """
    Parameters:
    - direction : ['h' | 'v']
        Determines the axis along which derivation takes place. 'h' for horizontal
        and 'v' for vertical differentiation.
    - kernel_type : ['g' | 'm']
        Determines the type of the smoothing kernel. 'g' for gaussian, and 'm' for 
        mean.
    - radius : int
        Determines the radius of the smoothing kernel.
    - sigma : float
        Determines the sigma parameter for gaussian kernels.
    Returns:
    - kernel : np.ndarray
        A kernel in np.ndarray format, with the specified characteristics and with
        dtype=np.float64.
    """
    assert direction == 'h' or direction == 'v', 'Parameter \'direction\' should be in [\'h\' | \'v\'].'
    assert kernel_type == 'g' or kernel_type == 'm', 'Parameter \'kernel_type\' should be in [\'g\' | \'m\'].'
    assert isinstance(radius, int) and radius > 0, 'Parameter \'radius\' should be a positive integer.'
    assert (kernel_type == 'm' and sigma == None) or ((isinstance(sigma, float) or isinstance(sigma, int)) and sigma > 0),\
    'Parameter \'sigma\' should be a positive number when \'kernel_type\' is \'g\', and should be left empty if \'kernel_type\' is \'m\'.'
    
    if kernel_type == 'g':
        smoothing_kernel = cv.getGaussianKernel(radius, sigma)
    else:
        smoothing_kernel = np.ones((1, radius)) / radius

    derivation_kernel = np.array([[-1, 0, 1]])
    kernel = smoothing_kernel @ derivation_kernel    
    if direction == 'v':
        kernel = kernel.T

    return kernel

def secondDerivateKernelRef(direction : str) -> np.ndarray:
    """
    Parameters:
    - direction : ['h' | 'v']
        Determines the axis along which derivation takes place. 'h' for horizontal
        and 'v' for vertical differentiation.
    Returns:
    - kernel : np.ndarray
        A derivative kernel in np.ndarray format, with the specified 
        characteristics and with dtype=np.float64.
    """
    assert direction == 'h' or direction == 'v', 'Parameter \'direction\' should be in [\'h\' | \'v\'].'

    kernel = np.array([[1, -2, 1]], dtype=np.float64)

    if direction == 'v':
        kernel = kernel.T

    return kernel

def laplacianKernelRef(alpha : float) -> np.ndarray:
    """
    Parameters:
    - alpha : float [0 1]
        The alpha parameter in the Laplacian kernel.
    Returns:
    - kernel : np.ndarray
        A Laplacian kernel in np.ndarray format, with the specified direction and
        with dtype=np.float64.
    """
    assert (isinstance(alpha, float) or isinstance(alpha, int)) and alpha <= 1 and alpha >= 0, 'Parameter \'alpha\' should be a number in [0 1].'
    
    kernel = np.arange(3).reshape(3, 1) + np.arange(3).reshape(1, 3)
    kernel = alpha * (kernel % 2 == 0) + (1 - alpha) * (kernel % 2 == 1)
    kernel[1,1] = -4
    return kernel

def sharpeningLPRef(image : np.ndarray, c : float) -> np.ndarray:
    """
    Parameters:
    - image : np.ndarray
        The image which the function will sharpen
    - c : float [0 +inf)
        The sharpening factor.
    Returns:
    - output : np.ndarray
        A sharpened version of image.
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.float64, 'Parameter \'image\' should be an np.ndarray with dtype=np.float64.'
    assert (isinstance(c, float) or isinstance(c, int)) and c >= 0, 'Parameter \'direction\' should be a non-negative number.'
        
    image_lp = cv.blur(image, (5, 5))
    output = (1 + c) * image - c * image_lp
    return output

def sharpeningUnsharpRef(image : np.ndarray, c : float) -> np.ndarray:
    """
    Parameters:
    - image : np.ndarray
        The image which the function will sharpen
    - c : float [0 +inf)
        The sharpening factor.
    Returns:
    - output : np.ndarray
        A sharpened version of image.
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.float64, 'Parameter \'image\' should be an np.ndarray with dtype=np.float64.'
    assert (isinstance(c, float) or isinstance(c, int)) and c >= 0, 'Parameter \'direction\' should be a non-negative number.'

    kernel = laplacianKernelRef(0)
    image_lap = cv.filter2D(image, cv.CV_64F, kernel)
    output = image - c * image_lap
    np.clip(output, 0., 1.)
    return output

def padAndTransformRef(image : np.ndarray) -> np.ndarray:
    """
    Parameters:
    - image : np.ndarray
        A grayscale image whose Fourier transform will be calculated.
    Returns:
    - output : np.ndarray
        The shifted fourier transform of the padded image, with twice the size.
    """
    assert isinstance(image, np.ndarray) and image.ndim == 2, 'Parameter \'image\' should be an np.ndarray.'
    
    h, w = image.shape
    image_padded = np.zeros((h*2, w*2), dtype=image.dtype)
    image_padded[:h,:w] = image
    output = np.fft.fftshift(np.fft.fft2(image_padded))
    return output

def padRef(image : np.ndarray) -> np.ndarray:
    """
    Parameters:
    - image : np.ndarray
        A grayscale image whose Fourier transform will be calculated.
    Returns:
    - output : np.ndarray
        The padded image, with twice the size.
    """
    assert isinstance(image, np.ndarray) and image.ndim == 2, 'Parameter \'image\' should be an np.ndarray.'
    
    h, w = image.shape
    output = np.zeros((h*2, w*2), dtype=image.dtype)
    output[:h,:w] = image
    output[h:,:w] = np.flip(image, axis=0)
    output[:h,w:] = np.flip(image, axis=1)
    output[h:,w:] = np.flip(image, axis=[0, 1])
    
    return output

def idealLowPassRef(image : np.ndarray, threshold : float):
    """
    Parameters:
    - image : np.ndarray
        An image on which the filtering will be applied. It should be a np.ndarray
        with dtype=float64.
    - threshold : float [0 inf)
        The highest frequency which should be allowed to pass. Should be a non-
        negative float.
    Returns:
    - output : np.ndarray
        The filtered image, with dtype=np.float64.
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.float64, 'Parameter \'image\' should be an np.ndarray with dtype=np.float64.'
    assert (isinstance(threshold, int) or isinstance(threshold, float)) and threshold >= 0, 'Parameter \'threshold\' should be a non-negative number.'

    h0, w0 = image.shape
    image_padded = padRef(image)
    
    h, w = image_padded.shape
    pos_i = np.linspace(-(h//2) / h * 2 * np.pi, ((h-1)//2) / h * 2 * np.pi, h).reshape(h, 1)
    pos_j = np.linspace(-(w//2) / h * 2 * np.pi, ((h-1)//2) / h * 2 * np.pi, h).reshape(1, w)
    filter = np.sqrt(pos_i**2 + pos_j**2) < threshold

    image_f = np.fft.fftshift(np.fft.fft2(image_padded))
    image_f_filtered = image_f * filter
    output = np.real(np.fft.ifft2(np.fft.ifftshift(image_f_filtered)))[:h0,:w0]

    return output

def gaussianLowPassRef(image : np.ndarray, sigma : float) -> np.ndarray:
    """
    Parameters:
    - image : np.ndarray
        An image on which the filtering will be applied. It should be a np.ndarray
        with dtype=float64.
    - sigma : float
        The sigma value for the Gaussian function. Should be a positive number.
    Returns:
    - output : np.ndarray
        The filtered image, with dtype=np.float64.
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.float64, 'Parameter \'image\' should be an np.ndarray with dtype=np.float64.'
    assert (isinstance(sigma, int) or isinstance(sigma, float)) and sigma > 0, 'Parameter \'sigma\' should be a non-negative number.'

    h0, w0 = image.shape
    image_padded = padRef(image)
    
    h, w = image_padded.shape
    pos_i = np.linspace(-(h//2) / h * 2 * np.pi, ((h-1)//2) / h * 2 * np.pi, h).reshape(h, 1)
    pos_j = np.linspace(-(w//2) / h * 2 * np.pi, ((h-1)//2) / h * 2 * np.pi, h).reshape(1, w)
    
    filter = np.exp(-(pos_i**2 + pos_j**2) / sigma**2 / 2)

    image_f = np.fft.fftshift(np.fft.fft2(image_padded))
    image_f_filtered = image_f * filter
    output = np.real(np.fft.ifft2(np.fft.ifftshift(image_f_filtered)))[:h0,:w0]

    return output

def butterworthLowPassRef(image : np.ndarray, d : float, order : float) -> np.ndarray:
    """
    Parameters:
    - image : np.ndarray
        An image on which the filtering will be applied. It should be a np.ndarray
        with dtype=float64.
    - d : float
        The d value for the Butterworth filter. Should be a positive number.
    - order : float
        The order of the Butterworth filter. Should be a positive number.
    Returns:
    - output : np.ndarray
        The filtered image, with dtype=np.float64.
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.float64, 'Parameter \'image\' should be an np.ndarray with dtype=np.float64.'
    assert (isinstance(d, int) or isinstance(d, float)) and d > 0, 'Parameter \'d\' should be a non-negative number.'
    assert (isinstance(order, int) or isinstance(order, float)) and order > 0, 'Parameter \'order\' should be a non-negative number.'

    h0, w0 = image.shape
    image_padded = padRef(image)
    
    h, w = image_padded.shape
    pos_i = np.linspace(-(h//2) / h * 2 * np.pi, ((h-1)//2) / h * 2 * np.pi, h).reshape(h, 1)
    pos_j = np.linspace(-(w//2) / h * 2 * np.pi, ((h-1)//2) / h * 2 * np.pi, h).reshape(1, w)
    
    filter = 1 / (1 + ((pos_i**2 + pos_j**2) / d**2)**order)

    image_f = np.fft.fftshift(np.fft.fft2(image_padded))
    image_f_filtered = image_f * filter
    output = np.real(np.fft.ifft2(np.fft.ifftshift(image_f_filtered)))[:h0,:w0]

    return output

def gaussianHighPassRef(image : np.ndarray, sigma : float) -> np.ndarray:
    """
    Parameters:
    - image : np.ndarray
        An image on which the filtering will be applied. It should be a np.ndarray
        with dtype=float64.
    - sigma : float
        The sigma value for the Gaussian function. Should be a positive number.
    Returns:
    - output : np.ndarray
        The filtered image, with dtype=np.float64.
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.float64, 'Parameter \'image\' should be an np.ndarray with dtype=np.float64.'
    assert (isinstance(sigma, int) or isinstance(sigma, float)) and sigma > 0, 'Parameter \'sigma\' should be a non-negative number.'

    h0, w0 = image.shape
    image_padded = padRef(image)
    
    h, w = image_padded.shape
    pos_i = np.linspace(-(h//2) / h * 2 * np.pi, ((h-1)//2) / h * 2 * np.pi, h).reshape(h, 1)
    pos_j = np.linspace(-(w//2) / h * 2 * np.pi, ((h-1)//2) / h * 2 * np.pi, h).reshape(1, w)
    
    filter = 1 - np.exp(-(pos_i**2 + pos_j**2) / sigma**2 / 2)

    image_f = np.fft.fftshift(np.fft.fft2(image_padded))
    image_f_filtered = image_f * filter
    output = np.real(np.fft.ifft2(np.fft.ifftshift(image_f_filtered)))[:h0,:w0]

    return output

def butterworthHighPassRef(image : np.ndarray, d : float, order : float) -> np.ndarray:
    """
    Parameters:
    - image : np.ndarray
        An image on which the filtering will be applied. It should be a np.ndarray
        with dtype=float64.
    - d : float
        The d value for the Butterworth filter. Should be a positive number.
    - order : float
        The order of the Butterworth filter. Should be a positive number.
    Returns:
    - output : np.ndarray
        The filtered image, with dtype=np.float64.
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.float64, 'Parameter \'image\' should be an np.ndarray with dtype=np.float64.'
    assert (isinstance(d, int) or isinstance(d, float)) and d > 0, 'Parameter \'d\' should be a positive number.'
    assert (isinstance(order, int) or isinstance(order, float)) and order > 0, 'Parameter \'order\' should be a positive number.'

    h0, w0 = image.shape
    image_padded = padRef(image)
    
    h, w = image_padded.shape
    pos_i = np.linspace(-(h//2) / h * 2 * np.pi, ((h-1)//2) / h * 2 * np.pi, h).reshape(h, 1)
    pos_j = np.linspace(-(w//2) / h * 2 * np.pi, ((h-1)//2) / h * 2 * np.pi, h).reshape(1, w)
    
    filter = 1 - 1 / (1 + ((pos_i**2 + pos_j**2) / d**2)**order)

    image_f = np.fft.fftshift(np.fft.fft2(image_padded))
    image_f_filtered = image_f * filter
    output = np.real(np.fft.ifft2(np.fft.ifftshift(image_f_filtered)))[:h0,:w0]

    return output

def gaussianBandRejectRef(image : np.ndarray, f : float, W : float) -> np.ndarray:
    """
    Parameters:
    - image : np.ndarray
        An image on which the filtering will be applied. It should be a np.ndarray
        with dtype=float64.
    - f : float
        The reject frequency for the filter. Should be a positive number.
    - W : float
        The width of the rejected band. Should be a positive number and lower than
        2f.
    Returns:
    - output : np.ndarray
        The filtered image, with dtype=np.float64.
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.float64, 'Parameter \'image\' should be an np.ndarray with dtype=np.float64.'
    assert (isinstance(f, int) or isinstance(f, float)) and f > 0, 'Parameter \'f\' should be a positive number smaller than 2f.'
    assert (isinstance(W, int) or isinstance(W, float)) and W > 0 and f > W/2, 'Parameter \'W\' should be a positive number.'

    h0, w0 = image.shape
    image_padded = padRef(image)
    
    h, w = image_padded.shape
    pos_i = np.linspace(-(h//2) / h * 2 * np.pi, ((h-1)//2) / h * 2 * np.pi, h).reshape(h, 1)
    pos_j = np.linspace(-(w//2) / h * 2 * np.pi, ((h-1)//2) / h * 2 * np.pi, h).reshape(1, w)

    eps = 1e-15
    
    d = np.sqrt(pos_i**2 + pos_j**2)
    g = ((d**2 - f**2) / (W * (d + eps)))**2
    
    filter = 1 - np.exp(-g)

    image_f = np.fft.fftshift(np.fft.fft2(image_padded))
    image_f_filtered = image_f * filter
    output = np.real(np.fft.ifft2(np.fft.ifftshift(image_f_filtered)))[:h0,:w0]

    return output

def butterworthBandRejectRef(image : np.ndarray, f : float, W : float, order : float) -> np.ndarray:
    """
    Parameters:
    - image : np.ndarray
        An image on which the filtering will be applied. It should be a np.ndarray
        with dtype=float64.
    - f : float
        The reject frequency for the filter. Should be a positive number.
    - W : float
        The width of the rejected band. Should be a positive number and lower than
        2f.
    - order : float
        The order of the Butterworth filter. should be a positive number.
    Returns:
    - output : np.ndarray
        The filtered image, with dtype=np.float64.
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.float64, 'Parameter \'image\' should be an np.ndarray with dtype=np.float64.'
    assert (isinstance(f, int) or isinstance(f, float)) and f > 0, 'Parameter \'f\' should be a positive number.'
    assert (isinstance(W, int) or isinstance(W, float)) and W > 0 and f > W/2, 'Parameter \'W\' should be a positive number smaller than 2f.'
    assert (isinstance(order, int) or isinstance(order, float)) and order > 0, 'Parameter \'order\' should be a positive number.'

    h0, w0 = image.shape
    image_padded = padRef(image)
    
    h, w = image_padded.shape
    pos_i = np.linspace(-(h//2) / h * 2 * np.pi, ((h-1)//2) / h * 2 * np.pi, h).reshape(h, 1)
    pos_j = np.linspace(-(w//2) / h * 2 * np.pi, ((h-1)//2) / h * 2 * np.pi, h).reshape(1, w)

    eps = 1e-15
    
    d = np.sqrt(pos_i**2 + pos_j**2)
    g = ((d**2 - f**2) / (W * (d)))**2
    
    filter = 1 - 1 / (1 + g**order)

    image_f = np.fft.fftshift(np.fft.fft2(image_padded))
    image_f_filtered = image_f * filter
    output = np.real(np.fft.ifft2(np.fft.ifftshift(image_f_filtered)))[:h0,:w0]

    return output

def gaussianBandPassRef(image : np.ndarray, f : float, W : float) -> np.ndarray:
    """
    Parameters:
    - image : np.ndarray
        An image on which the filtering will be applied. It should be a np.ndarray
        with dtype=float64.
    - f : float
        The pass frequency for the filter. Should be a positive number.
    - W : float
        The width of the passed band. Should be a positive number and lower than
        2f.
    Returns:
    - output : np.ndarray
        The filtered image, with dtype=np.float64.
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.float64, 'Parameter \'image\' should be an np.ndarray with dtype=np.float64.'
    assert (isinstance(f, int) or isinstance(f, float)) and f > 0, 'Parameter \'f\' should be a positive number smaller than 2f.'
    assert (isinstance(W, int) or isinstance(W, float)) and W > 0 and f > W/2, 'Parameter \'W\' should be a positive number.'

    h0, w0 = image.shape
    image_padded = padRef(image)
    
    h, w = image_padded.shape
    pos_i = np.linspace(-(h//2) / h * 2 * np.pi, ((h-1)//2) / h * 2 * np.pi, h).reshape(h, 1)
    pos_j = np.linspace(-(w//2) / h * 2 * np.pi, ((h-1)//2) / h * 2 * np.pi, h).reshape(1, w)

    eps = 1e-15
    
    d = np.sqrt(pos_i**2 + pos_j**2)
    g = ((d**2 - f**2) / (W * (d + eps)))**2
    
    filter = np.exp(-g)

    image_f = np.fft.fftshift(np.fft.fft2(image_padded))
    image_f_filtered = image_f * filter
    output = np.real(np.fft.ifft2(np.fft.ifftshift(image_f_filtered)))[:h0,:w0]

    return output

def butterworthBandPassRef(image : np.ndarray, f : float, W : float, order : float) -> np.ndarray:
    """
    Parameters:
    - image : np.ndarray
        An image on which the filtering will be applied. It should be a np.ndarray
        with dtype=float64.
    - f : float
        The pass frequency for the filter. Should be a positive number.
    - W : float
        The width of the passed band. Should be a positive number and lower than
        2f.
    - order : float
        The order of the Butterworth filter. should be a positive number.
    Returns:
    - output : np.ndarray
        The filtered image, with dtype=np.float64.
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.float64, 'Parameter \'image\' should be an np.ndarray with dtype=np.float64.'
    assert (isinstance(f, int) or isinstance(f, float)) and f > 0, 'Parameter \'f\' should be a positive number.'
    assert (isinstance(W, int) or isinstance(W, float)) and W > 0 and f > W/2, 'Parameter \'W\' should be a positive number smaller than 2f.'
    assert (isinstance(order, int) or isinstance(order, float)) and order > 0, 'Parameter \'order\' should be a positive number.'

    h0, w0 = image.shape
    image_padded = padRef(image)
    
    h, w = image_padded.shape
    pos_i = np.linspace(-(h//2) / h * 2 * np.pi, ((h-1)//2) / h * 2 * np.pi, h).reshape(h, 1)
    pos_j = np.linspace(-(w//2) / h * 2 * np.pi, ((h-1)//2) / h * 2 * np.pi, h).reshape(1, w)

    eps = 1e-15
    
    d = np.sqrt(pos_i**2 + pos_j**2)
    g = ((d**2 - f**2) / (W * (d)))**2
    
    filter = 1 / (1 + g**order)

    image_f = np.fft.fftshift(np.fft.fft2(image_padded))
    image_f_filtered = image_f * filter
    output = np.real(np.fft.ifft2(np.fft.ifftshift(image_f_filtered)))[:h0,:w0]

    return output