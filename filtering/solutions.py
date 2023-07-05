import numpy as np
import cv2 as cv

def meanKernelRef(kernel_size : tuple[int, int]) -> np.ndarray:
    """
    Parameters:
    - kernel_size : tuple[int, int]
        A tuple of two integers, which determines the size of the kernel.
    Returns:
    - kernel : np.ndarray
        A mean kernel in np.ndarray format, with the specified size,
        and with dtype=np.float32.
    """
    if not isinstance(kernel_size, tuple) or len(kernel_size) != 2 or not isinstance(kernel_size[0], int) or not isinstance(kernel_size[1], int):
        raise TypeError(f'Parameter \'kernel_size\' should be a tuple of two integers.')
        
    if kernel_size[0] <= 0 or kernel_size[1] <= 0:
        raise ValueError(f'Parameter \'kernel_size\' should be a tuple of two positive integers.')
        
    kernel = np.ndarray(kernel_size)
    kernel = kernel / np.sum(kernel)
    return kernel

def gaussianKernelRef(kernel_size : tuple[int, int], sigma : float) -> np.ndarray:
    """
    Parameters:
    - kernel_size : tuple[int, int]
        A tuple of two integers, which determines the size of the kernel.
    - sigma : float
        The sigma parameter in the gaussian distribution.
    Returns:
    - kernel : np.ndarray
        A gaussian kernel in np.ndarray format, with the specified size,
        and with dtype=np.float32.
    """
    if not isinstance(kernel_size, tuple) or len(kernel_size) != 2 or not isinstance(kernel_size[0], int) or not isinstance(kernel_size[1], int):
        raise TypeError(f'Parameter \'kernel_size\' should be a tuple of two integers.')
    if not (isinstance(sigma, int) or isinstance(sigma, float)):
        raise TypeError(f'Parameter \'sigma\' should be a float or an integer for Sobel kernels.')

    if kernel_size[0] <= 0 or kernel_size[1] <= 0:
        raise ValueError(f'Parameter \'kernel_size\' should be a tuple of two positive integers.')
    if sigma <= 0:
        raise ValueError(f'Parameter \'sigma\' should be larger than zero.')
        
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
        characteristics and with dtype=np.float32.
    """
    if direction != 'h' and direction != 'v':
        raise ValueError(f'Parameter \'direction\' should be either \'h\' or \'v\'.')
    if mode != 'c' and mode != 'f':
        raise ValueError(f'Parameter \'mode\' should be either \'c\' or \'f\'.')

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
        An image in np.ndarray format, with dtype=np.float32.
        
    Returns
    - gradient_magnitude : np.ndarray
        An np.ndarray representation of gradient magnitudes, with dtype=np.float32.
    - gradient_orientation : np.ndarray
        An np.ndarray representation of gradient orientations. Values can range from
        -pi to +pi. Should also have dtype=np.float32.
    
    """
    if not isinstance(image, np.ndarray) or image.dtype != np.float32:
        raise TypeError(f'parameter \'image\' should be an np.ndarray with dtype=np.float32.')
        
    dx_kernel = derivateKernelRef('h', 'f')
    dy_kernel = derivateKernelRef('v', 'f')
    dx = cv.filter2D(image, cv.CV_32F, dx_kernel)
    dy = cv.filter2D(image, cv.CV_32F, dy_kernel)
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
        dtype=np.float32.
    """
    if not isinstance(radius, int):
        raise TypeError(f'Parameter \'radius\' should be an integer.')
    if kernel_type == 'g' and not (isinstance(radius, int) or isinstance(radius, float)):
        raise TypeError(f'Parameter \'sigma\' should be a float or an integer for Sobel kernels.')
        
    if direction != 'h' and direction != 'v':
        raise ValueError(f'Parameter \'direction\' should be either \'h\' or \'v\'.')
    if kernel_type != 'g' and kernel_type != 'm':
        raise ValueError(f'Parameter \'kernel_type\' should be either \'g\' or \'m\'.')
    if radius <= 0:
        raise ValueError(f'Parameter \'radius\' should be a positive integer.')
    if kernel_type == 'g' and sigma <= 0:
        raise ValueError(f'Parameter \'sigma\' should be larger than zero.')

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
        characteristics and with dtype=np.float32.
    """
    if direction != 'h' and direction != 'v':
        raise ValueError(f'Parameter \'direction\' should be either \'h\' or \'v\'.')

    kernel = np.array([[1, -2, 1]], dtype=np.float32)

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
        with dtype=np.float32.
    """
    if not isinstance(alpha, int) and not isinstance(alpha, float):
        raise TypeError(f'Parameter \'alpha\' should be a float or an integer.')
    if alpha > 1 or alpha < 0:
        raise ValueError(f'Parameter \'direction\' should be in [0 1].')
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
    if not isinstance(c, int) and not isinstance(c, float):
        raise TypeError(f'Parameter \'c\' should be a float or an integer.')
    if c < 0:
        raise ValueError(f'Parameter \'direction\' should be a non-negative number.')
        
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
    if not isinstance(c, int) and not isinstance(c, float):
        raise TypeError(f'Parameter \'c\' should be a float or an integer.')
    if c < 0:
        raise ValueError(f'Parameter \'direction\' should be a non-negative number.')

    kernel = laplacianKernelRef(0)
    image_lap = cv.filter2D(image, cv.CV_32F, kernel)
    output = image - c * image_lap
    np.clip(output, 0., 1.)
    return output
    