import numpy as np
import cv2 as cv
from numpy import random
from typing import Union, Iterable

def addSNPNoiseRef(image : np.ndarray, p : float) -> np.ndarray:
    """
    Parameters:
    - image : np.ndarray
        An image on which the Salt & Pepper noise will be applied. It should be a
        np.ndarray with dtype=float64.
    - p : float [0 1]
        The p parameter in the S&P noise distribution. Should be in [0 1].
    Returns:
    - output : np.ndarray
        The noisy image, with dtype=np.float64, and values within [0 1].
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.float64, 'Parameter \'image\' should be an np.ndarray with dtype=np.float64.'
    assert (isinstance(p, float) or isinstance(p, int)) and p >= 0 and p <= 1, 'Parameter \'p\' should be a number within [0 1].'

    snp_map = random.randint(0, 2, image.shape) * 2 - 1
    noise_map = random.rand(*image.shape) < p
    output = image + noise_map * snp_map
    output = np.clip(output, 0., 1.)
    return output

def addSaltNoiseRef(image : np.ndarray, p : float) -> np.ndarray:
    """
    Parameters:
    - image : np.ndarray
        An image on which the salt noise will be applied. It should be a np.ndarray
        with dtype=float64.
    - p : float [0 1]
        The p parameter in the salt noise distribution. Should be in [0 1].
    Returns:
    - output : np.ndarray
        The noisy image, with dtype=np.float64, and values within [0 1].
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.float64, 'Parameter \'image\' should be an np.ndarray with dtype=np.float64.'
    assert (isinstance(p, float) or isinstance(p, int)) and p >= 0 and p <= 1, 'Parameter \'p\' should be a number within [0 1].'

    snp_map = random.randint(0, 2, image.shape)
    noise_map = random.rand(*image.shape) < p
    output = image + noise_map * snp_map
    output = np.clip(output, 0., 1.)
    return output

def addPepperNoiseRef(image : np.ndarray, p : float) -> np.ndarray:
    """
    Parameters:
    - image : np.ndarray
        An image on which the pepper noise will be applied. It should be a np.ndarray
        with dtype=float64.
    - p : float [0 1]
        The p parameter in the pepper noise distribution. Should be in [0 1].
    Returns:
    - output : np.ndarray
        The noisy image, with dtype=np.float64, and values within [0 1].
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.float64, 'Parameter \'image\' should be an np.ndarray with dtype=np.float64.'
    assert (isinstance(p, float) or isinstance(p, int)) and p >= 0 and p <= 1, 'Parameter \'p\' should be a number within [0 1].'

    snp_map = random.randint(0, 2, image.shape) - 1
    noise_map = random.rand(*image.shape) < p
    output = image + noise_map * snp_map
    output = np.clip(output, 0., 1.)
    return output

def addUniformNoiseRef(image : np.ndarray, a : float, b : float) -> np.ndarray:
    """
    Parameters:
    - image : np.ndarray
        An image on which the Uniform noise will be applied. It should be a
        np.ndarray with dtype=float64.
    - a : float [-1 1]
        The a parameter in the Uniform noise distribution. Should be in [-1 1].
    - b : float [-1 1]
        The b parameter in the Uniform noise distribution. Should be in [-1 1].
    Returns:
    - output : np.ndarray
        The noisy image, with dtype=np.float64, and values within [0 1].
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.float64, 'Parameter \'image\' should be an np.ndarray with dtype=np.float64.'
    assert (isinstance(a, float) or isinstance(a, int)) and a >= -1 and a <= 1, 'Parameter \'a\' should be a number within [-1 1].'
    assert (isinstance(b, float) or isinstance(b, int)) and b >= -1 and b <= 1, 'Parameter \'b\' should be a number within [-1 1].'
    assert a <= b, 'Parameter \'a\' should be less than or equal to \'b\'.'
    
    output = image + random.uniform(a, b, size=image.shape)
    output = np.clip(output, 0., 1.)
    return output

def addGaussianNoiseRef(image : np.ndarray, sigma : float) -> np.ndarray:
    """
    Parameters:
    - image : np.ndarray
        An image on which the Gaussian noise will be applied. It should be a
        np.ndarray with dtype=float64.
    - sigma : float [0 inf)
        The sigma parameter in the Gaussian noise distribution. Should be in [0 inf).
    Returns:
    - output : np.ndarray
        The noisy image, with dtype=np.float64, and values within [0 1].
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.float64, 'Parameter \'image\' should be an np.ndarray with dtype=np.float64.'
    assert (isinstance(sigma, float) or isinstance(sigma, int)) and sigma >= 0, 'Parameter \'sigma\' should be a non-negative number.'

    output = image + random.normal(0, sigma, size=image.shape)
    output = np.clip(output, 0., 1.)
    return output

def addRayleighNoiseRef(image : np.ndarray, sigma : float) -> np.ndarray:
    """
    Parameters:
    - image : np.ndarray
        An image on which the Rayleigh noise will be applied. It should be a
        np.ndarray with dtype=float64.
    - sigma : float [0 inf)
        The sigma parameter in the Rayleigh noise distribution. Should be in [0 inf).
    Returns:
    - output : np.ndarray
        The noisy image, with dtype=np.float64, and values within [0 1].
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.float64, 'Parameter \'image\' should be an np.ndarray with dtype=np.float64.'
    assert (isinstance(sigma, float) or isinstance(sigma, int)) and sigma >= 0, 'Parameter \'sigma\' should be a non-negative number.'

    output = image + random.rayleigh(sigma, size=image.shape)
    output = np.clip(output, 0., 1.)
    return output

def addErlangNoiseRef(image : np.ndarray, k : int, beta : float) -> np.ndarray:
    """
    Parameters:
    - image : np.ndarray
        An image on which the Erlang noise will be applied. It should be a
        np.ndarray with dtype=float64.
    - k : int [1 inf)
        The k parameter in the Erlang noise distribution. Should be a positive
        integer.
    - beta : float (0 inf)
        The beta parameter in the Erlang noise distribution. Should be in (0 inf).
    Returns:
    - output : np.ndarray
        The noisy image, with dtype=np.float64, and values within [0 1].
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.float64, 'Parameter \'image\' should be an np.ndarray with dtype=np.float64.'    
    assert isinstance(k, int) and k > 0, 'Parameter \'k\' should be a positive integer.'
    assert (isinstance(beta, float) or isinstance(beta, int)) and beta > 0, 'Parameter \'sigma\' should be a positive number.'

    output = image + random.gamma(k, beta, size=image.shape)
    output = np.clip(output, 0., 1.)
    return output

def padRef(image : np.ndarray, padding : Union[int, Iterable[int]]) -> np.ndarray:
    """
    Parameters:
    - image : np.ndarray
        An image on which the padding will be applied. It should be a np.ndarray
        with dtype=float64.
    - padding : Union[int, Iterable[int]]
        The amount of padding that should be applied to the image. Either an
        Iterable of 4 integers, in which case each integers determines the
        number of padding pixels that are to be added to each side, or an integer,
        in which case the same amount will be applied to all four side of the
        image. All the values should be non-negative integers.
        The pattern for Iterable paddings is [up, down, left, right].
    Returns:
    - output : np.ndarray
        The padded image, with dtype=np.float64.
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.float64, 'Parameter \'image\' should be an np.ndarray with dtype=np.float64.'
    assert ((isinstance(padding, int) and padding >= 0 ) or (isinstance(padding, Iterable)) and len(padding) == 4 and \
           all((isinstance(p, int) and p >= 0) for p in padding)), 'Parameter \'padding\' should either be an integer, or an Iterable of 4 integers of all non-negative values.'

    h, w = image.shape
    
    if isinstance(padding, int):
        output = np.zeros((h + 2*padding, w + 2*padding), dtype=np.float64)
        output[padding:-padding,padding:-padding] = image
        output[:padding,padding:-padding] = image[:1,:]
        output[-padding:,padding:-padding] = image[-1:,:]
        output[padding:-padding,:padding] = image[:,:1]
        output[padding:-padding,-padding:] = image[:,-1:]
        output[:padding,:padding] = image[0,0]
        output[:padding,-padding:] = image[0,-1]
        output[-padding:,:padding] = image[-1,0]
        output[-padding:,-padding:] = image[-1,-1]
    else:
        pu, pd, pl, pr = padding
        output = np.zeros((h + pu + pd, w + pl + pr), dtype=np.float64)
        output[pu:-pd,pl:-pr] = image
        output[:pu,pl:-pr] = image[:1,:]
        output[-pd:,pl:-pr] = image[-1:,:]
        output[pu:-pd,:pl] = image[:,:1]
        output[pu:-pd,-pr:] = image[:,-1:]
        output[:pu,:pl] = image[0,0]
        output[:pu,-pr:] = image[0,-1]
        output[-pd:,:pl] = image[-1,0]
        output[-pd:,-pr:] = image[-1,-1]
    
    return output

def kernelViewRef(image : np.ndarray, kernel_shape : Iterable[int]):
    """
    Parameters:
    - image : np.ndarray
        An image whose sliding window view will be viewed. Should be an np.ndarray
        with dtype=np.float64.
    - kernel_shape : Iterable[int]
        An Iterable of two positive integers which determines the size of the kernel.
    Returns:
    - view : np.ndarray
        The sliding window view, which should be an np.ndarray of 4 dimensions
        with dtype=np.float64.
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.float64, 'Parameter \'image\' should be an np.ndarray with dtype=np.float64.'
    assert isinstance(kernel_shape, Iterable) and len(kernel_shape) == 2 and \
    all((isinstance(d, int) and d > 0) for d in kernel_shape), 'Parameter \'kernel_shape\' should be an Iterable of two positive integers'

    h, w = kernel_shape
    pu, pd, pl, pr = h // 2, (h-1) // 2, w // 2, (w-1) // 2
    image_padded = padRef(image, (pu, pd, pl, pr))
    view = np.lib.stride_tricks.sliding_window_view(image_padded, kernel_shape)
    return view

def applyKernelRef(view : np.ndarray, kernel : np.ndarray) -> np.ndarray:
    """
    Parameters:
    - view : np.ndarray
        The sliding window view, which should be an np.ndarray of 4 dimensions
        with dtype=np.float64.
    - kernel : np.ndarray
        A 2-D kernel, with dimensions equal to the last dimensions of view.
    Returns:
    - image : np.ndarray
        The output image of applying the kernel.
    """
    assert isinstance(view, np.ndarray) and view.ndim == 4 and view.dtype == np.float64, 'Parameter \'view\' should be a 4D np.ndarray with dtype=np.float64.'
    assert isinstance(kernel, np.ndarray) and kernel.ndim == 2 and kernel.dtype == np.float64, 'Parameter \'kernel\' should be a 2D np.ndarray with dtype=np.float64.'
    kernel = kernel[np.newaxis, np.newaxis, :, :]
    image = np.sum(view * kernel, axis=(-1, -2))
    return image

def geometricMeanFilterRef(image : np.ndarray, kernel_shape : Iterable[int]) -> np.ndarray:
    """
    Parameters:
    - image : np.ndarray
        An image on which the geometric mean filter will be applied. Should be an 
        np.ndarray with dtype=np.float64.
    - kernel_shape : Iterable[int]
        An Iterable of two positive integers which determines the size of the kernel.
    Returns:
    - output : np.ndarray
        The filtered image, which should be an 2-dimensional np.ndarray with
        dtype=np.float64.
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.float64, 'Parameter \'image\' should be an np.ndarray with dtype=np.float64.'
    assert isinstance(kernel_shape, Iterable) and len(kernel_shape) == 2 and \
    all((isinstance(d, int) and d > 0) for d in kernel_shape), 'Parameter \'kernel_shape\' should be an Iterable of two positive integers.'
    
    view = kernelViewRef(image, kernel_shape)
    output = view.reshape(*view.shape[:2], -1)
    eps = 1e-3
    output = np.prod(output + eps, axis=-1)**(1 / np.prod(np.array(kernel_shape)))
    return output
    
def medianFilterRef(image : np.ndarray, kernel_shape : Iterable[int]) -> np.ndarray:
    """
    Parameters:
    - image : np.ndarray
        An image on which the median filter will be applied. Should be an np.ndarray
        with dtype=np.float64.
    - kernel_shape : Iterable[int]
        An Iterable of two positive integers which determines the size of the kernel.
    Returns:
    - output : np.ndarray
        The filtered image, which should be an 2-dimensional np.ndarray with
        dtype=np.float64.
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.float64, 'Parameter \'image\' should be an np.ndarray with dtype=np.float64.'
    assert isinstance(kernel_shape, Iterable) and len(kernel_shape) == 2 and \
    all((isinstance(d, int) and d > 0) for d in kernel_shape), 'Parameter \'kernel_shape\' should be an Iterable of two positive integers.'
    
    view = kernelViewRef(image, kernel_shape)
    output = view.reshape(*view.shape[:2], -1)
    output = np.median(output, axis=-1)
    return output

def alphaTrimmedMeanFilterRef(image : np.ndarray, kernel_shape : Iterable[int], alpha : float) -> np.ndarray:
    """
    Parameters:
    - image : np.ndarray
        An image on which the alpha-trimmed mean filter will be applied. Should be an 
        np.ndarray with dtype=np.float64.
    - kernel_shape : Iterable[int]
        An Iterable of two positive integers which determines the size of the kernel.
    - alpha : float [0 1)
        The alpha parameter in the alpha-trimmed mean. Should be a non-negative number
        less than 1.
    Returns:
    - output : np.ndarray
        The filtered image, which should be an 2-dimensional np.ndarray with
        dtype=np.float64.
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.float64, 'Parameter \'image\' should be an np.ndarray with dtype=np.float64.'
    assert isinstance(kernel_shape, Iterable) and len(kernel_shape) == 2 and \
    all((isinstance(d, int) and d > 0) for d in kernel_shape), 'Parameter \'kernel_shape\' should be an Iterable of two positive integers.'
    assert (isinstance(alpha, float) or isinstance(alpha, int)) and alpha >= 0 and alpha < 1, 'Parameter \'alpha\' should be a number within [0 1).'
    
    view = kernelViewRef(image, kernel_shape)
    view = view.reshape(*view.shape[:2], -1)
    view = np.sort(view, axis=-1)
    npixel = view.shape[-1]
    l = int(npixel * alpha / 2)
    r = int(npixel * (1 - alpha / 2))
    if l == r:
        r += 1
    view = view[:,:,l:r]
    output = np.mean(view, axis=-1)
    return output

def midpointFilterRef(image : np.ndarray, kernel_shape : Iterable[int]) -> np.ndarray:
    """
    Parameters:
    - image : np.ndarray
        An image on which the midpoint filter will be applied. Should be an np.ndarray
        with dtype=np.float64.
    - kernel_shape : Iterable[int]
        An Iterable of two positive integers which determines the size of the kernel.
    Returns:
    - output : np.ndarray
        The filtered image, which should be an 2-dimensional np.ndarray with
        dtype=np.float64.
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.float64, 'Parameter \'image\' should be an np.ndarray with dtype=np.float64.'
    assert isinstance(kernel_shape, Iterable) and len(kernel_shape) == 2 and \
    all((isinstance(d, int) and d > 0) for d in kernel_shape), 'Parameter \'kernel_shape\' should be an Iterable of two positive integers.'
    
    view = kernelViewRef(image, kernel_shape)
    view = view.reshape(*view.shape[:2], -1)
    view = np.sort(view, axis=-1)
    output = (view[:,:,0] + view[:,:,-1]) / 2
    return output

def harmonicFilterRef(image : np.ndarray, kernel_shape : Iterable[int]) -> np.ndarray:
    """
    Parameters:
    - image : np.ndarray
        An image on which the harmonic filter will be applied. Should be an np.ndarray
        with dtype=np.float64.
    - kernel_shape : Iterable[int]
        An Iterable of two positive integers which determines the size of the kernel.
    Returns:
    - output : np.ndarray
        The filtered image, which should be an 2-dimensional np.ndarray with
        dtype=np.float64.
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.float64, 'Parameter \'image\' should be an np.ndarray with dtype=np.float64.'
    assert isinstance(kernel_shape, Iterable) and len(kernel_shape) == 2 and \
    all((isinstance(d, int) and d > 0) for d in kernel_shape), 'Parameter \'kernel_shape\' should be an Iterable of two positive integers.'
    
    view = kernelViewRef(image, kernel_shape)
    view = view.reshape(*view.shape[:2], -1)

    eps = 1e-15
    view = 1 / (view + eps)
    output = 1 / np.mean(view, axis=-1)
    output = np.clip(output, 0., 1.)
    return output

def contraharmonicFilterRef(image : np.ndarray, kernel_shape : Iterable[int], Q : float) -> np.ndarray:
    """
    Parameters:
    - image : np.ndarray
        An image on which the contraharmonic filter will be applied. Should be an
        np.ndarray with dtype=np.float64.
    - kernel_shape : Iterable[int]
        An Iterable of two positive integers which determines the size of the kernel.
    - Q : float
        The Q parameter in the contraharmonic filter.
    Returns:
    - output : np.ndarray
        The filtered image, which should be an 2-dimensional np.ndarray with
        dtype=np.float64.
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.float64, 'Parameter \'image\' should be an np.ndarray with dtype=np.float64.'
    assert isinstance(kernel_shape, Iterable) and len(kernel_shape) == 2 and \
    all((isinstance(d, int) and d > 0) for d in kernel_shape), 'Parameter \'kernel_shape\' should be an Iterable of two positive integers.'
    assert isinstance(Q, int) or isinstance(Q, float), 'Parameter \'Q\' should be a number.'

    eps = 1e-15
    view = kernelViewRef(image, kernel_shape)
    view = view.reshape(*view.shape[:2], -1) + eps
        
    output = np.sum(view**Q, axis=-1) / np.sum(view**(Q - 1), axis=-1)
    output = np.clip(output, 0., 1.)
    return output

def adaptiveFilterRef(image : np.ndarray, kernel_shape : Iterable[int], base_std : float) -> np.ndarray:
    """
    Parameters:
    - image : np.ndarray
        An image on which the adaptive filter will be applied. Should be an np.ndarray
        with dtype=np.float64.
    - kernel_shape : Iterable[int]
        An Iterable of two positive integers which determines the size of the kernel.
    - base_variance : float [0 inf)
        An estimation of the standard deviation for additive noise. Should be a
        non-negative number.
    Returns:
    - output : np.ndarray
        The filtered image, which should be an 2-dimensional np.ndarray with
        dtype=np.float64.
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.float64, 'Parameter \'image\' should be an np.ndarray with dtype=np.float64.'
    assert isinstance(kernel_shape, Iterable) and len(kernel_shape) == 2 and \
    all((isinstance(d, int) and d > 0) for d in kernel_shape), 'Parameter \'kernel_shape\' should be an Iterable of two positive integers.'
    assert (isinstance(base_std, int) or isinstance(base_std, float)) and base_std >= 0, 'Parameter \'base_std\' should be a non-negative number.'

    view = kernelViewRef(image, kernel_shape)
    view = view.reshape(*view.shape[:2], -1)
    eps = 1e-15
    std = np.std(view, axis=-1) + eps
    coeff = np.clip(base_std**2 / std**2, 0., 1.)
    mean = np.mean(view, axis=-1)
    output = image - coeff * (image - mean)
    return output

def bilateralFilterRef(image : np.ndarray, kernel_shape : Iterable[int], spatial_sigma : float, intensity_sigma : float) -> np.ndarray:
    """
    Parameters:
    - image : np.ndarray
        An image on which the bilateral filter will be applied. Should be an np.ndarray
        with dtype=np.float64.
    - kernel_shape : Iterable[int]
        An Iterable of two positive integers which determines the size of the kernel.
    - spatial_sigma : float [0 inf)
        The sigma value for the spatial smoothing fucntion. Should be a non-negative
        number.
    - intensity_sigma : float [0 inf)
        The sigma value for the intensity smoothing fucntion. Should be a non-negative
        number.
    Returns:
    - output : np.ndarray
        The filtered image, which should be an 2-dimensional np.ndarray with
        dtype=np.float64.
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.float64, 'Parameter \'image\' should be an np.ndarray with dtype=np.float64.'
    assert isinstance(kernel_shape, Iterable) and len(kernel_shape) == 2 and \
    all((isinstance(d, int) and d > 0) for d in kernel_shape), 'Parameter \'kernel_shape\' should be an Iterable of two positive integers.'
    assert (isinstance(spatial_sigma, int) or isinstance(spatial_sigma, float)) and spatial_sigma >= 0, 'Parameter \'spatial_sigma\' should be a non-negative number.'
    assert (isinstance(intensity_sigma, int) or isinstance(intensity_sigma, float)) and intensity_sigma >= 0, 'Parameter \'intensity_sigma\' should be a non-negative number.'

    view = kernelViewRef(image, kernel_shape)
    spatial_kernel = (cv.getGaussianKernel(kernel_shape[0], spatial_sigma) @ cv.getGaussianKernel(kernel_shape[1], spatial_sigma).T)[np.newaxis,np.newaxis,:,:]
    gaussian = lambda z, sigma : np.exp(- z**2 / sigma**2 / 2)
    intensity_function = gaussian(view - image[:,:,np.newaxis,np.newaxis], intensity_sigma)
    output = np.sum(spatial_kernel * intensity_function * view, axis=(-1, -2)) / np.sum(spatial_kernel * intensity_function, axis=(-1, -2))
    return output
    