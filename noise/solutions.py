import numpy as np
from numpy import random

def addSNPNoiseRef(image : np.ndarray, p : float) -> np.ndarray:
    """
    Parameters:
    - image : np.ndarray
        An image on which the Salt & Pepper noise will be applied. It should be a
        np.ndarray with dtype=float32, and with values within [0 1]
    - p : float [0 1]
        The p parameter in the S&P noise distribution. Should be in [0 1].
    Returns:
    - output : np.ndarray
        The noisy image, with dtype=np.float32, and values within [0 1].
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.float32, 'Parameter \'image\' should be an np.ndarray with dtype=np.float32.'
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
        with dtype=float32, and with values within [0 1].
    - p : float [0 1]
        The p parameter in the salt noise distribution. Should be in [0 1].
    Returns:
    - output : np.ndarray
        The noisy image, with dtype=np.float32, and values within [0 1].
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.float32, 'Parameter \'image\' should be an np.ndarray with dtype=np.float32.'
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
        with dtype=float32, and with values within [0 1].
    - p : float [0 1]
        The p parameter in the pepper noise distribution. Should be in [0 1].
    Returns:
    - output : np.ndarray
        The noisy image, with dtype=np.float32, and values within [0 1].
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.float32, 'Parameter \'image\' should be an np.ndarray with dtype=np.float32.'
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
        np.ndarray with dtype=float32, and with values within [0 1].
    - a : float [-1 1]
        The a parameter in the Uniform noise distribution. Should be in [-1 1].
    - b : float [-1 1]
        The b parameter in the Uniform noise distribution. Should be in [-1 1].
    Returns:
    - output : np.ndarray
        The noisy image, with dtype=np.float32, and values within [0 1].
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.float32, 'Parameter \'image\' should be an np.ndarray with dtype=np.float32.'
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
        np.ndarray with dtype=float32, and with values within [0 1].
    - sigma : float [0 inf)
        The sigma parameter in the Gaussian noise distribution. Should be in [0 inf).
    Returns:
    - output : np.ndarray
        The noisy image, with dtype=np.float32, and values within [0 1].
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.float32, 'Parameter \'image\' should be an np.ndarray with dtype=np.float32.'
    assert (isinstance(sigma, float) or isinstance(sigma, int)) and sigma >= 0, 'Parameter \'sigma\' should be a non-negative number.'

    output = image + random.normal(0, sigma, size=image.shape)
    output = np.clip(output, 0., 1.)
    return output

def addRayleighNoiseRef(image : np.ndarray, sigma : float) -> np.ndarray:
    """
    Parameters:
    - image : np.ndarray
        An image on which the Rayleigh noise will be applied. It should be a
        np.ndarray with dtype=float32, and with values within [0 1].
    - sigma : float [0 inf)
        The sigma parameter in the Rayleigh noise distribution. Should be in [0 inf).
    Returns:
    - output : np.ndarray
        The noisy image, with dtype=np.float32, and values within [0 1].
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.float32, 'Parameter \'image\' should be an np.ndarray with dtype=np.float32.'
    assert (isinstance(sigma, float) or isinstance(sigma, int)) and sigma >= 0, 'Parameter \'sigma\' should be a non-negative number.'

    output = image + random.rayleigh(sigma, size=image.shape)
    output = np.clip(output, 0., 1.)
    return output

def addErlangNoiseRef(image : np.ndarray, k : int, beta : float) -> np.ndarray:
    """
    Parameters:
    - image : np.ndarray
        An image on which the Erlang noise will be applied. It should be a
        np.ndarray with dtype=float32, and with values within [0 1].
    - k : int [1 inf)
        The k parameter in the Erlang noise distribution. Should be a positive
        integer.
    - beta : float (0 inf)
        The beta parameter in the Erlang noise distribution. Should be in (0 inf).
    Returns:
    - output : np.ndarray
        The noisy image, with dtype=np.float32, and values within [0 1].
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.float32, 'Parameter \'image\' should be an np.ndarray with dtype=np.float32.'    
    assert isinstance(k, int) and k > 0, 'Parameter \'k\' should be a positive integer.'
    assert (isinstance(beta, float) or isinstance(beta, int)) and beta > 0, 'Parameter \'sigma\' should be a positive number.'

    output = image + random.gamma(k, beta, size=image.shape)
    output = np.clip(output, 0., 1.)
    return output
    