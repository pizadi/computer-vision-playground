import numpy as np

def bgr2rgbRef(image : np.ndarray) -> np.ndarray:
    """
    Parameters:
    - image : np.ndarray
        A color image in np.ndarray format with dtype=np.float64.
    Returns:
    - output : np.ndarray
        The original image, with channels converted from BGR to RGB.
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.float64 and image.ndim == 3, 'Parameter \'image\' should be an np.ndarray with ndim=3 and dtype=np.float64.'
    output = np.flip(image, axis=-1)
    return output

def rgb2cmyRef(image : np.ndarray) -> np.ndarray:
    """
    Parameters:
    - image : np.ndarray
        A color image in np.ndarray format with dtype=np.uint8.
    Returns:
    - output : np.ndarray
        The original image, with channels converted from RGB to CMY.
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.float64 and image.ndim == 3, 'Parameter \'image\' should be an np.ndarray with ndim=3 and dtype=np.float64.'
    output = 1 - image
    return output

def rgb2cmykRef(image : np.ndarray) -> np.ndarray:
    """
    Parameters:
    - image : np.ndarray
        A color image in np.ndarray format with dtype=np.float64.
    Returns:
    - output : np.ndarray
        The original image, with channels converted from RGB to CMYK.
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.float64 and image.ndim == 3, 'Parameter \'image\' should be an np.ndarray with ndim=3 and dtype=np.float64.'
    image_cmy = 1 - image

    eps = 1e-15
    k = np.min(image_cmy, axis=-1, keepdims=True)
    cmy = (image_cmy - k) / (1 - k + eps)
    output = np.concatenate([cmy, k], axis=-1)
    return output

def rgb2hsiRef(image : np.ndarray) -> np.ndarray:
    """
    Parameters:
    - image : np.ndarray
        A color image in np.ndarray format with dtype=np.float64.
    Returns:
    - output : np.ndarray
        The original image, with channels converted from RGB to HSI.
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.float64 and image.ndim == 3, 'Parameter \'image\' should be an np.ndarray with ndim=3 and dtype=np.float64.'
    eps = 1e-15
    output = np.zeros(image.shape)

    costheta = (image[:,:,0] - 0.5 * image[:,:,1] - 0.5 * image[:,:,2]) / np.sqrt((image[:,:,0] - image[:,:,1])**2 + (image[:,:,0] - image[:,:,2]) * (image[:,:,1] - image[:,:,2]) + eps)
    theta = np.arccos(costheta) / 2 / np.pi

    h_mapping = np.float64(image[:,:,1] >= image[:,:,2])
    
    output[:,:,0] = h_mapping * theta + (1 - h_mapping) * (1 - theta)
    output[:,:,1] = 1 - np.min(image, axis=2) / np.mean(image, axis=2)
    output[:,:,2] = np.mean(image, axis=2)

    return output

def hsi2rgbRef(image : np.ndarray) -> np.ndarray:
    """
    Parameters:
    - image : np.ndarray
        A color image in np.ndarray format with dtype=np.float64.
    Returns:
    - output : np.ndarray
        The original image, with channels converted from HSI to RGB.
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.float64 and image.ndim == 3, 'Parameter \'image\' should be an np.ndarray with ndim=3 and dtype=np.float64.'
    eps = 1e-15
    output = np.zeros(image.shape)
    region = np.uint8(np.floor(image[:,:,0] * 3))

    hp = image[:,:,0] - region / 3

    c1 = image[:,:,2] * (1 - image[:,:,1])
    c2 = image[:,:,2] * (1 + image[:,:,1] * np.cos(hp * 2 * np.pi) / np.cos(hp * 2 * np.pi - np.pi / 3))
    c3 = 3 * image[:,:,2] - c1 - c2

    output[:,:,0] = (region == 0) * c2 + (region == 1) * c1 + (region == 2) * c3
    output[:,:,1] = (region == 0) * c3 + (region == 1) * c2 + (region == 2) * c1
    output[:,:,2] = (region == 0) * c1 + (region == 1) * c3 + (region == 2) * c2

    return output

    