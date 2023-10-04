import numpy as np
import cv2 as cv
import bitarray as bt
from collections import OrderedDict
from typing import Iterable, Tuple, Dict, List
from bitarray.util import canonical_huffman, canonical_decode

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
    output[:,:,1] = 1 - np.min(image, axis=2) / (np.mean(image, axis=2) + eps)
    output[:,:,2] = np.mean(image, axis=2)

    output = np.clip(output, 0, 1)

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
    c2 = image[:,:,2] * (1 + (image[:,:,1] * np.cos(hp * 2 * np.pi) + eps) / (np.cos(hp * 2 * np.pi - np.pi / 3)) + eps)
    c3 = 3 * image[:,:,2] - c1 - c2

    output[:,:,0] = (region == 0) * c2 + (region == 1) * c1 + (region == 2) * c3
    output[:,:,1] = (region == 0) * c3 + (region == 1) * c2 + (region == 2) * c1
    output[:,:,2] = (region == 0) * c1 + (region == 1) * c3 + (region == 2) * c2

    output = np.clip(output, 0, 1)

    return output

def denoiseHSI(image : np.ndarray) -> np.ndarray:
    """
    Parameters:
    - image : np.ndarray
        A noisy color image with dtype=np.float64.
    Returns:
    - output : np.ndarray
        The original image, denoised in the HSI space.
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.float64 and image.ndim == 3, 'Parameter \'image\' should be an np.ndarray with ndim=3 and dtype=np.float64.'

    h, w, _ = image.shape
    kernel_size = 5, 5

    image_hsi = rgb2hsiRef(image)
    image_h_padded = np.zeros((h+4, w+4), dtype=np.float64)
    image_h_padded[2:-2,2:-2] = image_hsi[:,:,0]
    
    view = np.lib.stride_tricks.sliding_window_view(image_h_padded, kernel_size).reshape(h, w, 25) * 2 * np.pi
    
    view_i = np.mean(np.sin(view), axis=2)
    view_j = np.mean(np.cos(view), axis=2)
    new_view = np.arctan2(view_i, view_j) / 2 / np.pi % 1.
    
    image_hsi[:,:,0] = new_view
    output = hsi2rgbRef(image_hsi)
    return output
    

def baboonCompress(image : np.ndarray) -> bt.bitarray:
    """
    Parameters:
    - image : np.ndarray
        A specific greyscale image of a baboon, in which only 129
        of the 256 possible pixel values are used.
    Returns:
    - output : bt.bitarray
        The encoded version of the baboon's image, in which each
        pixel is represented by about 7 bits of data.
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.uint8 and image.ndim == 2, 'Parameter \'image\' should be an np.ndarray with ndim=2 and dtype=np.uint8.'
    assert image.shape == (480, 500), 'Parameter \'image\' should have dimensions of 480x500.'
    histogram = np.histogram(image,  bins=256, range=(-.5, 255.5))[0]
    used_bins = histogram > 0
    assert np.sum(used_bins) == 129, 'Parameter \'image\' should only have 129 intensity values.'
    
    dict = {i : histogram[i] for i in range(256)}
    sorted_vals = sorted(dict.items(), key = lambda x : x[1])
    sorted_vals.reverse()
    encoding_dict = {}
    for i, (val, hist) in enumerate(sorted_vals[:127]):
        encoding_dict[val] = bt.bitarray(f'{i:07b}')
    encoding_dict[sorted_vals[127][0]] = bt.bitarray(f'{254:b}')
    encoding_dict[sorted_vals[128][0]] = bt.bitarray(f'{255:b}')

    output = bt.bitarray()

    for pixel in image.flatten():
        output += encoding_dict[pixel]

    return output

def diffCompressRef(image : np.ndarray) -> Tuple[bt.bitarray, int, Tuple[Dict, List, List]]:
    """
    Parameters:
    - image : np.ndarray
        A greyscale image, with dtype=np.uint8.
    Returns:
    - output : bt.bitarray
        The compressed version of the image, using Huffman coding
        and calculating the difference map.
    - height : int
        The height of the image, used to extract the rightmost
        column.
    - huffman : Tuple[Dict, List, List]
        The output of the canonical_huffman function.
    """
    assert isinstance(image, np.ndarray) and image.dtype == np.uint8 and image.ndim == 2, 'Parameter \'image\' should be an np.ndarray with ndim=2 and dtype=np.uint8.'
    
    diffs = image[:,1:] - image[:,:-1]
    frequencies = np.histogram(diffs,  bins=256, range=(-.5, 255.5))[0]

    huffman_dict, count, symbol_canonical = canonical_huffman({i : frequencies[i] for i in range(256)})
    
    output = bt.bitarray()
    
    for pixel in image[:,0]:
        output += bt.bitarray(f'{pixel:08b}')
        
    for pixel in diffs.flatten():
        output += huffman_dict[pixel]

    return output, image.shape[0], (huffman_dict, count, symbol_canonical)