import numpy as np

def bgra2rgba(image : np.ndarray) -> np.ndarray:
    assert isinstance(image, np.ndarray) and image.dtype == np.float64 and image.ndim == 4, 'Parameter \'image\' should be an np.ndarray with ndim=4 and dtype=np.float64.'
    output = image
    output[:,:,:3] = np.flip(output[:,:,:,:3], axis=-1)
    return output