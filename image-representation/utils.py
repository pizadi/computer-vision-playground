import numpy as np
from matplotlib import pyplot as plt

def drawHist(image : np.ndarray) -> None:
    assert isinstance(image, np.ndarray) and image.dtype == np.uint8 and image.ndim == 2, 'Parameter \'image\' should be an np.ndarray with ndim=2 and dtype=np.uint8.'
    h0, h1 = np.histogram(image, bins=256, range=(-.5, 255.5))

    bins = np.sum(h0 > 0)
    
    _ = plt.figure(figsize=(10, 5))
    _ = plt.plot(h1[:-1], h0, 'r-')
    plt.title(f'The sample image\'s histogram with {bins} non-zero bins.')