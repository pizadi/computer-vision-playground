import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def drawHist(image : np.ndarray) -> None:
    assert isinstance(image, np.ndarray) and image.dtype == np.uint8 and image.ndim == 2, 'Parameter \'image\' should be an np.ndarray with ndim=2 and dtype=np.uint8.'
    h0, h1 = np.histogram(image, bins=256, range=(-.5, 255.5))

    bins = np.sum(h0 > 0)
    
    _ = plt.figure(figsize=(10, 5))
    _ = plt.plot(h1[:-1], h0, 'r-')
    plt.title(f'The sample image\'s histogram with {bins} non-zero bins.')

def noiseCompare() -> None:
    image = cv.imread('./data/barbara.bmp')
    image += np.uint8(np.random.randn(*image.shape) * 5)
    image_hsi = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    _ = plt.figure(figsize=(18, 12))
    _ = plt.subplot(2, 3, 1), plt.imshow(image[:,:,2], vmin=0, vmax=255), plt.axis('off'), plt.title('R Channel')
    _ = plt.subplot(2, 3, 2), plt.imshow(image[:,:,1], vmin=0, vmax=255), plt.axis('off'), plt.title('G Channel')
    _ = plt.subplot(2, 3, 3), plt.imshow(image[:,:,0], vmin=0, vmax=255), plt.axis('off'), plt.title('B Channel')
    _ = plt.subplot(2, 3, 4), plt.imshow(image_hsi[:,:,0], vmin=0, vmax=255), plt.axis('off'), plt.title('H Channel')
    _ = plt.subplot(2, 3, 5), plt.imshow(image_hsi[:,:,1], vmin=0, vmax=255), plt.axis('off'), plt.title('S Channel')
    _ = plt.subplot(2, 3, 6), plt.imshow(image_hsi[:,:,2], vmin=0, vmax=255), plt.axis('off'), plt.title('I Channel')
    