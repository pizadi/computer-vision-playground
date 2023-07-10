import numpy as np
from matplotlib import pyplot as plt
from typing import Iterable

def drawSNPPDF(resolution : int, p : float) -> None:
    assert isinstance(resolution, int) and resolution > 0, 'Parameter \'resolution\' should be a positive integer.'
    assert (isinstance(p, float) or isinstance(p, int)) and p >= 0 and p <= 1, 'Parameter \'p\' should be a number within [0 1].'
    x = np.linspace(-1.5, 1.5, 6 * resolution + 1)
    y = np.zeros(1 + 6 * resolution)
    y[3 * resolution] = 1 - p
    y[resolution] = y[-resolution-1] = p / 2
    plt.plot(x, y, 'b-')
    plt.axis((-1.5, 1.5, -.01, 1))
    plt.title(f'PDF for a Salt & Pepper Noise with p={p:.4f}\n*y-axis units are in delta')

def drawUniformPDF(resolution : int, a : float, b : float) -> None:
    assert isinstance(resolution, int) and resolution > 0, 'Parameter \'resolution\' should be a positive integer.'
    assert (isinstance(a, float) or isinstance(a, int)) and a >= -1 and a <= 1, 'Parameter \'a\' should be a number within [-1 1].'
    assert (isinstance(b, float) or isinstance(b, int)) and b >= -1 and b <= 1, 'Parameter \'b\' should be a number within [-1 1].'
    assert a <= b, 'Parameter \'a\' should be less than or equal to \'b\'.'

    x = np.linspace(-1.5, 1.5, 6 * resolution + 1)
    if b - a > 0:
        y = np.minimum(x > a, x < b)
        y = np.float32(y) / (b - a)
    else:
        y[3 * resolution] = 1   
    plt.plot(x, y, 'b-')
    _, _, _, ymax = plt.axis()
    plt.axis((-1.5, 1.5, -0.01, ymax))
    plt.title(f'PDF for a Uniform Noise with a={a:.4f} & b={b:.4f}')

def drawRayleighPDF(resolution : int, sigma : float) -> None:
    assert isinstance(resolution, int) and resolution > 0, 'Parameter \'resolution\' should be a positive integer.'
    assert (isinstance(sigma, float) or isinstance(sigma, int)) and sigma >= 0, 'Parameter \'sigma\' should be a non-negative number.'

    x = np.linspace(0, 1, 2 * resolution + 1)
    y = x * np.exp(- x**2 / sigma**2 / 2) / sigma**2

    e_x = np.array([sigma * np.sqrt(np.pi / 2)])
    e_y = e_x * np.exp(- e_x**2 / sigma**2 / 2) / sigma**2
    
    plt.plot(x, y, 'b-')
    plt.plot(e_x, e_y, 'r^')
    _, _, _, ymax = plt.axis()
    plt.axis((0, 1, -0.01, ymax))
    plt.title(f'PDF for a Rayleigh Noise with sigma={sigma:.4f}')

def drawErlangPDF(resolution : int, k : int, beta : float) -> None:
    assert isinstance(resolution, int) and resolution > 0, 'Parameter \'resolution\' should be a positive integer.'
    assert isinstance(k, int) and k > 0, 'Parameter \'k\' should be a positive integer.'
    assert (isinstance(beta, float) or isinstance(beta, int)) and beta > 0, 'Parameter \'sigma\' should be a positive number.'

    x = np.linspace(0, 1, 2 * resolution + 1)
    y = x**(k-1) * np.exp(-x / beta) / beta**k / np.math.factorial(k - 1)

    e_x = np.array([beta * k])
    e_y = e_x**(k-1) * np.exp(-e_x / beta) / beta**k / np.math.factorial(k - 1)
    
    plt.plot(x, y, 'b-')
    plt.plot(e_x, e_y, 'r^')
    _, _, _, ymax = plt.axis()
    plt.axis((0, 1, -0.01, ymax))
    plt.title(f'PDF for an Erlang Noise with k={k:d} and beta={beta:.4f}')

def drawHistogram(image) -> None:
    assert isinstance(image, np.ndarray) and image.ndim == 2 and image.dtype == np.float64 \
    and np.max(image) <= 1. and np.min(image) >= 0., 'Parameter \'image\' should be a valid np.ndarray image with dtype=np.float64.'

    hist, bins = np.histogram(image, 256, (0 - 1 / 510, 1 + 1 / 510))
    plt.figure(figsize=(14, 7))
    _ = plt.subplot(1, 2, 1), plt.bar(np.linspace(0, 1, 128), hist, 1/256), plt.axis((-.01, 1.01, 0, None)), plt.title('Histogram of the Noisy Region')
    _ = plt.subplot(1, 2, 2), plt.imshow(image, vmin=0, vmax=1), plt.title('The Noisy Region')

    return None

def drawCustomPDF(resolution : int, beta : float) -> None:
    assert isinstance(resolution, int) and resolution > 0, 'Parameter \'resolution\' should be a positive integer.'
    assert (isinstance(beta, float) or isinstance(beta, int)) and beta > 0, 'Parameter \'sigma\' should be a positive number.'

    x = np.linspace(-1, 1, 2 * resolution + 1)
    y = np.clip(1 - (x / beta)**2, 0., 1.) * .75 / beta 
    
    e_x = np.array([0])
    e_y = np.clip(1 - (e_x / beta)**2, 0., 1.) * .75 / beta 

        
    plt.plot(x, y, 'b-')
    plt.plot(e_x, e_y, 'r^')
    _, _, _, ymax = plt.axis()
    plt.axis((-1, 1, -0.01, ymax))
    plt.title(f'PDF for a Custom Parabolic Noise with beta={beta:.4f}')

    return None

def drawCustomCDF(resolution : int, beta : float) -> None:
    assert isinstance(resolution, int) and resolution > 0, 'Parameter \'resolution\' should be a positive integer.'
    assert (isinstance(beta, float) or isinstance(beta, int)) and beta > 0, 'Parameter \'sigma\' should be a positive number.'

    x = np.linspace(-1, 1, 2 * resolution + 1)
    y = (x - x**3 / 3 / beta**2 + 2 * beta / 3) * .75 / beta 
    y[x < -beta] = 0
    y[x > beta] = 1

        
    plt.plot(x, y, 'b-')
    _, _, _, ymax = plt.axis()
    plt.axis((-1, 1, -0.01, ymax))
    plt.title(f'CDF for a Custom Parabolic Noise with beta={beta:.4f}')

    return None

def generateCustomNoise(noise_shape : Iterable[int], beta : float) -> None:
    assert isinstance(noise_shape, Iterable) and len(noise_shape) == 2 and \
    all(isinstance(s, int) and s > 0 for s in noise_shape), 'Parameter \'noise_shape\' should be an Iterable of two positive integers.'
    assert (isinstance(beta, float) or isinstance(beta, int)) and beta > 0, 'Parameter \'sigma\' should be a positive number.'

    noise = np.random.rand(*noise_shape)
    x = np.linspace(-beta, beta, int(2 * beta * 255 + 1))
    y = (x - x**3 / 3 / beta**2 + 2 * beta / 3) * .75 / beta 
    noise = np.searchsorted(y, noise)
    noise = x[noise]
    
    hist, bins = np.histogram(noise, 511, (-1 - 1 / 510, 1 + 1 / 510))
    plt.figure(figsize=(14, 7))
    _ = plt.subplot(1, 2, 1), plt.bar(np.linspace(-1, 1, 511), hist, 1/128), plt.axis((-1.01, 1.01, 0, None)), plt.title('Histogram of the Noise')
    _ = plt.subplot(1, 2, 2), plt.imshow(noise, vmin=0, vmax=1), plt.title('Generated Noise')

    return None