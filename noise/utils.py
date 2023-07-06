import numpy as np
from matplotlib import pyplot as plt

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
    