import numpy as np

def gaussianNoise(I : np.ndarray, sigma : float) -> np.ndarray:
    noise = np.float32(np.random.normal(0, sigma, size=I.shape))
    J = I + noise
    J = np.minimum(np.maximum(J, 0), 1)
    return J