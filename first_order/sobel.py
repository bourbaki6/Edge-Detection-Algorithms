

import numpy as np

from filters import gradient

SOBEL = np.array([
    [-1, 0, 1], 
    [-2, 0, 2], 
    [-1, 0, 1]], float)


def sobel(img):
    return gradient(img, SOBEL)[2]