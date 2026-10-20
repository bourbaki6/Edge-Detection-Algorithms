

import numpy as np

from filters import gradient
 
PREWITT = np.array([
    [-1, 0, 1], 
    [-1, 0, 1], 
    [-1, 0, 1]],  float)



def prewitt(img):
    return gradient(img, PREWITT)[2]
