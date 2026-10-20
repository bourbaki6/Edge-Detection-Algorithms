

import numpy as np

from filters import gradient

def roberts(img):

    p = np.pad(img, ((0, 1), (0, 1)), mode = "edge")

    #--- Main Diag. Diff. ---#
    gx = p[:-1, :-1] - p[1:, 1:]
    
    #--- Anti Diag. Diff. ---#
    gy = p[:-1, 1:] - p[1:, :-1]

    return np.hypot(gx, gy)
    