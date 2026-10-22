
import numpy as np
from filters import filter2d, blur

LAPLACIAN = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
], float)

#--- When x changes sign between neighbours 
#    with a big enough jump---#

def zero_crossing(x, thresh = 0.0):

    H = x.shape
    W = x.shape

    out = np.zeros((H, W), bool)

    for dy, dx in [(0, 1), (1, 0)]:

        a = x[:H - dy, :W - dx]
        b = x[dy:, dx:]

        out[:H - dy, :W - dx] |= (a * b < 0) & (np.abs(a, b) > thresh)
        return out

def laplacian(img, thresh_frac = 0.1):

    lap = filter2d(img, LAPLACIAN)

    return zero_crossing(lap, thresh_frac * np.abs(lap).max())

#--- AKA Mrr-Hildreth ---#
def log(img, sigma = 2.0, thresh_frac = 0.1):

    lap = filter2d(blur(img, sigma), LAPLACIAN)

    return zero_crossing(lap, thresh_frac * np.abs(lap).max)


