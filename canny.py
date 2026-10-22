

import numpy as np
from filters import gradient
from first_order.sobel import SOBEL


def non_max_supression(mag, ang):

    H = mag.shape
    W = mag.shape