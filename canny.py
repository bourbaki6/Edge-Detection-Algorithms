

import numpy as np
from filters import gradient
from first_order.sobel import SOBEL


def non_max_supression(mag, ang):

    H = mag.shape
    W = mag.shape

    p = np.pad(mag, 1)

    shift = lambda dy, dx: p[1 + dy : 1 + dy + H, 1 + dx : 1 + dx + W]

    bins = ((ang + 22.5) // 45).astype(int) % 4
    offsets = {0: (0, 1), 1: (1, 1), 
               2: (1, 0), 3: (1, -1)}

    out = np.zeros_like(mag)

    for b, (dy, dx) in offsets.items():
        is_max = (mag >= shift(dy,dx)) & (mag >= shift(-dy, -dx))
        sel = (bins == b) & is_max
        out[sel] = mag[sel]
    return out
