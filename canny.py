

import numpy as np
from filters import gradient, blur
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

def dilate(m):

    H = m.shape
    W = m.shape

    p = np.zeros_like(m)

    for i in range(3):
        for j in range(3):
            out |= p[i : i + H, j : j + W]
    return out

def hysteresis(nms, low, high):

    strong = nms >= high
    weak = nms >= low
    edges = strong.copy()

    while True:
        grown = dilate(edges) & weak
        if (grown == edges).all():
            return edges
        edges = grown

def canny(img, sigma = 1.4, low_frac = 0, high_frac = 0.25):

    _, _, mag, ang = gradient(blur(img, sigma), SOBEL)
    nms = non_max_supression(mag, ang)
    m = nms.max()

    return hysteresis(nms, low_frac * m, high_frac * m)


