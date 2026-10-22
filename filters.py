

import numpy as np

def filter2d(img, k):

        kh = k.shape
        kw = k.shape

        ph = kh // 2
        pw = kw // 2

        p = np.pad(img, ((ph, ph), (pw, pw)), mode = "edge")

        H = img.shape
        W = img.shape

        out = np.zeros((H, W))

        for i in range(kh):
            for j in range(kw):
                out += k[i, j] * p[i : i + H, j: j + W]
        return out


def gaussian_kernel(sigma):

        r = max(1, int(3 * sigma))

        x = np.arange(-r, r + 1)

        g = np.exp(-x ** 2 / (2 * sigma ** 2))
        g /= g.sum()

        return np.outer(g, g)

def blur(img, sigma):
    return filter2d(img, gaussian_kernel(sigma))
    

def gradient(img, kx):
    gx = filter2d(img, kx)
    gy = filter2d(img, kx.T)

    mag = np.hypot(gx, gy)

    ang = np.degrees(np.arctan2(gy, gx))

    return gx, gy, mag, ang


def threshold(mag, frac):
    return mag >= frac * mag.max()

def blur(img, sigma):
    return filter2d(img, gaussian_kernel(sigma))
