import numpy as np
from scipy.interpolate import RectBivariateSpline


def LucasKanade(It, It1, rect, p0=np.zeros(2)):
    # Input:
    #   It: template image
    #   It1: Current image
    #   rect: Current position of the car
    #   (top left, bot right coordinates)
    #   p0: Initial movement vector [dp_x0, dp_y0]
    # Output:
    #   p: movement vector [dp_x, dp_y]

    #  Put your implementation here
    p = p0
    dp = np.array([np.inf, np.inf], dtype=np.float32)
    threshold = 1e-3
    x_min, y_min, x_max, y_max = rect[0], rect[1], rect[2], rect[3]

    interp_spline_It = RectBivariateSpline(
        np.arange(It.shape[0]), np.arange(It.shape[1]), It)
    interp_spline_It1 = RectBivariateSpline(
        np.arange(It1.shape[0]), np.arange(It1. shape[1]), It1)

    while np.sum(dp ** 2) >= threshold:
        warp_x1 = np.arange(x_min + p[0], x_max + p[0] + .5)
        warp_y1 = np.arange(y_min + p[1], y_max + p[1] + .5)
        warp_X1, warp_Y1 = np.meshgrid(warp_x1, warp_y1)
        interped_I1 = interp_spline_It1.ev(warp_Y1, warp_X1)

        warp_x = np.arange(x_min, x_max + .5)
        warp_y = np.arange(y_min, y_max + .5)
        warp_X, warp_Y = np.meshgrid(warp_x, warp_y)
        interped_I = interp_spline_It.ev(warp_Y, warp_X)

        # calculate gradient
        interped_gx = interp_spline_It1.ev(
            warp_Y1, warp_X1, dx=0, dy=1).flatten()
        interped_gy = interp_spline_It1.ev(
            warp_Y1, warp_X1, dx=1, dy=0).flatten()

        B = interped_I.flatten() - interped_I1.flatten()
        A = np.vstack([interped_gx, interped_gy])

        dp = np.dot(np.linalg.inv(np.dot(A, A.T)), np.dot(A, B))
        p += dp.flatten()
    return p
