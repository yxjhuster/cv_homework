import numpy as np
from scipy.interpolate import RectBivariateSpline

import pdb


def InverseCompositionAffine(It, It1):
    # Input:
    #   It: template image
    #   It1: Current image

    #  Output:
    #   M: the Affine warp matrix [2x3 numpy array]

    # put your implementation here
    th = 0.001
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p = M.flatten()
    dp = np.array([np.inf] * 6, dtype=np.float32)
    x_min, y_min, x_max, y_max = 0, 0, It.shape[1] - 1, It.shape[0] - 1

    interp_spline_It1 = RectBivariateSpline(
        np.arange(It1.shape[0]), np.arange(It1. shape[1]), It1)
    interp_spline_It = RectBivariateSpline(
        np.arange(It.shape[0]), np.arange(It.shape[1]), It)

    x = np.arange(x_min, x_max + 0.5)
    y = np.arange(y_min, y_max + 0.5)
    X, Y = np.meshgrid(x, y)

    interped_gx = interp_spline_It.ev(Y, X, dx=0, dy=1).flatten()
    interped_gy = interp_spline_It.ev(Y, X, dx=1, dy=0).flatten()

    x = X.flatten()
    y = Y.flatten()
    A = np.array([interped_gx * x,
                  interped_gx * y,
                  interped_gx,
                  interped_gy * x,
                  interped_gy * y,
                  interped_gy]).T

    while np.sum(dp**2) >= th:
        warp_X = p[0] * X + p[1] * Y + p[2]
        warp_Y = p[3] * X + p[4] * Y + p[5]

        # check is valid
        valid = (warp_X > x_min) & (warp_X <= x_max) & (
            warp_Y > y_min) & (warp_Y <= y_max)

        warp_X = warp_X[valid]
        warp_Y = warp_Y[valid]
        interped_I1 = interp_spline_It1.ev(warp_Y, warp_X)

        A_valid = A[valid.flatten()]
        B = interped_I1.flatten() - It[valid].flatten()

        dp = np.dot(np.linalg.pinv(np.dot(A_valid.T, A_valid)),
                    np.dot(A_valid.T, B))
        M = np.copy(p).reshape(2, 3)
        M = np.vstack((M, np.array([[0, 0, 1]])))
        dM = np.vstack((np.copy(dp).reshape(2, 3), np.array([[0, 0, 1]])))
        dM[0, 0] += 1
        dM[1, 1] += 1
        M = np.dot(M, np.linalg.pinv(dM))
        p = M[:2, :].flatten()
    return M
