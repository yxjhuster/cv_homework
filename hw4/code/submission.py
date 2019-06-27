"""
Homework4.
Replace 'pass' by your implementation.
"""

import numpy as np
import helper
import matplotlib.pyplot as plt
import scipy.optimize

import pdb
from IPython import embed


'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''


def eightpoint(pts1, pts2, M):
    # Normalization
    n = pts1.shape[0]
    scale_matrix = np.diag(np.array([1. / M, 1. / M, 1.], dtype=np.float32))
    x1, y1 = pts1[:, 0] / M, pts1[:, 1] / M
    x2, y2 = pts2[:, 0] / M, pts2[:, 1] / M
    # eight points
    A = np.vstack((x2 * x1, x2 * y1, x2, y2 * x1,
                   y2 * y1, y2, x1, y1, np.ones(n))).T
    _, _, v = np.linalg.svd(A)
    F = np.reshape(v.T[:, -1], (3, 3))
    # refine with local normalization
    F = helper.refineF(F, pts1 / M, pts2 / M)
    # make F to be rank 2
    u_f, s_f, v_f = np.linalg.svd(F)
    s_f[-1] = 0
    F = u_f.dot(np.diag(s_f).dot(v_f))
    # scale back
    F = np.dot(scale_matrix.T, np.dot(F, scale_matrix))
    return F


'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''


def sevenpoint(pts1, pts2, M):
    # Normalization
    n = pts1.shape[0]
    scale_matrix = np.diag(np.array([1. / M, 1. / M, 1.], dtype=np.float32))
    x1, y1 = pts1[:, 0] / M, pts1[:, 1] / M
    x2, y2 = pts2[:, 0] / M, pts2[:, 1] / M
    # seven points
    A = np.vstack((x2 * x1, x2 * y1, x2, y2 * x1,
                   y2 * y1, y2, x1, y1, np.ones(n))).T
    _, _, v = np.linalg.svd(A)
    F1 = np.reshape(v.T[:, -1], (3, 3))
    F2 = np.reshape(v.T[:, -2], (3, 3))

    # solve for alpha
    def eqn(alpha):
        F_p = alpha * F1 + (1 - alpha) * F2
        return np.linalg.det(F_p)

    # get the coefficients of the polynomial
    a0 = eqn(0)
    a1 = 2 * (eqn(1) - eqn(-1)) / 3 - (eqn(2) - eqn(-2)) / 12
    a2 = (eqn(1) + eqn(-1)) / 2 - a0
    a3 = (eqn(1) - eqn(-1)) / 2 - a1
    # solve for alpha
    alpha = np.roots([a3, a2, a1, a0])

    Farray = [a * F1 + (1 - a) * F2 for a in alpha]
    # refine F
    Farray = [helper.refineF(F, pts1 / M, pts2 / M) for F in Farray]
    # scale back
    Farray = [np.dot(scale_matrix.T, np.dot(F, scale_matrix)) for F in Farray]
    return Farray


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''


def essentialMatrix(F, K1, K2):
    return np.dot(K2.T, np.dot(F, K1))


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''


def triangulate(C1, pts1, C2, pts2):
    n = pts1.shape[0]

    def skew(x):
        return np.array([[0, -x[2], x[1]],
                         [x[2], 0, x[0]],
                         [x[1], x[0], 0]])

    p1 = np.hstack((pts1, np.ones((n, 1))))
    p2 = np.hstack((pts2, np.ones((n, 1))))
    P = []
    # 3d projection
    for idx in range(n):
        s1 = skew(p1[idx])
        s2 = skew(p2[idx])
        A = np.vstack((np.dot(s1, C1), (np.dot(s2, C2))))
        _, _, v = np.linalg.svd(A)
        x = v.T[:, -1]
        x = x / x[-1]
        P.append(x[:-1])
    P = np.array(P)
    # 2d projection
    P_prime = np.hstack((P, np.ones((n, 1))))
    error = 0
    for idx in range(n):
        projected_1 = np.dot(C1, P_prime[idx, :].T)
        projected_2 = np.dot(C2, P_prime[idx, :].T)
        projected_1 = projected_1[:-2] / projected_1[-1]
        projected_2 = projected_2[:-2] / projected_2[-1]
        error += np.sum((projected_1 - pts1[idx])
                        ** 2 + (projected_2 - pts2[idx])**2)
    return P, error


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''


def epipolarCorrespondence(im1, im2, F, x1, y1):
    x1, y1 = int(round(x1)), int(round(y1))

    # create gaussian weight matrix
    sigma = 3
    window_size = 9
    center = window_size // 2
    mask = np.ones((window_size, window_size)) * center
    mask = np.repeat(np.array([range(window_size)]),
                     window_size, axis=0) - mask
    mask = np.sqrt(mask**2 + np.transpose(mask)**2)
    weight = np.exp(-0.5 * (mask**2) / (sigma**2))
    weight /= np.sum(weight)

    if len(im1.shape) > 2:
        weight = np.repeat(np.expand_dims(
            weight, axis=2), im1.shape[-1], axis=2)

    # get the window
    w1 = im1[y1 - center:y1 + center + 1, x1 - center:x1 + center + 1]

    # compute epipolar line
    p = np.array([x1, y1, 1])
    l2 = np.dot(F, p.T)

    # search along epipolar line
    search_range = 40
    y = np.array(range(y1 - search_range, y1 + search_range))
    x = np.round(-(l2[1] * y + l2[2]) / l2[0]).astype(np.int)
    h, w, _ = im2.shape
    valid = (x >= center) & (x < w - center) & (y >= center) & (y < h - center)
    x, y = x[valid], y[valid]

    min_dist = None
    x2, y2 = None, None
    for i in range(len(x)):
        # get the patch around the pixel in image2
        w2 = im2[y[i] - center:y[i] + center +
                 1, x[i] - center:x[i] + center + 1]
        # calculate the distance
        dist = np.sum((w1 - w2)**2 * weight)
        if min_dist is None or dist < min_dist:
            min_dist = dist
            x2, y2 = x[i], y[i]

    return x2, y2


'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''


def ransacF(pts1, pts2, M):
    iterations = 500
    n = pts1.shape[0]
    threshold = 1
    max_inlier = 0
    F = None
    idx_list = np.arange(n)
    np.random.shuffle(idx_list)
    idx = 0
    for i in range(iterations):
        if idx + 7 >= n:
            np.random.shuffle(idx_list)
            idx = 0
        sample1 = pts1[idx_list[idx: idx + 7], :]
        sample2 = pts2[idx_list[idx: idx + 7], :]
        idx += 7
        F_tmp_list = sevenpoint(sample1, sample2, M)
        for F_tmp in F_tmp_list:
            # calculate the epipolar lines
            pts1_homo = np.vstack((np.transpose(pts1), np.ones((1, n))))
            l2s = np.dot(F_tmp, pts1_homo)
            l2s = l2s / np.sqrt(np.sum(l2s[:2, :]**2, axis=0))
            # calculate the deviation of pts2 away from the epiploar lines
            pts2_homo = np.vstack((np.transpose(pts2), np.ones((1, n))))
            deviate = abs(np.sum(pts2_homo * l2s, axis=0))

            # determine the inliners
            tmp_inliers = np.transpose(deviate < threshold)

            if tmp_inliers[tmp_inliers].shape[0] > max_inlier:
                max_inlier = tmp_inliers[tmp_inliers].shape[0]
                F = F_tmp
                inliers = tmp_inliers

    return F, inliers


'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''


def rodrigues(r):
    pass


'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''


def invRodrigues(R):
    pass


'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''


def rodriguesResidual(K1, M1, p1, K2, p2, x):
    pass


'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''


def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    pass


if __name__ == '__main__':
    M = 640

    data = np.load('../data/some_corresp.npz')
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')
    # Q 2.1
    pts1, pts2 = data['pts1'], data['pts2']
    F = eightpoint(pts1, pts2, M)
    # helper.displayEpipolarF(im1, im2, F1)
    # Q 2.2
    pts1_sub = np.array([[256, 270], [162, 152], [199, 127], [
                        147, 131], [381, 236], [193, 290], [157, 231]])
    pts2_sub = np.array([[257, 266], [161, 151], [197, 135], [
                        146, 133], [380, 215], [194, 284], [157, 211]])
    F2 = sevenpoint(pts1_sub, pts2_sub, M)
    # helper.displayEpipolarF(im1, im2, F2[1])
    # Q 3.1
    intrinsics = np.load('../data/intrinsics.npz')
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    E = essentialMatrix(F, K1, K2)
    # Q 3.2
    M1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    M2 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    triangulate(M1, pts1, M2, pts2)
    # Q 4.1
    # selected_pts1, selected_pts2 = helper.epipolarMatchGUI(im1, im2, F)
    # Q 5.1
    noise_data = np.load('../data/some_corresp_noisy.npz')
    pts1, pts2 = noise_data['pts1'], noise_data['pts2']
    F, inliers = ransacF(pts1, pts2, M)
    helper.displayEpipolarF(im1, im2, F)