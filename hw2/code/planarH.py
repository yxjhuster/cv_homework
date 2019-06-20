import numpy as np
import cv2
from BRIEF import briefLite, briefMatch
import pdb


def computeH(p1, p2):
    '''
    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
            equation
    '''
    assert(p1.shape[1] == p2.shape[1])
    assert(p1.shape[0] == 2)
    #############################
    A = []
    for i in range(p1.shape[1]):
        A.append(np.array([p2[0, i], p2[1, i], 1, 0, 0, 0,
                           -p1[0, i] * p2[0, i], -p1[0, i] * p2[1, i], -p1[0, i]]))
        A.append(np.array([0, 0, 0, -p2[0, i], -p2[1, i], -1,
                           p1[1, i] * p2[0, i], p1[1, i] * p2[1, i], p1[1, i]]))

    A = np.array(A)
    u, s, vh = np.linalg.svd(A)
    H2to1 = vh[-1, :].reshape((3, 3))

    return H2to1


def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using
    RANSAC
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches - matrix specifying matches between these two sets of point locations
        nIter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    '''
    ###########################
    p1, p2 = [], []
    for items in matches:
        p1.append(locs1[items[0], :-1])
        p2.append(locs2[items[1], :-1])
    p1 = np.array(p1).T
    p2 = np.array(p2).T

    num_points = matches.shape[0]
    max_num_inlier_points = -1
    bestH = np.zeros((3, 3))

    for i in range(num_iter):
        try:
            indices = np.random.randint(low=0, high=num_points, size=4)
            points1 = p1[:, indices]
            points2 = p2[:, indices]
            H = computeH(points1, points2)

            homo_p2 = np.dot(H, np.vstack((p2, np.ones((1, num_points)))))
            homo_p2 /= homo_p2[-1, :]
            homo_p2 = homo_p2[:-1, :]

            dist = np.sum((homo_p2 - p1)**2, axis=0) ** 0.5

            num_inlier_points = dist[dist <= tol].shape[0]

            if(num_inlier_points > max_num_inlier_points):
                bestH = H
                max_num_inlier_points = num_inlier_points
        except IndexError:
            print('Bad selection for the matched pairs!')
    return bestH


if __name__ == '__main__':
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    bestH = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    pdb.set_trace()
