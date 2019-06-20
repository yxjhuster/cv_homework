import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite, briefMatch, plotMatches

import pdb


def blend_mask(im):
    """ This is to generate the warped masks for the input images
        for further image blending.

    Args:
      im1: input image1 in numpy.array with size [H, W, 3]
      im2: input image2 in numpy.array with size [H, W, 3]
      homography1: the homography to warp image1 onto the panorama
      homography2: the homography to warp image2 onto the panorame
      out_shape: the size of the final panarama, in format of (width, height)
    Returns:
      warp_mask1: The warped mask for im1, namely the weights for im1 to blend
      warp_mask2: The warped mask for im2, namely the weights for im2 to blend
    """
    H, W, _ = im.shape
    # create mask for im1, zero at the borders and 1 at the center of the image
    mask = np.zeros((H, W))
    mask[0, :] = 1
    mask[-1, :] = 1
    mask[:, 0] = 1
    mask[:, -1] = 1
    mask = distance_transform_edt(1 - mask)
    mask = mask / np.max(mask)

    return mask


def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix

    INPUT
        Warps img2 into img1 reference frame using the provided warpH() function
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    #######################################
    H1, W1, _ = im1.shape
    H2, W2, _ = im2.shape

    # # find boundary
    b_points = np.array(
        [[0, 0, 1], [W2 - 1, 0, 1], [0, H2 - 1, 1], [W2 - 1, H2 - 1, 1]])
    b_points_projected = np.dot(H2to1, b_points.T)
    b_points_projected /= b_points_projected[-1, :]

    # find area
    min_w, min_h = int(np.round(np.min(b_points_projected[0, :]))), int(
        np.round(np.min(b_points_projected[1, :])))
    max_w, max_h = int(np.round(np.max(b_points_projected[0, :]))), int(
        np.round(np.max(b_points_projected[1, :])))

    width, height = max(max_w - min(min_w, 0),
                        W1), max(max_h - min(min_h, 0), H1)
    warp_im2 = cv2.warpPerspective(im2, H2to1, (width, height))
    warp_im1 = cv2.warpPerspective(im1, np.eye(3), (width, height))

    cv2.imwrite('../results/warp_im2.jpg', warp_im2)

    mask1 = blend_mask(im1)
    mask2 = blend_mask(im2)

    mask1 = cv2.warpPerspective(mask1, np.eye(3), (width, height))
    mask2 = cv2.warpPerspective(mask2, H2to1, (width, height))

    sum_mask = mask1 + mask2
    mask1 /= sum_mask
    mask1 = np.tile(mask1[:, :, np.newaxis], (1, 1, 3))
    mask2 /= sum_mask
    mask2 = np.tile(mask2[:, :, np.newaxis], (1, 1, 3))

    Blends1 = mask1 * warp_im1
    Blends2 = mask2 * warp_im2

    pano_im = Blends1 + Blends2
    cv2.imwrite('../results/panorama.jpg', pano_im)
    return pano_im


def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix without cliping.
    '''
    ######################################
    H1, W1, _ = im1.shape
    H2, W2, _ = im2.shape

    # # find boundary
    b_points = np.array(
        [[0, 0, 1], [W2 - 1, 0, 1], [0, H2 - 1, 1], [W2 - 1, H2 - 1, 1]])
    b_points_projected = np.dot(H2to1, b_points.T)
    b_points_projected /= b_points_projected[-1, :]

    # find area
    min_w, min_h = int(np.round(np.min(b_points_projected[0, :]))), int(
        np.round(np.min(b_points_projected[1, :])))
    max_w, max_h = int(np.round(np.max(b_points_projected[0, :]))), int(
        np.round(np.max(b_points_projected[1, :])))

    width, height = max(max_w - min(min_w, 0),
                        W1), max(max_h - min(min_h, 0), H1)

    M = np.float32([[1, 0, max(-min_w, 0)], [0, 1, max(-min_h, 0)], [0, 0, 1]])
    warp_im2 = cv2.warpPerspective(im2, np.matmul(M, H2to1), (width, height))
    warp_im1 = cv2.warpPerspective(im1, M, (width, height))

    cv2.imwrite('../results/warp_im2_noclip.jpg', warp_im2)

    mask1 = blend_mask(im1)
    mask2 = blend_mask(im2)

    mask1 = cv2.warpPerspective(mask1, M, (width, height))
    mask2 = cv2.warpPerspective(mask2, np.matmul(M, H2to1), (width, height))

    sum_mask = mask1 + mask2
    mask1 /= sum_mask
    mask1 = np.tile(mask1[:, :, np.newaxis], (1, 1, 3))
    mask2 /= sum_mask
    mask2 = np.tile(mask2[:, :, np.newaxis], (1, 1, 3))

    Blends1 = mask1 * warp_im1
    Blends2 = mask2 * warp_im2

    pano_im = Blends1 + Blends2
    cv2.imwrite('../results/panorama_noclip.jpg', pano_im)
    return pano_im


def generatePanorama(im1, im2):
    """ This is to generate the panorama given im1 and im2 by detecting and
        matching keypoints, calculating homography with RANSAC.

    Args:
      im1: input image1 in numpy.array with size [H, W, 3]
      im2: input image2 in numpy.array with size [H, W, 3]
    Returns:
      im3: stitched panorama in numpy.array.
    """
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    im3 = imageStitching_noClip(im1, im2, H2to1)

    return im3


if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    # plotMatches(im1,im2,matches,locs1,locs2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    pano_im_clip = imageStitching(im1, im2, H2to1)
    # cv2.imwrite('../results/panoImg_clip.png', pano_im_clip)
    # cv2.imshow('panoramas', pano_im_clip)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    pano_im = imageStitching_noClip(im1, im2, H2to1)
    # print(H2to1)
    # np.save('../results/q6_1.npy', H2to1)
    # cv2.imwrite('../results/panoImg.png', pano_im)
    # cv2.imshow('panoramas', pano_im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    im3 = generatePanorama(im1, im2)
