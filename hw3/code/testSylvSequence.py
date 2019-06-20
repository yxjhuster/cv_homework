import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import os
import cv2

from LucasKanade import LucasKanade
from LucasKanadeBasis import LucasKanadeBasis

# write your script here, we recommend the above libraries for making your animation
if __name__ == '__main__':
    result_dir = '../result/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    data = np.load('../data/sylvseq.npy')
    base_data = np.load('../data/sylvbases.npy')
    frame = data[:, :, 0]
    rect_list = []
    rect = np.array([101, 61, 155, 107])
    rect_list.append(rect)
    for i in range(1, data.shape[2]):
        next_frame = data[:, :, i]
        p = LucasKanadeBasis(frame, next_frame, rect, base_data)

        rect = [rect[0] + p[0], rect[1] + p[1], rect[2] + p[0], rect[3] + p[1]]
        rect_list.append(rect)

        # show the image
        tmp_img = np.zeros((next_frame.shape[0], next_frame.shape[1], 3))
        tmp_img[:, :, 0] = next_frame
        tmp_img[:, :, 1] = next_frame
        tmp_img[:, :, 2] = next_frame
        cv2.rectangle(tmp_img, (int(round(rect[0])), int(round(rect[1]))), (int(
            round(rect[2])), int(round(rect[3]))), color=(0, 255, 0), thickness=2)

        if i in [1, 100, 200, 300, 400]:
            cv2.imwrite(os.path.join(
                result_dir, 'q2-3_{}.jpg'.format(i)), tmp_img * 255)

        frame = next_frame

    rect_list = np.array(rect_list)
    np.save(os.path.join('csylvseqrects.npy'), rect_list)
