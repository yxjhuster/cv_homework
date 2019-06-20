import numpy as np
import scipy.ndimage
import skimage


def extract_deep_feature(x, vgg16_weights):
    feat = skimage.transform.resize(x, (224, 224))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    feat = (feat - mean) / std
    linear_count = 0
    for layer in vgg16_weights:
        if layer[0] == 'conv2d':
            feat = multichannel_conv2d(feat, layer[1], layer[2])
        elif layer[0] == 'relu':
            feat = relu(feat)
            if linear_count == 2:
                break
        elif layer[0] == 'maxpool2d':
            feat = max_pool2d(feat, layer[1])
        elif layer[0] == 'linear':
            if len(feat.shape) != 1:
                feat = np.swapaxes(feat, 0, 2)
                feat = np.swapaxes(feat, 1, 2)
            feat = feat.flatten()
            feat = linear(feat, layer[1], layer[2])
            linear_count += 1
        else:
            continue
    return feat


def multichannel_conv2d(x, weight, bias):
    h, w, d = x.shape
    output_dim = weight.shape[0]
    feat = np.zeros((h, w, output_dim))
    for i in range(output_dim):
        kernel = weight[i, :, :, :]
        out_feat = np.zeros((h, w))
        for j in range(d):
            out_feat += scipy.ndimage.convolve(x[:, :, j],
                                               kernel[j, ::-1, ::-1], mode='constant', cval=0)
        feat[:, :, i] = out_feat + bias[i]
    return feat


def relu(x):
    return np.where(x > 0, x, 0)


def max_pool2d(x, size):
    return skimage.measure.block_reduce(x, (size[0], size[1]), np.max)


def linear(x, W, b):
    return np.dot(W, x) + b
