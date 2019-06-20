import numpy as np
import pandas
import threading
import queue
import imageio
import os
import time
import math
import visual_words
import skimage.io
import multiprocessing
import pdb


def get_feature_for_one_image(args):
    """

    To extract the feature for image and make pair of the feature and label

    [input]
    * image_path: the path of the input image
    * label: the label of the image
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    [output]
    * feature: numpy.ndarray of shape (K)
    * label: the label of the image

    """
    image_path, label, dictionary, SPM_layer_num = args
    feature = get_image_feature(
        image_path, dictionary, SPM_layer_num, dictionary.shape[0])
    return feature, label


def build_recognition_system(num_workers=2):
    print("build")
    '''

    Creates a trained recognition system by generating training features from all training images.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''
    # load train data
    image_names = pandas.read_csv('../data/image_name.csv')
    image_names = image_names.train_imagenames.tolist()
    labels = pandas.read_csv('../data/labels.csv')
    labels = labels.train_labels.tolist()
    dictionary = np.load('./dictionary.npy')
    SPM_layer_num = 2

    image_path = [os.path.join('../data', image_name)
                  for image_name in image_names]
    args = zip(image_path, labels, [dictionary for _ in image_path], [
               SPM_layer_num for _ in image_path])

    # feature = get_image_feature(
    #     image_path[0], dictionary, 2, dictionary.shape[0])
    # pdb.set_trace()

    # extract features for images
    pool = multiprocessing.Pool(processes=num_workers)
    results = pool.map(get_feature_for_one_image, args)
    features = np.array([result[0] for result in results])
    labels = np.array([result[1] for result in results])

    np.savez('trained_system.npz', features=features, labels=labels,
             dictionary=dictionary, SPM_layer_num=SPM_layer_num)


def evaluate_recognition_system(num_workers):
    print("eval")
    '''

    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''
    # load test data
    image_names = pandas.read_csv('../data/image_name_test.csv')
    test_image_names = image_names.test_imagenames.tolist()
    labels = pandas.read_csv('../data/labels_test.csv')
    test_labels = labels.test_labels.tolist()

    # load trained system
    trained_system = np.load("trained_system.npz")
    train_features = trained_system['features']
    train_labels = trained_system['labels']
    dictionary = trained_system['dictionary']
    SPM_layer_num = trained_system['SPM_layer_num']

    test_image_path = [os.path.join('../data', item)
                       for item in test_image_names]

    # feature = get_image_feature(
    #     test_image_path[0], dictionary, 2, dictionary.shape[0])
    # pdb.set_trace()

    # obtain the feature for test image
    pool = multiprocessing.Pool(processes=num_workers)
    args = zip(test_image_path, test_labels, [dictionary for _ in test_image_path],
               [SPM_layer_num for _ in test_image_path])
    test_result = pool.map(get_feature_for_one_image, args)
    test_features = [result[0] for result in test_result]
    test_labels = [result[1] for result in test_result]

    # calculate the confusion matrix
    class_num = max(len(set(test_labels)), len(set(train_labels)))
    conf = np.zeros((class_num, class_num))
    for i, feature in enumerate(test_features):
        if (i == 46):
            pdb.set_trace()
        sim = distance_to_set(feature, train_features)
        index = np.where(sim == np.max(sim))[0]
        index = index[0]
        predict_label = train_labels[index]
        true_label = test_labels[i]
        conf[true_label - 1, predict_label - 1] += 1
        print('{} test case, prediction: {}, label: {}'.format(
            i, predict_label, true_label))

    accuracy = np.diag(conf).sum() / conf.sum()

    return conf, accuracy


def get_image_feature(file_path, dictionary, layer_num, K):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * file_path: path of image file to read
    * dictionary: numpy.ndarray of shape (K,3F)
    * layer_num: number of spatial pyramid layers
    * K: number of clusters for the word maps

    [output]
    * feature: numpy.ndarray of shape (K)
    '''
    image = skimage.io.imread(file_path)
    image = image.astype('float') / 255
    wordmap = visual_words.get_visual_words(image, dictionary)
    feature = get_feature_from_wordmap_SPM(wordmap, layer_num, K)
    return feature


def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''
    return np.sum(np.minimum(word_hist, histograms), axis=1)


def get_feature_from_wordmap(wordmap, dict_size):
    '''
    Compute histogram of visual words.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * dict_size: dictionary size K
    [output]
    * hist: numpy.ndarray of shape (K)
    '''
    hist, _ = np.histogram(wordmap.flatten(), bins=np.arange(
        dict_size + 1), normed=True, density=True)
    return hist


def get_feature_from_wordmap_SPM(wordmap, layer_num, dict_size):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * layer_num: number of spatial pyramid layers
    * dict_size: dictionary size K

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''
    h, w = wordmap.shape
    hist_list = []
    # for finest layer
    cellNum = 2 ** layer_num
    sub_h, sub_w = h // cellNum, w // cellNum
    finest_hist = np.zeros((cellNum, cellNum, dict_size))
    for i in range(cellNum):
        for j in range(cellNum):
            sub_map = wordmap[i * sub_h: (i + 1)
                              * sub_h, j * sub_w: (j + 1) * sub_w]
            sub_hist = get_feature_from_wordmap(sub_map, dict_size)
            finest_hist[i, j, :] = sub_hist
    hist_list.append(finest_hist)
    # for other layers
    for i in range(layer_num - 1, - 1, -1):
        cellNum = 2 ** i
        hist_tmp = np.zeros((cellNum, cellNum, dict_size))
        for i in range(cellNum):
            for j in range(cellNum):
                last_layer = hist_list[-1]
                hist_tmp[i][j] = last_layer[2 * i][2 * j] + \
                    last_layer[2 * i + 1][2 * j] + \
                    last_layer[2 * i][2 * j + 1] + \
                    last_layer[2 * i + 1][2 * j + 1]
        hist_list.append(hist_tmp)

    # create output
    hist_all = []
    hist_list = hist_list[::-1]
    for i in range(len(hist_list)):
        tmp = hist_list[i]
        tmp = np.reshape(tmp, ((2 ** i)**2, dict_size))
        if(i == 0 or i == 1):
            for j in range(tmp.shape[0]):
                hist_all.append(tmp[j, :] * 2 ** (-2))
        else:
            for j in range(tmp.shape[0]):
                hist_all.append(tmp[j, :] * 2 ** (i - 2 - 1))
    hist_all = np.array(hist_all)
    hist_all = hist_all / np.sum(hist_all)
    return hist_all
