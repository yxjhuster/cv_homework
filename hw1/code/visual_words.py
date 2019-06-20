import numpy as np
import multiprocessing
import pandas
import scipy.ndimage
import skimage.color
import sklearn.cluster
import scipy.spatial.distance
import os


def extract_filter_responses(image):
    '''
    Extracts the filter responses for the given image.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    # adjust the image shape
    image_shape = image.shape
    if len(image_shape) == 2:
        image = np.tile(image[:, :, np.newaxis], 3)
    else:
        if image_shape[-1] == 4:
            image = image[:, :, :-1]

    # rgb to lab
    image = skimage.color.rgb2lab(image)

    # create scale list
    filter_scale = [1, 2, 4, 8, 8 * np.sqrt(2)]
    filter_list = [scipy.ndimage.gaussian_filter, scipy.ndimage.gaussian_laplace,
                   scipy.ndimage.gaussian_filter, scipy.ndimage.gaussian_filter]
    order_list = [[0, 0], [0, 0], [0, 1], [1, 0]]

    # filter response
    response_list = []

    for i, scale in enumerate(filter_scale):
        for f, s, o in zip(filter_list, filter_scale, order_list):
            response = rgbConv(image, f, s, o)
            response_list.append(response)

    for i in range(len(response_list)):
        if i == 0:
            filter_response = response_list[i]
        else:
            filter_response = np.concatenate(
                (filter_response, response_list[i]), axis=2)
    return filter_response


def rgbConv(image, filter, sigma, order=None):
    for i in range(3):
        if filter == scipy.ndimage.gaussian_filter:
            img = filter(image[:, :, i], sigma, order)
        else:
            img = filter(image[:, :, i], sigma)
        if i == 0:
            imgs = img[:, :, np.newaxis]
        else:
            imgs = np.concatenate((imgs, img[:, :, np.newaxis]), axis=2)
    return imgs


def get_visual_words(image, dictionary):
    '''
    Compute visual words mapping for the given image using the dictionary of visual words.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)

    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    filter_response = extract_filter_responses(image)
    h, w, c = filter_response.shape
    filter_response = np.reshape(filter_response, (h * w, -1))
    dists = scipy.spatial.distance.cdist(
        filter_response, dictionary, 'euclidean')
    wordmap = np.argmin(dists, axis=1).reshape(h, w)
    return wordmap


def compute_dictionary_one_image(args):
    '''
    Extracts random samples of the dictionary entries from an image.
    This is a function run by a subprocess.

    [input]
    * i: index of training image
    * alpha: number of random samples
    * image_path: path of image file
    * time_start: time stamp of start time

    [saved]
    * sampled_response: numpy.ndarray of shape (alpha,3F)
    '''
    img_num, alpha, path = args
    if os.path.exists('../tmp' + str(img_num) + ".npy"):
        print("File Already Exists!")
        return
    image = skimage.io.imread(path)
    image = image / 255.
    filter_response = extract_filter_responses(image)
    h, w, c = filter_response.shape

    idx_list = np.random.permutation(h * w)
    sampled_response = []
    filter_response = np.reshape(filter_response, (h * w, c))

    for i in range(alpha):
        idx = idx_list[i]
        sampled_response.append(filter_response[idx, :])
    tmp_dir = '../tmp'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    np.save(os.path.join(tmp_dir, str(img_num) + '.npy'), sampled_response)
    print("Worker {} Finished!".format(img_num))
    return


def compute_dictionary(num_workers=2):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * dictionary: numpy.ndarray of shape (K,3F)
    '''
    if os.path.exists("dictionary.npy"):
        print("Dictionary Already Exists!")
        return

    # train_data = np.load("../data/train_data.npz")
    # image_names = train_data.f.image_names
    # labels = train_data.f.labels
    image_names = pandas.read_csv('../data/image_name.csv')
    image_names = image_names.train_imagenames.tolist()
    labels = pandas.read_csv('../data/labels.csv')
    labels = labels.train_labels.tolist()

    image_paths = [os.path.join('../data', item) for item in image_names]

    pool = multiprocessing.Pool(num_workers)

    alpha = 150
    K = 200

    args = zip(range(len(image_paths)), [
               alpha for _ in image_paths], image_paths)

    pool.map(compute_dictionary_one_image, args)
    print("-" * 50)

    # collect all the responses
    filter_responses = np.array([])
    tmp_dir = '../tmp'
    for file in os.listdir(tmp_dir):
        sampled_response = np.load(os.path.join(tmp_dir, file))
        filter_responses = np.array(np.append(filter_responses, sampled_response, axis=0)
                                    if filter_responses.shape[0] != 0 else sampled_response)
    filter_responses = np.array(filter_responses)
    print("Number of Points: {} | Dimensions: {}".format(
        filter_responses.shape[0], filter_responses.shape[1]))
    print("Running K Means...")

    kmeans = sklearn.cluster.KMeans(
        n_clusters=K, n_jobs=4).fit(filter_responses)
    dictionary = kmeans.cluster_centers_

    np.save('dictionary.npy', dictionary)
