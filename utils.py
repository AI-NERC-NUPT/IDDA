import tensorflow as tf
import numpy as np
from scipy.io import loadmat
from tensorflow.examples.tutorials.mnist import input_data


def return_svhn(path_train, path_test):
    svhn_train = loadmat(path_train)
    svhn_test = loadmat(path_test)
    svhn_train_im = svhn_train['X']
    svhn_train_im = svhn_train_im.transpose(3, 0, 1, 2)
    svhn_train_im = np.reshape(svhn_train_im, (svhn_train_im.shape[0], 32, 32, 3))
    svhn_label = dense_to_one_hot_svhn(svhn_train['y'])
    svhn_test_im = svhn_test['X']
    svhn_test_im = svhn_test_im.transpose(3, 0, 1, 2)
    svhn_label_test = dense_to_one_hot_svhn(svhn_test['y'])
    svhn_test_im = np.reshape(svhn_test_im, (svhn_test_im.shape[0], 32, 32, 3))

    return svhn_train_im, svhn_label, svhn_test_im, svhn_label_test

    #data_s_im, data_s_im_test, data_s_label, data_s_label_test = return_svhn(path_svhn_train, path_svhn_test)

def return_mnist(path_train, path_test):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    mnist_train = np.reshape(np.load(path_train), (55000, 32, 32, 1))
    mnist_train = np.reshape(mnist_train, (55000, 32, 32, 1))
    mnist_train = mnist_train.astype(np.float32)
    mnist_test = np.reshape(np.load(path_test), (10000, 32, 32, 1)).astype(
        np.float32)
    mnist_test = np.reshape(mnist_test, (10000, 32, 32, 1))
    mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
    mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)
    return mnist_train, mnist.train.labels, mnist_test, mnist.test.labels


def select_class(labels, data, num_class=10, per_class=10):
    classes = np.argmax(labels, axis=1)
    labeled = []
    train_label = []
    unlabels = []
    for i in xrange(num_class):
        class_list = np.array(np.where(classes == i))
        class_list = class_list[0]
        class_ind = labels[np.where(classes == i), :]
        rands = np.random.permutation(len(class_list))
        unlabels.append(class_list[rands[per_class:]])
        labeled.append(class_list[rands[:per_class]])
        label_i = np.zeros((per_class, num_class))
        label_i[:, i] = 1
        train_label.append(label_i)
    unlabel_ind = []
    label_ind = []
    for t in unlabels:
        for i in t:
            unlabel_ind.append(i)
    for t in labeled:
        for i in t:
            label_ind.append(i)
    unlabel_data = data[unlabel_ind, :, :, :]
    labeled_data = data[label_ind, :, :, :]
    train_label = np.array(train_label).reshape((num_class * per_class, num_class))
    return np.array(labeled_data), np.array(train_label), unlabel_data

def select_labelled_data(data, label, num_sel, num_class=10):
    start = num_sel * num_class * 5  # start should be smaller than data.shape[0]
    num_samples = data.shape[0]
    min_num_each_cls = num_sel // num_class
    remain_num_sel = num_sel - min_num_each_cls * num_class
    #count_signal_cls = np.zeros(num_calss, dtype=int)
    new_data = []
    new_label = []
    for cls in range(num_class):
        count_cls = 0
        for i in range(num_samples):
            if np.argmax(label[i,:])==cls:
                new_data.append(data[i,:])
                new_label.append(label[i,:])
                count_cls = count_cls + 1
            if (count_cls == min_num_each_cls):
                break
    if (remain_num_sel > 0):
        for i in range(start, start+remain_num_sel):
            new_data.append(data[i, :])
            new_label.append(label[i, :])

    np_new_label = np.array(new_label)
    print('np_new_label.shape=',np_new_label.shape)
    return np.array(new_data), np_new_label

def judge_func(data, pred1, pred2, upper=0.9, num_class=10):
    num = pred1.shape[0]
    new_ind = []
    new_data = []
    new_label = []

    for i in range(num):
        cand_data = data[i, :, :, :]
        label_data = np.zeros((1, num_class))
        ind1 = np.argmax(pred1[i, :])
        value1 = np.max(pred1[i, :])
        ind2 = np.argmax(pred2[i, :])
        value2 = np.max(pred2[i, :])
        if ind1 == ind2:
            if max(value1, value2) > upper:
                label_data[0, ind1] = 1
                new_label.append(label_data)
                new_data.append(cand_data)
                new_ind.append(i)
    return np.array(new_data), np.array(new_label)

def judge_idda_func(data, pred1, lower=0.12, num_class=10):   #upper=0.11
    num = pred1.shape[0]
    new_ind = []
    new_data = []
    new_label = []

    for i in range(num):
        cand_data = data[i, :, :, :]
        label_data = np.zeros((1, num_class))
        ind1 = np.argmax(pred1[i, :])
        entropy = - np.sum(pred1[i,:] * np.log(pred1[i,:] + 1e-10))

        if (entropy < lower):
            label_data[0, ind1] = 1
            new_label.append(label_data)
            new_data.append(cand_data)
            new_ind.append(i)
    np_new_label = np.array(new_label)
    print('np_new_label.shape=',np_new_label.shape)
    np_new_label = np_new_label.reshape((np_new_label.shape[0], np_new_label.shape[2]))  #squash dimension[1]=1
    return np.array(new_data), np_new_label, np.array(new_ind)

def judge_init_func(data, pred1, lower=0.12, num_sel=512,  num_class=10):   #upper=0.11
    num = pred1.shape[0]
    new_ind = []
    new_data = []
    new_label = []

    pred1_list = [pred1[i,:] for i in range(pred1.shape[0])]
    pred1_entropy = map(lambda x: -np.sum(x[:] * np.log(x[:] + 1e-10)), pred1_list)
    pred1_entropy_list = list(pred1_entropy)
    pred_entropy_low = []

    for i in range(num):
        cand_data = data[i, :, :, :]
        label_data = np.zeros((1, num_class))
        ind1 = np.argmax(pred1[i, :])
        #entropy = - np.sum(pred1[i,:] * np.log(pred1[i,:] + 1e-10))
        entropy = pred1_entropy_list[i]
        if (entropy < lower):
            label_data[0, ind1] = 1
            new_label.append(label_data)
            new_data.append(cand_data)
            new_ind.append(i)
            pred_entropy_low.append(entropy)

    num0 = len(new_data)
    sort_ind = sorted(range(num0), key = lambda i: pred_entropy_low[i])
    print('Number of Generated Labels', num0)
    if num0 >= num_sel:
        sort_ind_sel=[sort_ind[i] for i in range(num_sel)]

        out_data  = [new_data[i] for i in sort_ind_sel]
        out_label = [new_label[i] for i in sort_ind_sel]
        out_ind   = [new_ind[i] for i in sort_ind_sel]
    else:
        print('The number of picked samples is smaller than the required ')
        out_data  = new_data
        out_label = new_label
        out_ind   = new_ind

    np_out_label = np.array(out_label)
    print('np_out_label.shape=',np_out_label.shape)
    np_out_label = np_out_label.reshape((np_out_label.shape[0], np_out_label.shape[2]))  #squash dimension[1]=1
    return np.array(out_data), np_out_label, np.array(out_ind)

#--pick samples with strong domain-shift (> threshold)
def pick_samples(pred1, threshold=0.4, lower_or_upper=False):   #upper=0.11
    num = pred1.shape[0]
    new_ind = []

    if (lower_or_upper == False):
        upper = threshold
        for i in range(num):
            entropy = - np.sum(pred1[i,:] * np.log(pred1[i,:] + 1e-10))
            if (entropy > upper):
                new_ind.append(i)
    else:
        lower = threshold
        for i in range(num):
            entropy = - np.sum(pred1[i,:] * np.log(pred1[i,:] + 1e-10))
            if (entropy < lower):
                new_ind.append(i)

    return new_ind

def weight_variable(shape, stddev=0.1, name=None, train=True):
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name:
        return tf.Variable(initial, name=name, trainable=train)
    else:
        return tf.Variable(initial)


def bias_variable(shape, init=0.1, name=None):
    initial = tf.constant(init, shape=shape)
    if name:
        return tf.Variable(initial, name=name)
    else:
        return tf.Variable(initial)


def batch_norm_conv(x, out_channels):
    mean, var = tf.nn.moments(x, axes=[0, 1, 2])
    beta = tf.Variable(tf.zeros([out_channels]), name="beta")
    gamma = weight_variable([out_channels])
    batch_norm = tf.nn.batch_norm_with_global_normalization(x, mean, var, beta, gamma, 0.001,
                                                            scale_after_normalization=True)
    return batch_norm


def batch_norm_fc(x, out_channels):
    mean, var = tf.nn.moments(x, axes=[0])
    beta = tf.Variable(tf.zeros([out_channels]), name="beta")
    gamma = weight_variable([out_channels])
    batch_norm = tf.nn.batch_norm_with_global_normalization(x, mean, var, beta, gamma, 0.001,
                                                            scale_after_normalization=True)
    return batch_norm


def conv2d(x, W, padding='SAME'):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]


def batch_generator(data, batch_size, shuffle=True, test=False):
    if shuffle:
        data = shuffle_aligned_list(data)
    batch_count = 0
    while True:
        if test:
            if batch_count * batch_size >= len(data[0]):
                batch_count = 0
                if shuffle:
                    data = shuffle_aligned_list(data)
        else:
            if batch_count * batch_size + batch_size >= len(data[0]):
                batch_count = 0
                if shuffle:
                    data = shuffle_aligned_list(data)
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]

def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    labels_one_hot = np.zeros((len(labels_dense), num_classes))
    labels_dense = list(labels_dense)
    for i, t in enumerate(labels_dense):
        labels_one_hot[i, t] = 1
    return labels_one_hot


def dense_to_one_hot_svhn(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    labels_one_hot = np.zeros((len(labels_dense), num_classes))
    labels_dense = list(labels_dense)
    for i, t in enumerate(labels_dense):
        if t == 10:
            t = 0
        labels_one_hot[i, t] = 1
    return labels_one_hot
