"""loads images to np.arrays

todo: try https://keras.io/preprocessing/image/
todo: improve accuracy (better pic size, more layers, more epochs)

useful resources:
https://keras.io/layers/convolutional/
https://keras.io/getting-started/sequential-model-guide/
"""

import os

import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical


def load_images_into_array(path, pic_size=(20, 20)):
    """iterates over a directory and reads all images
    into an ndarray with dimensions (nfiles, width, height, 3)
    assumes all files in the given directory are images

    :return: ndarray - 4d (nfiles, width, height, 3)
    """
    files = os.listdir(path)
    return np.stack([img_to_array(load_img(f"{path}//{file}", target_size=pic_size)) for file in files], axis=0)


def load_sets(train_path, valid_path, pic_size=(20, 20)):
    """reads cats and dogs into keras friendly arrays

    :return: tuple of ndarrays (x_train, y_train, x_valid, y_valid)
    """
    if train_path[-1] != '\\':
        train_path = train_path + '\\'

    if valid_path[-1] != '\\':
        valid_path = valid_path + '\\'

    train_cats = load_images_into_array(path=f"{train_path}cats", pic_size=pic_size)
    train_doggos = load_images_into_array(path=f"{train_path}dogs", pic_size=pic_size)
    valid_cats = load_images_into_array(path=f"{valid_path}cats", pic_size=pic_size)
    valid_doggos = load_images_into_array(path=f"{valid_path}dogs", pic_size=pic_size)

    x_train = np.concatenate((train_cats, train_doggos), axis=0)
    x_valid = np.concatenate((valid_cats, valid_doggos), axis=0)

    train_labels = np.array([0 for i in range(train_cats.shape[0])] + [1 for j in range(train_doggos.shape[0])])
    valid_labels = np.array([0 for k in range(valid_cats.shape[0])] + [1 for l in range(valid_doggos.shape[0])])

    shuffle_train = np.random.permutation(x_train.shape[0])
    shuffle_valid = np.random.permutation(x_valid.shape[0])

    x_train = x_train[shuffle_train, :, :, :]
    train_labels = train_labels[shuffle_train]

    x_valid = x_valid[shuffle_valid, :, :, :]
    valid_labels = valid_labels[shuffle_valid]

    y_train = to_categorical(train_labels, num_classes=2)
    y_valid = to_categorical(valid_labels, num_classes=2)

    return x_train, y_train, x_valid, y_valid
