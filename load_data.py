# coding=utf-8

"""loads images to np.arrays"""

import os
from random import shuffle

import numpy as np
from keras.preprocessing.image import img_to_array, load_img


def load_images_into_array(directory, pic_size=(28, 28), sample_size=-1):
    """

    :param directory: directory containing pictures
    :param pic_size:
    :param sample_size: integer - how many images should be loaded, if -1 then all pictures are loaded
    :return: np.Array with dims (sample_size, width, height, channels)
    """
    files = os.listdir(directory)
    if sample_size > len(files):
        print("sample_size > len(files)")
    elif sample_size > 0:
        shuffle(files)
        files = files[0:sample_size]
    temp_list = []
    for file in files:
        # noinspection PyBroadException
        try:
            img = img_to_array(load_img(f"{directory}/{file}", target_size=pic_size)) / 127.5 - 1.
            temp_list.append(img)
        except Exception:
            print(f"problem with: {directory}/{file}")

    return np.stack(temp_list, axis=0)


def load_train_valid_test_arrays(classes_dict, directory='dataset/'):
    """

    :param classes_dict: dictionary with classes and train, valid, test sample number eg.
    dictionary = {'A': (10, 10, 10),
                  'D': (10, 10, 10)
                 }

    :param directory: directory which contains all classes subfolders
    :return: tuple of 6 numpy arrays
    """

    x_train = []
    y_train = []

    x_valid = []
    y_valid = []

    x_test = []
    y_test = []

    i = 0  # class binary label

    for class_ in classes_dict:
        samples = load_images_into_array(directory + class_, sample_size=sum(classes_dict[class_]))

        x1, x2, x3 = np.vsplit(samples[np.random.permutation(samples.shape[0])],
                               (classes_dict[class_][0], classes_dict[class_][0] + classes_dict[class_][1]))

        x_train.append(x1)
        x_valid.append(x2)
        x_test.append(x3)

        y1 = np.array([i] * classes_dict[class_][0])
        y2 = np.array([i] * classes_dict[class_][1])
        y3 = np.array([i] * classes_dict[class_][2])

        y_train.append(y1)
        y_valid.append(y2)
        y_test.append(y3)

        i += 1

    x_train = np.concatenate(x_train, axis=0)
    x_valid = np.concatenate(x_valid, axis=0)
    x_test = np.concatenate(x_test, axis=0)

    y_train = np.concatenate(y_train, axis=0)
    y_valid = np.concatenate(y_valid, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    return x_train, y_train, x_valid, y_valid, x_test, y_test
