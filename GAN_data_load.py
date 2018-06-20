"""loads images to np.arrays

todo: try https://keras.io/preprocessing/image/
"""

import os
from random import shuffle
import defaults

import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical


def load_images_into_array(path, pic_size=defaults.PIC_SIZE, sample_size=-1):
    """iterates over a directory and reads all images
    into an ndarray with dimensions (nfiles, width, height, 3)
    assumes all files in the given directory are images

    :return: ndarray - 4d (nfiles, width, height, 3)
    """
    files = os.listdir(path)
    if sample_size > len(files):
        print("sample_size > len(files)")
    elif sample_size > 0:
        shuffle(files)
        files = files[0:sample_size]
    return np.stack([img_to_array(load_img(f"{path}//{file}", target_size=pic_size)) for file in files], axis=0)


def load_sets(path=defaults.PATH,
              pic_size=defaults.PIC_SIZE,
              sample_size=(-1, -1),
              classes_to_read=('cats', 'dogs')):
    """reads train and valid pictures into keras friendly arrays
    assumes that each class has a sepearate directory with a proper name
    eg.
    C://magisterka_data//dogscats//train//dogs
    C://magisterka_data//dogscats//train//cats
    C://magisterka_data//dogscats//valid//dogs
    C://magisterka_data//dogscats//valid//cats

    :param pic_size:
    :param path:
    :param classes_to_read: should match names of proper directories
    :param sample_size: tuple -1 means all images
    :return: tuple of ndarrays (x_train, y_train, x_valid, y_valid), shuffled
    """

    if path[-1] != '\\':
        path = path + '\\'

    x_train = []
    x_valid = []
    y_train = []
    y_valid = []
    label = 0

    for c in classes_to_read:
        loaded = load_images_into_array(path=f"{path}train\\{c}", pic_size=pic_size, sample_size=sample_size[0])
        x_train.append(loaded)
        y_train += [label for i in range(loaded.shape[0])]

        loaded = load_images_into_array(path=f"{path}valid\\{c}", pic_size=pic_size, sample_size=sample_size[1])
        x_valid.append(loaded)
        y_valid += [label for i in range(loaded.shape[0])]

        label += 1

    x_train = np.concatenate(x_train, axis=0)
    x_valid = np.concatenate(x_valid, axis=0)

    y_train = np.array(y_train)
    y_valid = np.array(y_valid)

    shuffle_train = np.random.permutation(x_train.shape[0])
    shuffle_valid = np.random.permutation(x_valid.shape[0])

    x_train = x_train[shuffle_train, :, :, :]
    y_train = y_train[shuffle_train]

    x_valid = x_valid[shuffle_valid, :, :, :]
    y_valid = y_valid[shuffle_valid]

    y_train = to_categorical(y_train, num_classes=len(classes_to_read))
    y_valid = to_categorical(y_valid, num_classes=len(classes_to_read))

    return x_train, y_train, x_valid, y_valid


def load_all_pictures(path=defaults.PATH,
                      pic_size=defaults.PIC_SIZE,
                      sample_size=(-1, -1),
                      classes_to_read=('cats', 'dogs')):
    """
    loads all pictures from a given directory to one 4d ndarray

    :param path:
    :param pic_size:
    :param sample_size:
    :param classes_to_read:
    :return: ndarray (npictures, width, height, RGB)
    """
    if path[-1] != '\\':
        path = path + '\\'

    images = [load_images_into_array(path=f"{path}{v}\\{c}", pic_size=pic_size, sample_size=sample_size[0]) for c in
              classes_to_read for v in ('train', 'valid')]
    images = np.concatenate(images, axis=0)

    return images


def augment_sets(x_train, y_train, x_valid, y_valid):
    """work in progress"""
    # todo: implement https://keras.io/preprocessing/image/
    return x_train, y_train, x_valid, y_valid


# test cases ====================================================================================
a, b, c, d = load_sets(defaults.PATH, defaults.PIC_SIZE, sample_size=(-1, -1),
                       classes_to_read=('dogs', 'cats'))

print(a.shape == (23000, 20, 20, 3))
print(b.shape == (23000, 2))
print(c.shape == (2000, 20, 20, 3))
print(d.shape == (2000, 2))

a, b, c, d = load_sets(defaults.PATH, defaults.PIC_SIZE, sample_size=(-1, -1),
                       classes_to_read=['dogs'])

print(a.shape == (11500, 20, 20, 3))
print(b.shape == (11500, 1))
print(c.shape == (1000, 20, 20, 3))
print(d.shape == (1000, 1))

a, b, c, d = load_sets(defaults.PATH, defaults.PIC_SIZE, sample_size=(1000, 1000),
                       classes_to_read=('dogs', 'cats'))

print(a.shape == (2000, 20, 20, 3))
print(b.shape == (2000, 2))
print(c.shape == (2000, 20, 20, 3))
print(d.shape == (2000, 2))
