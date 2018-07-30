"""loads images to np.arrays

todo: try https://keras.io/preprocessing/image/
"""

import os
from random import shuffle

import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical

import defaults


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
    temp_list = []
    for file in files:
        try:
            img = img_to_array(load_img(f"{path}//{file}", target_size=pic_size)) / 127.5 - 1.
            temp_list.append(img)
        except:
            print(f"problem with: {path}\\{file}")
            pass

    return np.stack(temp_list, axis=0)


def load_sets(path=defaults.PATH,
              pic_size=defaults.PIC_SIZE,
              sample_size=(-1, -1),
              classes_to_read=defaults.CLASSES_TO_READ):
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
        # loaded = load_images_into_array(path=f"{path}train\\{c}", pic_size=pic_size, sample_size=sample_size[0])
        [os.rename(f"{path}{c}\\{f}", f"{path}{c}\\" + f.replace('=', '')) for f in os.listdir(f"{path}{c}")]
        loaded = load_images_into_array(path=f"{path}{c}", pic_size=pic_size, sample_size=sample_size[0])
        x_train.append(loaded)
        y_train += [label] * loaded.shape[0]

        # loaded = load_images_into_array(path=f"{path}valid\\{c}", pic_size=pic_size, sample_size=sample_size[1])
        # x_valid.append(loaded)
        # y_valid += [label for i in range(loaded.shape[0])]

        label += 1

    x_train, y_train = prepare_dataset(x=x_train, y=y_train, classes_to_read=classes_to_read)
    # x_valid, y_valid = prepare_dataset(x=x_valid, y=y_valid, classes_to_read=classes_to_read)

    return x_train, y_train, x_valid, y_valid


def prepare_dataset(x, y, classes_to_read):
    """divides datasets for x and y

        :param x:
        :param y:
        :param classes_to_read: number of classes
        :return: ready x and y for given type of dataset
        """
    x_dataset = np.concatenate(x, axis=0)
    y_dataset = np.array(y)
    shuffle_dataset = np.random.permutation(x_dataset.shape[0])
    x_dataset = np.array(x_dataset[shuffle_dataset, :, :])
    y_dataset = y_dataset[shuffle_dataset]
    y_dataset = to_categorical(y_dataset, num_classes=len(classes_to_read))
    x_dataset = np.array(x_dataset).astype(np.float32)

    return x_dataset, y_dataset


def load_all_pictures(path=defaults.PATH,
                      pic_size=defaults.PIC_SIZE,
                      sample_size=(-1, -1),
                      classes_to_read=defaults.CLASSES_TO_READ):
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
# a, b, c, d = load_sets(defaults.PATH, defaults.PIC_SIZE, sample_size=(-1, -1),
#                        classes_to_read=('dogs', 'cats'))
#
# print(a.shape == (23000, 20, 20, 3))
# print(b.shape == (23000, 2))
# print(c.shape == (2000, 20, 20, 3))
# print(d.shape == (2000, 2))
#
# a, b, c, d = load_sets(defaults.PATH, defaults.PIC_SIZE, sample_size=(-1, -1),
#                        classes_to_read=['dogs'])
#
# print(a.shape == (11500, 20, 20, 3))
# print(b.shape == (11500, 1))
# print(c.shape == (1000, 20, 20, 3))
# print(d.shape == (1000, 1))
#
# a, b, c, d = load_sets(defaults.PATH, defaults.PIC_SIZE, sample_size=(1000, 1000),
#                        classes_to_read=('dogs', 'cats'))
#
# print(a.shape == (2000, 20, 20, 3))
# print(b.shape == (2000, 2))
# print(c.shape == (2000, 20, 20, 3))
# print(d.shape == (2000, 2))
