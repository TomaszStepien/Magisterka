"""loads images to np.arrays

todo: try https://keras.io/preprocessing/image/
"""

import os
import random
import shutil
from collections import defaultdict
from os import listdir
from os.path import isfile, join
from random import shuffle

import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical

import config


def load_images_into_array(path, pic_size=config.PIC_SIZE, sample_size=-1):
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
            img = img_to_array(load_img(f"{path}/{file}", target_size=pic_size)) / 127.5 - 1.
            temp_list.append(img)
        except:
            print(f"problem with: {path}/{file}")
            pass

    return np.stack(temp_list, axis=0)


def load_sets(path=config.DATA_PATH,
              pic_size=config.PIC_SIZE,
              sample_size=(-1, -1),
              classes_to_read=config.CLASSES_TO_READ):
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

    if path[-1] != '/':
        path = path + '/'

    x_train = []
    x_valid = []
    y_train = []
    y_valid = []
    label = 0

    for c in classes_to_read:
        # loaded = load_images_into_array(path=f"{path}train\\{c}", pic_size=pic_size, sample_size=sample_size[0])
        [os.rename(f"{path}{c}/{f}", f"{path}{c}/" + f.replace('=', '')) for f in os.listdir(f"{path}{c}")]
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


def load_all_pictures(path=config.DATA_PATH,
                      pic_size=config.PIC_SIZE,
                      sample_size=(-1, -1),
                      classes_to_read=config.CLASSES_TO_READ):
    """
    loads all pictures from a given directory to one 4d ndarray

    :param path:
    :param pic_size:
    :param sample_size:
    :param classes_to_read:
    :return: ndarray (npictures, width, height, RGB)
    """
    if path[-1] != '/':
        path = path + '/'

    images = [load_images_into_array(path=f"{path}{v}/{c}", pic_size=pic_size, sample_size=sample_size[0]) for c in
              classes_to_read for v in ('train', 'valid')]
    images = np.concatenate(images, axis=0)

    return images


def augment_sets(x_train, y_train, x_valid, y_valid):
    """work in progress"""
    # todo: implement https://keras.io/preprocessing/image/
    return x_train, y_train, x_valid, y_valid


def _prepare_classification_folders(folders_list, letters, gan=False):
    for path in folders_list:
        current_path = os.path.join(path, f"{letters[0]}_{letters[1]}")
        _prepare_folder(path)
        _prepare_folder(current_path)
        _prepare_folder(os.path.join(current_path, 'train'))
        _prepare_folder(os.path.join(current_path, 'validation'))
        if gan:
            _prepare_folder(os.path.join(current_path, 'generated'))

        for letter in letters:
            _remove_trash(os.path.join(current_path, 'train', letter))
            _remove_trash(os.path.join(current_path, 'validation', letter))
            _prepare_folder(os.path.join(current_path, 'train', letter))
            _prepare_folder(os.path.join(current_path, 'validation', letter))
            if gan:
                _remove_trash(os.path.join(current_path, 'generated', letter))
                _prepare_folder(os.path.join(current_path, 'generated', letter))


def prepare_final_datasets(letters):
    """
    Remove old directories, create new one.
    Randomly choose pictures from large dataset and copy them to created folders
    """

    _prepare_folder(config.PATH_FINAL_DATA)
    _prepare_folder(config.PATH_ROOT)
    _prepare_folder(config.PATH_STATS)
    _prepare_folder(config.PATH_GAN_LETTERS)
    _prepare_folder(config.PATH_CLASS_LETTERS)

    _prepare_classification_folders(config.PATH_CLASS_MAX_HALF_TEN, letters)
    _prepare_classification_folders(config.PATH_CLASS_GAN, letters, gan=True)

    """ PREPARE SUBFOLDERS WITH LETTERS"""
    for letter in letters:
        # GAN FILES
        for path in config.PATH_GAN_MAX_HALF_TEN:
            _remove_trash(path + letter)
            _prepare_folder(path + letter)

    letters_dict = defaultdict(dict)
    first = True
    for letter in letters:
        # PREPARE LIST OF FILES
        letters_all = [f for f in listdir(os.path.join(config.DATA_PATH, letter)) if
                        isfile(join(os.path.join(config.DATA_PATH, letter), f))]
        letters_dict[letter]['max'] = random.sample(letters_all, config.DATASET_MAX)
        letters_dict[letter]['half'] = random.sample(letters_dict[letter]['max'], int(config.DATASET_MAX / 2))
        letters_dict[letter]['ten_p'] = random.sample(letters_dict[letter]['half'], int(config.DATASET_MAX * 0.1))

        # COPY FILES TO GAN
        _copy_files(os.path.join(config.DATA_PATH, letter, ''),
                    os.path.join(config.PATH_GAN_MAX + letter, ''),
                    letters_dict[letter]['max'], letter)
        _copy_files(os.path.join(config.DATA_PATH, letter, ''),
                    os.path.join(config.PATH_GAN_HALF + letter, ''),
                    letters_dict[letter]['half'], letter)
        _copy_files(os.path.join(config.DATA_PATH, letter, ''),
                    os.path.join(config.PATH_GAN_TEN_P + letter, ''),
                    letters_dict[letter]['ten_p'], letter)

    # COPY FILES TO CLASS
    train_validation_dividing(config.PATH_GAN_MAX + letters[0],
                              os.path.join(config.PATH_CLASS_MAX, f"{letters[0]}_{letters[1]}"),
                              letters_dict[letters[0]]['max'], letters[0], 0.7)
    train_validation_dividing(config.PATH_GAN_MAX + letters[0],
                              os.path.join(config.PATH_CLASS_HALF, f"{letters[0]}_{letters[1]}"),
                              letters_dict[letters[0]]['max'], letters[0], 0.7)
    train_validation_dividing(config.PATH_GAN_MAX + letters[0],
                              os.path.join(config.PATH_CLASS_TEN_P, f"{letters[0]}_{letters[1]}"),
                              letters_dict[letters[0]]['max'], letters[0], 0.7)

    # COPY FILES TO CLASS (+ GENERATED PHOTOS)
    _copy_files(os.path.join(config.DATA_PATH, letters[0], ''),
                os.path.join(config.PATH_CLASS_GAN_MAX, f"{letters[0]}_{letters[1]}", 'generated', letters[0], ''),
                letters_dict[letters[0]]['max'], letters[0])
    _copy_files(os.path.join(config.DATA_PATH, letters[0], ''),
                os.path.join(config.PATH_CLASS_GAN_HALF, f"{letters[0]}_{letters[1]}", 'generated', letters[0], ''),
                letters_dict[letters[0]]['max'], letters[0])
    _copy_files(os.path.join(config.DATA_PATH, letters[0], ''),
                os.path.join(config.PATH_CLASS_GAN_TEN_P, f"{letters[0]}_{letters[1]}", 'generated', letters[0], ''),
                letters_dict[letters[0]]['max'], letters[0])

    train_validation_dividing(config.PATH_GAN_MAX + letters[1],
                              os.path.join(config.PATH_CLASS_MAX, f"{letters[0]}_{letters[1]}"),
                              letters_dict[letters[1]]['max'], letters[1], 0.7)
    train_validation_dividing(config.PATH_GAN_MAX + letters[1],
                              os.path.join(config.PATH_CLASS_HALF, f"{letters[0]}_{letters[1]}"),
                              letters_dict[letters[1]]['half'], letters[1], 0.7)
    train_validation_dividing(config.PATH_GAN_MAX + letters[1],
                              os.path.join(config.PATH_CLASS_TEN_P, f"{letters[0]}_{letters[1]}"),
                              letters_dict[letters[1]]['ten_p'], letters[1], 0.7)

    # COPY FILES TO CLASS (+ GENERATED PHOTOS)
    _copy_files(os.path.join(config.DATA_PATH, letters[1], ''),
                os.path.join(config.PATH_CLASS_GAN_MAX, f"{letters[0]}_{letters[1]}", 'generated', letters[1], ''),
                letters_dict[letters[1]]['max'], letters[1])
    _copy_files(os.path.join(config.DATA_PATH, letters[1], ''),
                os.path.join(config.PATH_CLASS_GAN_HALF, f"{letters[0]}_{letters[1]}", 'generated', letters[1], ''),
                letters_dict[letters[1]]['half'], letters[1])
    _copy_files(os.path.join(config.DATA_PATH, letters[1], ''),
                os.path.join(config.PATH_CLASS_GAN_TEN_P, f"{letters[0]}_{letters[1]}", 'generated', letters[1], ''),
                letters_dict[letters[1]]['ten_p'], letters[1])


def train_validation_dividing(source_path, destination_path, files, letter, percentage):
    if len(files) == 0:
        files = [f for f in listdir(source_path) if isfile(join(source_path, f))]
    sample = random.sample(files, int(len(files) * percentage))
    train = [x for x in files if x in sample]
    valid = [x for x in files if x not in sample]

    _copy_files(os.path.join(source_path, ''),
                os.path.join(destination_path, 'train', letter, ''),
                train, letter)
    _copy_files(os.path.join(source_path, ''),
                os.path.join(destination_path, 'validation', letter, ''),
                valid, letter)


def _copy_files(source_folder, destination_folder, files, letter):
    for image in files:
        shutil.copyfile(source_folder + image, destination_folder + image)
    print(f"{str(len(files))} letters for {letter} copied")


def _remove_trash(path):
    """
    Remove directory with files
    :param path: path to the directory
    """
    try:
        shutil.rmtree(path)
        print(f"{path} folder deleted")
    except:
        print(f"Nothing to delete ({path})")


def _prepare_folder(path):
    """
    Make directory
    :param path: path to the directory
    """
    try:
        os.mkdir(path)
    except:
        print(f"{path} directory already exists")

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
