# coding=utf-8

"""loads images to np.arrays"""

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
from src.tools import processing


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
        # noinspection PyBroadException
        try:
            img = img_to_array(load_img(f"{path}/{file}", target_size=pic_size)) / 127.5 - 1.
            temp_list.append(img)
        except Exception:
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
        path += '/'

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
        path += '/'

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
        _prepare_folder(os.path.join(current_path, 'test'))
        if gan:
            _prepare_folder(os.path.join(current_path, 'generated'))

        for letter in letters:
            _remove_trash(os.path.join(current_path, 'train', letter))
            _remove_trash(os.path.join(current_path, 'validation', letter))
            _remove_trash(os.path.join(current_path, 'test', letter))

            _prepare_folder(os.path.join(current_path, 'train', letter))
            _prepare_folder(os.path.join(current_path, 'validation', letter))
            _prepare_folder(os.path.join(current_path, 'test', letter))
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
    _prepare_folder(config.PATH_STATS_GAN)
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

    for letter in letters:
        """ Prepare list of letters which will be moved copied into folders """
        letters_all = processing.return_all_files(os.path.join(config.DATA_PATH, letter))
        letters_dict[letter]['max'] = random.sample(letters_all, int(config.DATASET_MAX + config.DATASET_MAX * 0.2))
        letters_dict[letter]['test'] = random.sample(letters_dict[letter]['max'], int(config.DATASET_MAX * 0.2))

        letters_dict[letter]['max'] = list(set(letters_dict[letter]['max']) - set(letters_dict[letter]['test']))
        letters_dict[letter]['half'] = random.sample(letters_dict[letter]['max'], int(config.DATASET_MAX / 2))
        letters_dict[letter]['ten_p'] = random.sample(letters_dict[letter]['half'], int(config.DATASET_MAX * 0.1))

        # COPY FILES TO GAN
        _copy_files(source_folder=os.path.join(config.DATA_PATH, letter, ''),
                    destination_folder=os.path.join(config.PATH_GAN_MAX + letter, ''),
                    files=letters_dict[letter]['max'],
                    letter=letter)
        _copy_files(source_folder=os.path.join(config.DATA_PATH, letter, ''),
                    destination_folder=os.path.join(config.PATH_GAN_HALF + letter, ''),
                    files=letters_dict[letter]['half'],
                    letter=letter)
        _copy_files(source_folder=os.path.join(config.DATA_PATH, letter, ''),
                    destination_folder=os.path.join(config.PATH_GAN_TEN_P + letter, ''),
                    files=letters_dict[letter]['ten_p'],
                    letter=letter)

    """ Split files into train, valid and test folder and copy it into these fold """
    split_to_train_valid(letters=letters,
                         letters_dict=letters_dict,
                         first_letter=0)

    # COPY FILES TO CLASS (+ GENERATED PHOTOS)
    copy_files_to_class_generated(letters=letters,
                                  letters_dict=letters_dict,
                                  first_letter=0)

    split_to_train_valid(letters=letters,
                         letters_dict=letters_dict,
                         first_letter=1)

    # COPY FILES TO CLASS (+ GENERATED PHOTOS)
    copy_files_to_class_generated(letters=letters,
                                  letters_dict=letters_dict,
                                  first_letter=1)


def copy_files_to_class_generated(letters, letters_dict, first_letter):
    """
    Funtion for moving files into generated and test folders in loop depending on option
    ('max', 'half' and 'ten percent')
    :param letters: letters on which we are doing operations
    :param letters_dict: letters dictionary
    :param first_letter: boolean value - tells if operation is on first or second letter from list
    :return:
    """
    for option in ('max', 'half', 'ten_p'):
        _copy_files(source_folder=os.path.join(config.DATA_PATH, letters[first_letter], ''),
                    destination_folder=os.path.join(config.PATH_CLASS_GAN_MAX, f"{letters[0]}_{letters[1]}",
                                                    'generated', letters[first_letter], ''),
                    files=letters_dict[letters[first_letter]][option],
                    letter=letters[1])
    _copy_files(source_folder=os.path.join(config.DATA_PATH, letters[first_letter], ''),
                destination_folder=os.path.join(config.PATH_CLASS_GAN_TEN_P, f"{letters[0]}_{letters[1]}",
                                                'test', letters[first_letter], ''),
                files=letters_dict[letters[first_letter]]['test'],
                letter=letters[first_letter])


def split_to_train_valid(letters, letters_dict, first_letter):
    """
    Funtion for splitting files into train, validation and test folders in loop depending on option
    ('max', 'half' and 'ten percent')
    :param letters: letters on which we are doing operations
    :param letters_dict: letters dictionary
    :param first_letter: boolean value - tells if operation is on first or second letter from list
    :return:
    """
    for option in ['max', 'half', 'ten_p']:
        train_validation_test_dividing(source_path=config.PATH_GAN_MAX + letters[first_letter],
                                       destination_path=os.path.join(config.PATH_CLASS_MAX,
                                                                     f"{letters[0]}_{letters[1]}"),
                                       files=letters_dict[letters[first_letter]][option],
                                       test_files=letters_dict[letters[first_letter]]['test'],
                                       letter=letters[first_letter],
                                       percentage=config.PROPORTION)


def train_validation_test_dividing(source_path, destination_path, files, test_files, letter, percentage):
    """
    Function for dividing list of letters into train and validation set.
    It also gets test_files list and moves it into test folder
    :param source_path: paths from which images should be copied
    :param destination_path: paths for which images should be copied
    :param files: list of images which should be splitted into training and validation set
    :param test_files: list of files which should be copied into test folder
    :param letter: letter on which we are doing operations
    :param percentage: on what proportion dataset should be splitted into training and validation set
    :return: fill train, validation and test folders
    """
    if len(files) == 0:
        files = [f for f in listdir(source_path) if isfile(join(source_path, f))]
    sample = random.sample(files, int(len(files) * percentage))
    train = [x for x in files if x in sample]
    valid = [x for x in files if x not in sample]

    _copy_files(source_folder=os.path.join(source_path, ''),
                destination_folder=os.path.join(destination_path, 'train', letter, ''),
                files=train,
                letter=letter)
    _copy_files(source_folder=os.path.join(source_path, ''),
                destination_folder=os.path.join(destination_path, 'validation', letter, ''),
                files=valid,
                letter=letter)
    _copy_files(source_folder=os.path.join(config.DATA_PATH, letter, ''),
                destination_folder=os.path.join(destination_path, 'test', letter, ''),
                files=test_files,
                letter=letter)


def train_validation_dividing(source_path, destination_path, files, letter, percentage):
    """
    Function for dividing list of letters into train and validation set.
    :param source_path: paths from which images should be copied
    :param destination_path: paths for which images should be copied
    :param files: list of images which should be splitted into training and validation set
    :param letter: letter on which we are doing operations
    :param percentage: on what proportion dataset should be splitted into training and validation set
    :return: fill train, validation and test folders
    """
    if len(files) == 0:
        files = [f for f in listdir(source_path) if isfile(join(source_path, f))]
    sample = random.sample(files, int(len(files) * percentage))
    train = [x for x in files if x in sample]
    valid = [x for x in files if x not in sample]

    _copy_files(source_folder=os.path.join(source_path, ''),
                destination_folder=os.path.join(destination_path, 'train', letter, ''),
                files=train,
                letter=letter)
    _copy_files(source_folder=os.path.join(source_path, ''),
                destination_folder=os.path.join(destination_path, 'validation', letter, ''),
                files=valid,
                letter=letter)


def _copy_files(source_folder, destination_folder, files, letter):
    """
    Copies files from one folder into another
    :param source_folder: source folder from which files should be copied
    :param destination_folder: folder for which files should be copied
    :param files: list of files on which we are doing operations
    :param letter: letter on which we are doing operations
    :return:
    """
    for image in files:
        shutil.copyfile(source_folder + image, destination_folder + image)
    print(f"{str(len(files))} letters for {letter} copied")


def _remove_trash(path):
    """
    Remove directory with files
    :param path: path to the directory
    """

    # noinspection PyBroadException
    try:
        shutil.rmtree(path)
        print(f"{path} folder deleted")
    except Exception:
        print(f"Nothing to delete ({path})")


def _prepare_folder(path):
    """
    Make directory
    :param path: path to the directory
    """

    # noinspection PyBroadException
    try:
        os.mkdir(path)
    except Exception:
        print(f"{path} directory already exists")
