"""contains default values for parameters,
also creates repository structure
# https://www.kaggle.com/volperosso/simple-cnn-classifier-on-notmnist/data
"""

import os
import sys

# create folder structure
if sys.platform.startswith('linux'):
    home = "/home/tomasz/magisterka_data/"
else:
    home = "C:/magisterka_data/"

DATA_PATH = home + "notMNIST_small"
DATA_PATH = home + 'notMNIST_large'

SAVED_FILES = home + 'saved_files/'
SAVED_IMAGES = home + 'saved_files/saved_images/'
SAVED_MODELS = home + 'saved_files/saved_models/'
CLASS_DATA_PATH = home + 'notMNIST_training/'
STATS_FILES = home + 'saved_files/stats_files/'
IMAGE_PATH = os.path.join(home, 'notMNIST_large')

LETTERS = [['A', 'D'], ['G', 'C'], ['I', 'J']]
DATASET_MAX = 1000
DATASETS_LIST = [f"{str(DATASET_MAX)}_{str(DATASET_MAX)}",
                 f"{str(DATASET_MAX)}_{str(int(DATASET_MAX/2))}",
                 f"{str(DATASET_MAX)}_{str(int(DATASET_MAX*0.1))}"]

DATASETS_ROOT_LIST = ['A_D', 'G_C', 'I_J']

PATH_FINAL_DATA = os.path.join(home, 'MASTER_DATA')
PATH_ROOT = os.path.join(PATH_FINAL_DATA, 'root')
PATH_GAN_LETTERS = os.path.join(PATH_ROOT, 'GAN')
PATH_CLASS_LETTERS = os.path.join(PATH_ROOT, 'CLASS')

# CLASSIFICATION FOLDERS
PATH_CLASS_MAX = os.path.join(PATH_CLASS_LETTERS, DATASETS_LIST[0], '')
PATH_CLASS_HALF = os.path.join(PATH_CLASS_LETTERS, DATASETS_LIST[1], '')
PATH_CLASS_TEN_P = os.path.join(PATH_CLASS_LETTERS, DATASETS_LIST[2], '')
PATH_CLASS_MAX_HALF_TEN = [PATH_CLASS_MAX, PATH_CLASS_HALF, PATH_CLASS_TEN_P]

# CLASSIFICATION + GAN FOLDERS
PATH_CLASS_GAN_MAX = os.path.join(PATH_CLASS_LETTERS, f"{DATASETS_LIST[0]}_GAN", '')
PATH_CLASS_GAN_HALF = os.path.join(PATH_CLASS_LETTERS, f"{DATASETS_LIST[1]}_GAN", '')
PATH_CLASS_GAN_TEN_P = os.path.join(PATH_CLASS_LETTERS, f"{DATASETS_LIST[2]}_GAN", '')
PATH_CLASS_GAN = [PATH_CLASS_GAN_MAX, PATH_CLASS_GAN_HALF, PATH_CLASS_GAN_TEN_P]

# GAN FILES FOLDERS
PATH_GAN_MAX = os.path.join(PATH_GAN_LETTERS, f"{str(DATASET_MAX)}_")
PATH_GAN_HALF = os.path.join(PATH_GAN_LETTERS, f"{str(int(DATASET_MAX/2))}_")
PATH_GAN_TEN_P = os.path.join(PATH_GAN_LETTERS, f"{str(int(DATASET_MAX*0.1))}_")
PATH_GAN_MAX_HALF_TEN = [PATH_GAN_MAX, PATH_GAN_HALF, PATH_GAN_TEN_P]


def create_subfolder(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
        print(directory + ' created')


for f in (SAVED_FILES, SAVED_IMAGES, SAVED_MODELS, CLASS_DATA_PATH, STATS_FILES):
    create_subfolder(f)

# setup flags
FLAG_PREPARE_DATASETS = True
FLAG_TRAIN_GAN = False
FLAG_CLASSIFY = False
FLAG_GENERATE_IMAGES = False

# dataset fixed values
PIC_SIZE = (28, 28)  # only odd numbers
CHANNELS = 3
NUM_CLASSES = 1
CLASSES_TO_READ = ('B', 'D')

# classification model parameters
N_LAYERS = 4
MIN_NEURONS = 20
MAX_NEURONS = 120
KERNEL = (3, 3)

# gan model parameters
EPOCHS = 1  # todo
BATCH_SIZE = 1
SAMPLE_INTERVAL = 1000
