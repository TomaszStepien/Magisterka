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

SAVED_FILES = home + 'saved_files/'
SAVED_IMAGES = home + 'saved_files/saved_images/'
SAVED_MODELS = home + 'saved_files/saved_models/'
CLASS_DATA_PATH = home + 'notMNIST_training/'
STATS_FILES = home + 'saved_files/stats_files/'


def create_subfolder(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
        print(directory + ' created')


for f in (SAVED_FILES, SAVED_IMAGES, SAVED_MODELS, CLASS_DATA_PATH, STATS_FILES):
    create_subfolder(f)

# setup flags
FLAG_TRAIN_GAN = True
FLAG_GAN_AAE = False
FLAG_GAN_DCGAN = True

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
EPOCHS = 1000001  # todo
BATCH_SIZE = 100
SAMPLE_INTERVAL = 1000
