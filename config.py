"""contains default values for parameters used in many files"""
import os
import sys

""" GAN model training variables """
EPOCHS = 1000001  # todo
BATCH_SIZE = 100
SAMPLE_INTERVAL = 1000

# Dataset

# DATASET = "C:\\magisterka_data\\dogscats\\"

if sys.platform.startswith('linux'):
    DATASET = "/home/tomasz/magisterka_data/notMNIST_large/"  # https://www.kaggle.com/volperosso/simple-cnn-classifier-on-notmnist/data
else:
    DATASET = "C:\\magisterka_data\\notMNIST_large\\"

SAVED_FILES = 'saved_files'
IMAGES = 'images'
MODELS = 'models'

""" PATHS FIXED VALUES """
PATH = DATASET
SAVED_FILES_PATH = os.path.join(SAVED_FILES, '')
SAVED_IMAGES_PATH = os.path.join(SAVED_FILES, IMAGES, '')
SAVED_MODELS_PATH = os.path.join(SAVED_FILES, MODELS, '')

CLASS_DATA_PATH = "C:\\magisterka_data\\notMNIST_training\\"

""" DATASET FIXED VALUES """
PIC_SIZE = (28, 28)  # allowed only odd numbers
CHANNELS = 3
NUM_CLASSES = 1
CLASSES_TO_READ = ('B', 'D')

""" CLASSIFICATION MODEL FIXED VALUES """
N_LAYERS = 4
MIN_NEURONS = 20
MAX_NEURONS = 120
KERNEL = (3, 3)

""" SETUP FLAGS """
FLAG_PREPARE_FOLDER_STRUCTURE = True
FLAG_TRAIN_GAN = True

FLAG_GAN_AAE = False
FLAG_GAN_DCGAN = True
