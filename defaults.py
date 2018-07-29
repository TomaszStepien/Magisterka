"""contains default values for parameters used in many files"""

""" PATHS FIXED VALUES """
# PATH = "C:\\magisterka_data\\dogscats\\"
PATH = "C:\\magisterka_data\\notMNIST_large\\"  # https://www.kaggle.com/volperosso/simple-cnn-classifier-on-notmnist/data
SAVED_FILES_PATH = "/SAVED_FILES/"
SAVED_IMAGES_PATH = "/SAVED_FILES/images/"
SAVED_MODELS_PATH = "/SAVED_FILES/models/"

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
