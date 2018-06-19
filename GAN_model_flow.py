import sys
sys.path.append('C:\\Users\\Karolina\\Documents\\MAGISTERKA\\magisterka')

import Keras_GAN.acgan.acgan as acgan
import GAN_data_load as dl
import GAN_model_train

# Dataset variables
PIC_SIZE = (20, 20)
TRAIN_PATH = 'C://magisterka_data//dogscats//train'
VALID_PATH = 'C://magisterka_data//dogscats//valid'

# GAN model training variables
EPOCHS = 1 #todo
BATCH_SIZE = 128
SAMPLE_INTERVAL = 50

# Load dataset
X_train, y_train, _, _ = dl.load_sets(train_path=TRAIN_PATH, valid_path=VALID_PATH, pic_size=PIC_SIZE)


