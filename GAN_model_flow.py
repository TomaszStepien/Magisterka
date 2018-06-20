import sys
# sys.path.append('C:\\Users\\Karolina\\Documents\\MAGISTERKA\\magisterka')

import GAN_files.aae as aae_gan
import GAN_data_load as dl

import numpy as np

# import GAN_model_train as mt

# Dataset variables
PIC_SIZE = (32, 32)
TRAIN_PATH = 'C://magisterka_data//dogscats'
CHANNELS = 3
NUM_CLASSES = 1

# GAN model training variables
EPOCHS = 10000  # todo
BATCH_SIZE = 32
SAMPLE_INTERVAL = 100

# Load dataset
print("LOADING DATASET")
X_train, _, _, _ = dl.load_sets()
print("DATASET LOADED")

X_train = np.array(X_train).astype(np.float32)
# Create neural network model
aae = aae_gan.AdversarialAutoencoder(pic_size=PIC_SIZE, channels=CHANNELS, num_classes=NUM_CLASSES)
aae.train(X_train=X_train, epochs=EPOCHS, batch_size=BATCH_SIZE, sample_interval=SAMPLE_INTERVAL)
