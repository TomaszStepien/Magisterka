import sys
# sys.path.append('C:\\Users\\Karolina\\Documents\\MAGISTERKA\\magisterka')

import GAN_files.aae as aae_gan
import GAN_files.dcgan as dc_gan
import GAN_data_load as dl
import defaults
import PROCES_FLOW.preparation as prep

import numpy as np
from keras.datasets import mnist

# GAN model training variables
EPOCHS = 1000001  # todo
BATCH_SIZE = 100
SAMPLE_INTERVAL = 1000

prep.prepare_folder_structure()

# Load dataset
print("LOADING DATASET")
X_train, _, _, _ = dl.load_sets(sample_size=(20000,100), classes_to_read=['B'])
#(X_train, _), (_, _) = mnist.load_data()
print("DATASET LOADED")

# Create neural network model
#aae = aae_gan.AdversarialAutoencoder(pic_size=PIC_SIZE, channels=CHANNELS, num_classes=NUM_CLASSES)
#aae.train(X_train=X_train, epochs=EPOCHS, batch_size=BATCH_SIZE, sample_interval=SAMPLE_INTERVAL)
DCGAN = dc_gan.DCGAN(pic_size=defaults.PIC_SIZE, channels=defaults.CHANNELS, num_classes=defaults.NUM_CLASSES)
DCGAN.train(X_train=X_train, epochs=EPOCHS, batch_size=BATCH_SIZE, save_interval=SAMPLE_INTERVAL)
