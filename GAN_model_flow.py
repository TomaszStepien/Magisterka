import sys
# sys.path.append('C:\\Users\\Karolina\\Documents\\MAGISTERKA\\magisterka')

import GAN_files.aae as aae_gan
import GAN_files.dcgan as dc_gan
import GAN_data_load as dl
import defaults

import numpy as np

# GAN model training variables
EPOCHS = 1000000  # todo
BATCH_SIZE = 100
SAMPLE_INTERVAL = 10

# Load dataset
print("LOADING DATASET")
X_train, _, _, _ = dl.load_sets(classes_to_read=['dogs'])
print("DATASET LOADED")

X_train = np.array(X_train).astype(np.float32)
# Create neural network model
#aae = aae_gan.AdversarialAutoencoder(pic_size=PIC_SIZE, channels=CHANNELS, num_classes=NUM_CLASSES)
#aae.train(X_train=X_train, epochs=EPOCHS, batch_size=BATCH_SIZE, sample_interval=SAMPLE_INTERVAL)
DCGAN = dc_gan.DCGAN(pic_size=defaults.PIC_SIZE, channels=defaults.CHANNELS, num_classes=defaults.NUM_CLASSES)
DCGAN.train(X_train=X_train, epochs=EPOCHS, batch_size=BATCH_SIZE, save_interval=SAMPLE_INTERVAL)
