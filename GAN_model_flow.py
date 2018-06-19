
import sys
sys.path.append('C:\\Users\\Karolina\\Documents\\MAGISTERKA\\magisterka')

import GAN_files.aae as aae_gan
import GAN_data_load as dl
#import GAN_model_train as mt

# Dataset variables
PIC_SIZE = (32, 32)
TRAIN_PATH = 'C://magisterka_data//dogscats//train'
VALID_PATH = 'C://magisterka_data//dogscats//valid'
CHANNELS = 3
NUM_CLASSES = 1

# GAN model training variables
EPOCHS = 900 #todo
BATCH_SIZE = 32
SAMPLE_INTERVAL = 50
# Load dataset
print("LOADING DATASET")
X_train, _, _, _ = dl.load_sets(train_path=TRAIN_PATH, valid_path=VALID_PATH, pic_size=PIC_SIZE)
print("DATASET LOADED")

# Create neural network model
aae = aae_gan.AdversarialAutoencoder(pic_size=PIC_SIZE, channels=CHANNELS, num_classes=NUM_CLASSES)
aae.train(X_train=X_train, epochs=EPOCHS, batch_size=BATCH_SIZE, sample_interval=SAMPLE_INTERVAL)
