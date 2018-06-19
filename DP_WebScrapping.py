import sys
sys.path.append('C:\\Users\\Karolina\\Documents\\MAGISTERKA\\magisterka')


import Keras_GAN.acgan.acgan as acgan

acgan = acgan.ACGAN()

X_train=X_train
epochs=EPOCHS
batch_size=BATCH_SIZE
sample_interval=SAMPLE_INTERVAL