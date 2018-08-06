import os

import tensorflow as tf

import classifier1
import config
import gans.dcgan as dc_gan
import load_data as dl

if __name__ == "__main__":

    if config.FLAG_PREPARE_DATASETS:
        dl.prepare_final_datasets(config.DATASET_MAX)

    X_train, _, _, _ = dl.load_sets(path=config.PATH_GAN_LETTERS,
                                    sample_size=(100, 100),
                                    classes_to_read=['100_A'])

    if config.FLAG_TRAIN_GAN:
        # (Ustawienia do CUDA)
        c = tf.ConfigProto()
        c.gpu_options.allow_growth = True
        session = tf.Session(config=c)

        dcgan = dc_gan.DCGAN(pic_size=config.PIC_SIZE, channels=config.CHANNELS, num_classes=config.NUM_CLASSES)
        dcgan.train(X_train=X_train, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, \
                    save_interval=config.SAMPLE_INTERVAL)

    if config.FLAG_CLASSIFY:
        img_width, img_height = 28, 28
        home_path = os.path.join("C:\magisterka_data//MASTER_DATA//first_assumption", 'CLASS', '1000_100')
        nb_train_samples = 2000
        nb_validation_samples = 800
        epochs = 5
        batch_size = 16

        classifier1.train_classifier(home_path, img_width, img_height, epochs, batch_size)
