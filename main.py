import os

import tensorflow as tf

import classifier1
import config
import gans.dcgan as dc_gan
import load_data as dl
import numpy as np


def train_gans(option):
    # Convert images to numpy arrays
    X_train, _, _, _ = dl.load_sets(path=config.PATH_GAN_LETTERS,
                                    sample_size=(100, 100),
                                    classes_to_read=[f"{option[1]}_{letters[1]}"])
    # CUDA settings
    c = tf.ConfigProto()
    c.gpu_options.allow_growth = True
    session = tf.Session(config=c)
    # Train GAN
    dcgan = dc_gan.DCGAN(pic_size=config.PIC_SIZE, channels=config.CHANNELS,
                         num_classes=config.NUM_CLASSES)
    dcgan.train(X_train=X_train, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
                save_interval=config.SAMPLE_INTERVAL)
    return dcgan


if __name__ == "__main__":

    try_balance = False

    for option in config.DATASETS_OPTIONS:
        option = [int(option[0]), int(option[1])]
        for letters in config.LETTERS:
            if config.FLAG_PREPARE_DATASETS:
                dl.prepare_final_datasets(letters)

            if (option[0] == option[1]):
                pass
            else:
                if config.FLAG_TRAIN_GAN:
                    dcgan = train_gans(option)

                if config.FLAG_GENERATE_IMAGES:
                    path_first_letter = os.path.join(config.PATH_CLASS_LETTERS, f"{option[0]}_{option[1]}_GAN",
                                                     'generated', letters[0])
                    path_second_letter = os.path.join(config.PATH_CLASS_LETTERS, f"{option[0]}_{option[1]}_GAN",
                                                      'generated', letters[1])

                    dcgan.save_imgs(path=path_second_letter, full=True, amount=option[1])
                    dl.train_validation_dividing(source_path=path_first_letter,
                                                 destination_path=os.path.join(config.PATH_CLASS_LETTERS,
                                                                               f"{option[0]}_{option[1]}_GAN"),
                                                 files=[], letter=letters[0], percentage=0.7)
                    dl.train_validation_dividing(source_path=path_second_letter,
                                                 destination_path=os.path.join(config.PATH_CLASS_LETTERS,
                                                                               f"{option[0]}_{option[1]}_GAN"),
                                                 files=[], letter=letters[1], percentage=0.7)

            if config.FLAG_CLASSIFY:
                img_width, img_height = config.PIC_SIZE[0], config.PIC_SIZE[1]
                if try_balance:
                    home_path = os.path.join(config.PATH_CLASS_LETTERS, f"{option[0]}_{option[1]}_GAN")
                else:
                    home_path = os.path.join(config.PATH_CLASS_LETTERS, f"{option[0]}_{option[1]}")

                nb_train_samples = 2000
                nb_validation_samples = 800
                epochs = 500
                batch_size = 16

                classifier1.train_classifier(home_path, img_width, img_height, epochs, batch_size)
