# coding=utf-8

"""magisterka"""

import os

import tensorflow as tf

import config
import gans.dcgan as dc_gan
import load_data as dl
from classifier import classfication_stats
from classifier import classifier
from src.tools import processing


def train_gans(option):
    """

    :param option:
    :return:
    """
    # Convert images to numpy arrays
    x_train, _, _, _ = dl.load_sets(path=config.PATH_GAN_LETTERS,
                                    sample_size=(int(option[1]), 100),
                                    classes_to_read=[f"{option[1]}_{letters[1]}"])

    # Train GAN
    dcgan = dc_gan.DCGAN(pic_size=config.PIC_SIZE,
                         channels=config.CHANNELS)
    dcgan.train(x_train=x_train,
                epochs=config.EPOCHS_GAN,
                batch_size=config.BATCH_SIZE,
                save_interval=config.SAMPLE_INTERVAL,
                stats_path=os.path.join(config.PATH_STATS_GAN, f"{option[1]}_{letters[1]}.csv"))
    return dcgan


def generate_images(option, dcgan):
    """

    :param option:
    :param dcgan:
    """
    path_first_letter = os.path.join(config.PATH_CLASS_LETTERS, f"{option[0]}_{option[1]}_GAN",
                                     f"{letters[0]}_{letters[1]}",
                                     'generated', letters[0])
    path_second_letter = os.path.join(config.PATH_CLASS_LETTERS, f"{option[0]}_{option[1]}_GAN",
                                      f"{letters[0]}_{letters[1]}",
                                      'generated', letters[1])

    dcgan.save_imgs(path=path_second_letter, full=True, amount=option[1])

    dl.train_validation_dividing(source_path=path_first_letter,
                                 destination_path=os.path.join(config.PATH_CLASS_LETTERS,
                                                               f"{option[0]}_{option[1]}_GAN",
                                                               f"{letters[0]}_{letters[1]}"),
                                 files=processing.return_all_files(path_first_letter),
                                 letter=letters[0],
                                 percentage=config.PROPORTION)
    dl.train_validation_dividing(source_path=path_second_letter,
                                 destination_path=os.path.join(config.PATH_CLASS_LETTERS,
                                                               f"{option[0]}_{option[1]}_GAN",
                                                               f"{letters[0]}_{letters[1]}"),
                                 files=processing.return_all_files(path_second_letter),
                                 letter=letters[1],
                                 percentage=config.PROPORTION)


def classify_images(path, option, folder):
    """

    :param path:
    :param option:
    :param folder:
    """
    img_width, img_height = config.PIC_SIZE[0], config.PIC_SIZE[1]
    home_path = os.path.join(path, option, folder)
    print(f"Classification {option} {folder}")
    if len(os.listdir(os.path.join(home_path, 'train', folder.split('_')[0]))) == 0:
        print("FOLDER EMPTY!!")
    else:
        samples = option.split('_')
        if len(samples) == 3:
            amount = int(samples[0]) + int(samples[1]) * 2
        else:
            amount = int(samples[0]) + int(samples[1])

        nb_train_samples = amount * config.PROPORTION
        nb_validation_samples = amount - nb_train_samples

        epochs = config.EPOCHS_CLASS
        batch_size = config.BATCH_SIZE
        classifier.train_classifier(home_path, option, folder, img_width, img_height, nb_train_samples,
                                    nb_validation_samples, epochs, batch_size)


if __name__ == "__main__":

    # CUDA settings
    c = tf.ConfigProto()
    c.gpu_options.allow_growth = True
    session = tf.Session(config=c)

    if config.FLAG_PREPARE_DATASETS:
        processing_time = processing.start_process('datasets preparation')
        for letters in config.LETTERS:
            dl.prepare_final_datasets(letters)
        processing.end_process(processing_time, 'datasets preparation')

    for index, option in enumerate(config.DATASETS_OPTIONS):
        option = [int(option[0]), int(option[1])]
        for letters in config.LETTERS:
            if config.FLAG_TRAIN_GAN:
                if index == 0:
                    pass
                else:
                    processing_time = processing.start_process('training GAN')
                    dcgan = train_gans(option)
                    processing.end_process(processing_time, 'training GAN')

                    if config.FLAG_GENERATE_IMAGES:
                        processing_time = processing.start_process('generating images')
                        generate_images(option, dcgan)
                        processing.end_process(processing_time, 'generating images')

    if config.FLAG_CLASSIFY:
        processing_time = processing.start_process('classifying images')
        option_folders = os.listdir(config.PATH_CLASS_LETTERS)
        for option in option_folders:
            to_classify = os.listdir(os.path.join(config.PATH_CLASS_LETTERS, option))
            for folder in to_classify:
                classify_images(config.PATH_CLASS_LETTERS, option, folder)
            processing.end_process(processing_time, 'classifying images')

    if config.FLAG_TEST_CLASSIFICATION:
        processing_time = processing.start_process('testing classification')
        for index, option in enumerate(config.DATASETS_LIST):
            for letters in config.LETTERS:
                model_name = f"{option}_{letters[0]}_{letters[1]}_model"
                images_path = os.path.join(config.PATH_CLASS_LETTERS, f"{option}", f"{letters[0]}_{letters[1]}/test",
                                           "")
                model_path = os.path.join(config.PATH_MODELS_CLASS, f"{model_name}.h5")
                classfication_stats.save_roc(images_path=images_path, model_path=model_path, model_name=model_name)
        processing.end_process(processing_time, 'testing classification')
