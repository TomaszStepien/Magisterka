# coding=utf-8

"""classifier"""

import os

from keras import backend as K
from keras.callbacks import CSVLogger
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

import config
from classifier import classfication_stats


def train_classifier(home_path, option, folder, img_width, img_height, nb_train_samples, nb_validation_samples, epochs,
                     batch_size):
    """

    :param home_path:
    :param option:
    :param folder:
    :param img_width:
    :param img_height:
    :param nb_train_samples:
    :param nb_validation_samples:
    :param epochs:
    :param batch_size:
    """
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)
    path_csv = os.path.join(config.PATH_STATS, f"{option}_{folder}_class_output.csv")
    path_model = os.path.join(config.PATH_MODELS_CLASS, f"{option}_{folder}_model.h5")

    csv_logger = CSVLogger(path_csv, append=True, separator=';')
    train_data_dir = os.path.join(home_path, 'train', '')
    validation_data_dir = os.path.join(home_path, 'validation', '')

    model = _create_model(input_shape)
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                        target_size=(img_width, img_height),
                                                        batch_size=batch_size,
                                                        class_mode='binary')
    validation_generator = test_datagen.flow_from_directory(validation_data_dir,
                                                            target_size=(img_width, img_height),
                                                            batch_size=batch_size,
                                                            class_mode='binary')
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=nb_train_samples // batch_size,
                                  epochs=epochs,
                                  validation_data=validation_generator,
                                  validation_steps=nb_validation_samples // batch_size,
                                  callbacks=[csv_logger],
                                  verbose=False)

    classfication_stats.save_plots(history, option, folder)
    model.save(path_model)


def _create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model