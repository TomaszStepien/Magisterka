# coding=utf-8

"""classifier"""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
from keras.callbacks import CSVLogger
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator


def train_classifier(x_train, y_train, x_valid, y_valid, set_configuration, model_configuration):
    """
    trains_classifier and saves all the stuff to a folder
    """
    # (Ustawienia do CUDA)
    cf = tf.ConfigProto()
    cf.gpu_options.allow_growth = True

    if K.image_data_format() == 'channels_first':
        input_shape = (3, x_train.shape[1], x_train.shape[2])
    else:
        input_shape = (x_train.shape[1], x_train.shape[2], 3)

    letters = list(set_configuration.keys())
    folder_name = _create_model_folder(letters, (set_configuration[letters[0]][0], set_configuration[letters[1]][0]))

    path_csv = f'saved_models/{folder_name}/csv_logs.csv'
    csv_logger = CSVLogger(path_csv, append=True, separator=';')

    model = _create_model(input_shape)
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    validation_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(x=x_train,
                                         y=y_train,
                                         batch_size=model_configuration['batch_size'])

    validation_generator = validation_datagen.flow(x=x_valid,
                                                   y=y_valid,
                                                   batch_size=model_configuration['batch_size'])

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=x_train.shape[0] // model_configuration['batch_size'],
                                  epochs=model_configuration['epochs'],
                                  validation_data=validation_generator,
                                  validation_steps=x_valid.shape[0] // model_configuration['batch_size'],
                                  callbacks=[csv_logger],
                                  verbose=True)

    _save_plots(history, f'saved_models/{folder_name}')

    path_model = f'saved_models/{folder_name}/model.h5'
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
    model.add(Activation('softmax'))

    return model


def _save_plots(history, directory):
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['loss'], 'r', linewidth=3.0)
    plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=16)
    plt.savefig(f"{directory}/loss.png")
    plt.close()

    plt.figure(figsize=[8, 6])
    plt.plot(history.history['acc'], 'r', linewidth=3.0)
    plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)
    plt.savefig(f"{directory}/accuracy.png")
    plt.close()


def _create_model_folder(classes_names, classes_numbers):
    if not os.path.isdir('saved_folders'):
        os.mkdir('saved_folders')

    model_folder = f"{classes_names[0]}{classes_numbers[0]}_" \
                   f"{classes_names[1]}{classes_numbers[1]}_" \
                   f"{str(datetime.now())[:16].replace('-', '').replace(':', '').replace(' ', '_')}"

    os.mkdir(f'saved_models/{model_folder}')
    return model_folder
