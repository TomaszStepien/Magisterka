from importlib import reload

import numpy as np
from keras.layers import Activation, Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

import defaults

reload(defaults)


def cnn(size, n_layers):
    # INPUTS
    # size     - size of the input images
    # n_layers - number of layers
    # OUTPUTS
    # model    - compiled CNN

    # Define model hyperparamters
    MIN_NEURONS = defaults.MIN_NEURONS
    MAX_NEURONS = defaults.MAX_NEURONS
    KERNEL = defaults.KERNEL

    # Determine the # of neurons in each convolutional layer
    steps = np.floor(MAX_NEURONS / (n_layers + 1))
    nuerons = np.arange(MIN_NEURONS, MAX_NEURONS, steps)
    nuerons = nuerons.astype(np.int32)

    # Define a model
    model = Sequential()

    # Add convolutional layers
    for i in range(0, n_layers):
        if i == 0:
            shape = (size[0], size[1], size[2])
            model.add(Conv2D(nuerons[i], KERNEL, input_shape=shape))
        else:
            model.add(Conv2D(nuerons[i], KERNEL))

        model.add(Activation('relu'))

    # Add max pooling layer with dropout
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(MAX_NEURONS))
    model.add(Activation('relu'))

    # Add output layer
    model.add(Dense(1))  # Number of categories
    model.add(Activation('sigmoid'))  # softmax for many categories

    # Compile the model
    model.compile(loss='binary_crossentropy',  # binary_crossentropy for bin., categorical_crossentropy for cat.
                  optimizer='adam',
                  metrics=['accuracy'])

    # Print a summary of the model
    model.summary()

    return model


# Instantiate the model
model = cnn(size=(28, 28, 3), n_layers=defaults.N_LAYERS)

# TRAIN DATA
datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True,
                             rotation_range=5)  # , zoom_range=[1,0.9])
batch_size = 10
model_type = "binary"
generator_train = datagen.flow_from_directory(
    f"{defaults.CLASS_DATA_PATH}train",
    target_size=defaults.PIC_SIZE,
    batch_size=batch_size,
    class_mode=model_type,
    shuffle=False)

print(generator_train.class_indices)

# VALID DATA
datagen = ImageDataGenerator(rescale=1. / 255)
generator_valid = datagen.flow_from_directory(
    f"{defaults.CLASS_DATA_PATH}valid",
    target_size=defaults.PIC_SIZE,
    batch_size=batch_size,
    class_mode=model_type,
    shuffle=False)

print(generator_valid.class_indices)

# TRAIN MODEL
model.fit_generator(
    generator_train,
    steps_per_epoch=2000 // batch_size,
    epochs=10,
    validation_data=generator_valid,
    validation_steps=800 // batch_size)

# TEST MODEL
import os

test_labels = []
for x in range(len(os.listdir(f"{defaults.CLASS_DATA_PATH}test"))):
    test_labels += [x] * len(os.listdir(f"{defaults.CLASS_DATA_PATH}test/" +
                                        os.listdir(f"{defaults.CLASS_DATA_PATH}test")[x]))

batch_size = 50
datagen = ImageDataGenerator(rescale=1. / 255)
generator = datagen.flow_from_directory(f"{defaults.CLASS_DATA_PATH}test/",
                                        batch_size=batch_size,
                                        target_size=defaults.PIC_SIZE,
                                        classes=None, shuffle=False)

st_per_e_test = 1
test_data_features = model.predict_generator(generator,
                                             steps=st_per_e_test,
                                             use_multiprocessing=False, verbose=1)

from IPython.display import Image, display

k = 0

for x in range(len(test_data_features)):
    print(generator.filenames[x])
    k += 1
    print(round(test_data_features[x][0], 2))
    img_path = str(f"{defaults.CLASS_DATA_PATH}test/{generator.filenames[x]}")
    img = Image(img_path)
    display(img)
