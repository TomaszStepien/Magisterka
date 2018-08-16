# coding=utf-8

"""dcgan defined here"""

from __future__ import print_function, division

import csv
import datetime

import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import config


class DCGAN:
    """
    dcgan class
    """

    def __init__(self, pic_size=config.PIC_SIZE, channels=config.CHANNELS):
        if (pic_size[0] % 4) != 0 or (pic_size[1] % 4) != 0:
            raise ValueError('Picture size must be number possible to divide by 4!')

        self.img_rows = pic_size[0]
        self.img_cols = pic_size[0]
        self.channels = channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(100,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        """

        :return: 
        """
        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        """

        :return: 
        """
        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, x_train, epochs, batch_size=128, save_interval=50, stats_path='gan_output.csv'):
        """

        :param x_train:
        :param epochs:
        :param batch_size:
        :param save_interval:
        :param stats_path:
        """
        start_time = datetime.datetime.now()
        images_path = config.SAVED_IMAGES
        models_path = config.SAVED_MODELS

        # x_train = x_train / 127.5 - 1.
        # x_train = np.expand_dims(x_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs = x_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            computation_time = str(datetime.datetime.now() - start_time)
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f] exec.time: %s" % (
                epoch, d_loss[0], 100 * d_loss[1], g_loss, computation_time))
            with open(stats_path, 'a', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerow([epoch, d_loss[0], 100 * d_loss[1], g_loss, computation_time])

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(path=images_path, epoch=epoch)
                self.save_model(path=models_path)

    def save_imgs(self, path, epoch=1, full=False, amount=100, r=5, c=5):
        """
        Function for generating set of images from pretrained model or
        to control photo during training (r x c array of images)
        :param path: Path where image(s) should be save
        :param epoch: Number of epoch (while generating control photo during training)
        :param full: boolean variable - it True then generate photos if False then generate control photo
        :param amount: How many images should be generated (while generating set of images)
        :param r: number of rows in control photo
        :param c: number of columns in control photo
        :return: saved image
        """
        if full:
            noise = np.random.normal(0, 1, (amount, self.latent_dim))
            gen_imgs = self.generator.predict(noise)
            for i in range(amount):
                matplotlib.image.imsave(f"{path}/GAN_{i}.png", gen_imgs[i, :, :, 0])
                print(f"SAVED: GAN_{i}.png")
        else:
            noise = np.random.normal(0, 1, (r * c, self.latent_dim))
            gen_imgs = self.generator.predict(noise)
            # Rescale images 0 - 1
            gen_imgs = 0.5 * gen_imgs + 0.5
            fig, axs = plt.subplots(r, c)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                    axs[i, j].axis('off')
                    cnt += 1
            fig.savefig(f"{path}/{epoch}.png")
            plt.close()

    @staticmethod
    def save(model, path, model_name):
        """

        :param model:
        :param path:
        :param model_name:
        """
        model_path = f"{path}/{model_name}.json"
        weights_path = f"{path}/{model_name}_weights.hdf5"
        options = {"file_arch": model_path,
                   "file_weight": weights_path}
        json_string = model.to_json()
        open(options['file_arch'], 'w').write(json_string)
        model.save_weights(options['file_weight'])

    def save_model(self, path):
        """

        :param path:
        """
        self.save(self.generator, path, "dcgan_generator")
        self.save(self.discriminator, path, "dcgan_discriminator")
