"""implements a simple GAN for dogscats pictures"""

import numpy as np
from GAN_data_load import load_sets

x_train, y_train, x_valid, y_valid = load_sets()

# %%
print(x_train)
print(y_train)
print(x_valid)
print(y_valid)


# %%
class TomaszGAN:
    """simplest GAN, trained to discriminate between real and fake images only"""

    def __init__(self, images):
        self.real_images = images

    def generate_fake_images(self):
        pass

    def generate_real_fake_set(self):
        pass

    def build_discriminator(self):
        pass

    def train(self):
        pass