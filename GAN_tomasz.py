"""implements a simple GAN for dogscats pictures"""

import numpy as np
from GAN_data_load import load_all_pictures

real_pics = load_all_pictures()

# %%
print(real_pics.shape)


# %%
class TomaszGAN:
    """simplest GAN, trained to discriminate between real and fake images only"""

    def __init__(self, real_images):
        self.real_images = real_images

    def generate_fake_images(self, npics, picsize=(20,20)):
        """generate random noise"""

        pass

    def generate_real_fake_set(self):
        pass

    def build_discriminator(self):
        pass

    def train(self):
        pass
