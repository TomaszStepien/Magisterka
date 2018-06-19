import numpy as np

def train(self, epochs, batch_size=128, sample_interval=50):
    # Load the dataset
    (X_train, y_train), (_, _) = mnist.load_data()

    # Configure inputs
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=3)
    y_train = y_train.reshape(-1, 1)

    # Adversarial ground truths
    valid = np.ones((batch_size, 1))

    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random batch of images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]

        # Sample noise as generator input
        noise = np.random.normal(0, 1, (batch_size, 100))

        # The labels of the digits that the generator tries to create an
        # image representation of
        sampled_labels = np.random.randint(0, 10, (batch_size, 1))

        # Generate a half batch of new images
        gen_imgs = self.generator.predict([noise, sampled_labels])

        # Image labels. 0-9 if image is valid or 10 if it is generated (fake)
        img_labels = y_train[idx]
        fake_labels = 10 * np.ones(img_labels.shape)

        # Train the discriminator
        d_loss_real = self.discriminator.train_on_batch(imgs, [valid, img_labels])
        d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, fake_labels])
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------

        # Train the generator
        g_loss = self.combined.train_on_batch([noise, sampled_labels], [valid, sampled_labels])

        # Plot the progress
        print("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (
        epoch, d_loss[0], 100 * d_loss[3], 100 * d_loss[4], g_loss[0]))

        # If at save interval => save generated image samples
        if epoch % sample_interval == 0:
            self.save_model()
            self.sample_images(epoch)
