import config
import gans.aae as aae_gan
import gans.dcgan as dc_gan
import load_data as dl

if __name__ == "__main__":

    print("Loading data...")
    X_train, _, _, _ = dl.load_sets(sample_size=(20000, 100), classes_to_read=['B'])
    # (X_train, _), (_, _) = mnist.load_data()
    print("Data loaded")

    if config.FLAG_TRAIN_GAN and config.FLAG_GAN_DCGAN:
        dcgan = dc_gan.DCGAN(pic_size=config.PIC_SIZE, channels=config.CHANNELS, num_classes=config.NUM_CLASSES)
        dcgan.train(X_train=X_train, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, \
                    save_interval=config.SAMPLE_INTERVAL)

    if config.FLAG_TRAIN_GAN and config.FLAG_GAN_AAE:
        aae = aae_gan.AdversarialAutoencoder(pic_size=config.PIC_SIZE, channels=config.CHANNELS, \
                                             num_classes=config.NUM_CLASSES)
        aae.train(X_train=X_train, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, \
                  sample_interval=config.SAMPLE_INTERVAL)
