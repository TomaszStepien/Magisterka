import GAN_data_load as dl
import GAN_files.dcgan as dc_gan
import GAN_files.aae as aae_gan
import PROCES_FLOW.preparation as prep
import config


def main():
    if config.FLAG_PREPARE_FOLDER_STRUCTURE:
        prep.prepare_folder_structure()

    print("LOADING DATASET")
    X_train, _, _, _ = dl.load_sets(sample_size=(20000, 100), classes_to_read=['B'])
    # (X_train, _), (_, _) = mnist.load_data()
    print("DATASET LOADED")

    if config.FLAG_TRAIN_GAN and config.FLAG_GAN_DCGAN:
        dcgan = dc_gan.DCGAN(pic_size=config.PIC_SIZE, channels=config.CHANNELS, num_classes=config.NUM_CLASSES)
        dcgan.train(X_train=X_train, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, \
                    save_interval=config.SAMPLE_INTERVAL)

    if config.FLAG_TRAIN_GAN and config.FLAG_GAN_AAE:
        aae = aae_gan.AdversarialAutoencoder(pic_size=config.PIC_SIZE, channels=config.CHANNELS, \
                                             num_classes=config.NUM_CLASSES)
        aae.train(X_train=X_train, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, \
                  sample_interval=config.SAMPLE_INTERVAL)


if __name__ == "__main__":
    main()
