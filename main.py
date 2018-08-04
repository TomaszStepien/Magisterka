import config
import gans.dcgan as dc_gan
import load_data as dl

if __name__ == "__main__":

    if config.FLAG_PREPARE_DATASETS:
        dl.prepare_final_datasets()

    X_train, _, _, _ = dl.load_sets(path=config.LETTERS_PATH, sample_size=(100, 100),
                                    classes_to_read=['A_100'])

    if config.FLAG_TRAIN_GAN:
        dcgan = dc_gan.DCGAN(pic_size=config.PIC_SIZE, channels=config.CHANNELS, num_classes=config.NUM_CLASSES)
        dcgan.train(X_train=X_train, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, \
                    save_interval=config.SAMPLE_INTERVAL)

    if config.FLAG_CLASSIFY:
