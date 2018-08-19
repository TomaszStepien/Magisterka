import os

import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

os.chdir('C:\\Users\\Karolina\\Documents\\[PRACA]\\Magisterka\\')
from keras.preprocessing.image import ImageDataGenerator
import config


def save_roc(images_path, model_path, model_name):
    model = load_model(model_path)
    images_labels = _get_files_labels(images_path)
    datagen = ImageDataGenerator(rescale=1. / 255)
    generator = datagen.flow_from_directory(images_path,
                                            batch_size=config.BATCH_SIZE, target_size=config.PIC_SIZE,
                                            classes=None, shuffle=False)
    st_per_set = int(np.floor(len(images_labels) / config.BATCH_SIZE))
    test_data_features = model.predict_generator(generator, steps=st_per_set, use_multiprocessing=False, verbose=1)
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(images_labels, test_data_features, pos_label=1)
    auc_keras = auc(fpr_keras, tpr_keras)

    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='AUC (area = {:.3f})'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(f"ROC curve for {model_name}")
    plt.legend(loc='best')
    plt.savefig(os.path.join(config.PATH_STATS_CLASS_ROC, f"{model_name}_roc.png"))
    plt.close()

    # Zoom in view of the upper left corner.
    plt.figure()
    plt.xlim(0, 0.4)
    plt.ylim(0.6, 1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='AUC (area = {:.3f})'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(f"ROC curve for {model_name} (zoomed in at top left)")
    plt.legend(loc='best')
    plt.savefig(os.path.join(config.PATH_STATS_CLASS_ROC, f"{model_name}_zoomed.png"))
    plt.close()


def _get_files_labels(images_path):
    test_labels = []
    for x in range(len(os.listdir(images_path))):
        test_labels += [x] * len(os.listdir(f"{images_path}/{os.listdir(images_path)[x]}"))
    return test_labels[:int(np.floor(len(test_labels) / config.BATCH_SIZE) * config.BATCH_SIZE)]


def save_plots(history, option, folder):
    plt.figure()
    plt.plot(history.history['loss'], 'r', linewidth=3.0)
    plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=16)
    plt.savefig(os.path.join(config.PATH_STATS_CLASS_ACC, f"{option}_{folder}_loss.png"))
    plt.close()

    plt.figure()
    plt.plot(history.history['acc'], 'r', linewidth=3.0)
    plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)
    plt.savefig(os.path.join(config.PATH_STATS_CLASS_ACC, f"{option}_{folder}_accuracy.png"))
    plt.close()
