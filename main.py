# coding=utf-8

"""magisterka"""

from random import seed

from classifier import train_classifier
from load_data import load_train_valid_test_arrays

seed(2137)

set_configuration = {
    'A': (1000, 1000, 40),
    'D': (100, 1000, 40)
}

model_configuration = {
    'batch_size': 32,
    'epochs': 4
}

x_train, y_train, x_valid, y_valid, x_test, y_test = load_train_valid_test_arrays(set_configuration)

train_classifier(x_train, y_train, x_valid, y_valid, set_configuration, model_configuration)
