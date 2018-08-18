# coding=utf-8

"""magisterka"""

from random import seed

from classifier import train_classifier
from load_data import load_train_valid_test_arrays

seed(2137)

set_configuration = {'A': (10, 10, 10),
                     'D': (10, 10, 10)
                     }

x_train, y_train, x_valid, y_valid, x_test, y_test = load_train_valid_test_arrays(set_configuration)

print(x_train.shape)
print(y_train.shape)
print(x_valid.shape)
print(y_valid.shape)
print(x_test.shape)
print(y_test.shape)

print(y_train)

model_configuration = {
    'batch_size': 32,
    'epochs': 1
}

train_classifier(x_train, y_train, x_valid, y_valid, set_configuration, model_configuration)
