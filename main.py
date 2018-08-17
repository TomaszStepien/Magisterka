# coding=utf-8

"""magisterka"""

from random import seed

from load_data import load_train_valid_test_arrays

seed(2137)

dictionary = {'A': (10, 10, 10),
              'D': (10, 10, 10)
              }

x_train, y_train, x_valid, y_valid, x_test, y_test = load_train_valid_test_arrays(dictionary)

print(x_train.shape)
print(y_train.shape)
print(x_valid.shape)
print(y_valid.shape)
print(x_test.shape)
print(y_test.shape)
