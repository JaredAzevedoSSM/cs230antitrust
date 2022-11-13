"""
Name: process.py
Author(s): Jared Azevedo & Andres Suarez
Desc: merge CSV files, purge duplicate records, and convert to numerical format
"""

import tensorflow as tf
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn.metrics as sk_metrics
import tempfile
import os

def train_dev_test_datasets(directory, data, train_frac, dev_frac):
    """
    Desc: Divides the data in train, dev, and test sets. Splits each database between features and labels. 
          Finally, converts the three dataset to type tensor.
    """
    #
    compiled_data = pd.read_csv(f'{directory}/diary{directory[-2:]}_merged.csv')
    #
    train_dataset = compiled_data.sample(frac = train_frac, random_state=1)
    dev_dataset = compiled_data.drop(train_dataset.index).sample(frac = dev_frac, random_state =1)
    test_dataset = compiled_data.drop(train_dataset.index).drop(dev_dataset.index)

    #
    x_train, y_train = train_dataset.iloc[:, 1:], train_dataset.iloc[:, 0]
    x_dev, y_dev = dev_dataset.iloc[:, 1:], dev_dataset.iloc[:, 0]
    x_test, y_test = test_dataset.iloc[:, 1:], test_dataset.iloc[:, 0]    

    #
    x_train, y_train = tf.convert_to_tensor(x_train, dtype=tf.float32), tf.convert_to_tensor(y_train, dtype=tf.float32)
    x_dev, y_dev = tf.convert_to_tensor(x_dev, dtype=tf.float32), tf.convert_to_tensor(y_dev, dtype=tf.float32)
    x_test, y_test = tf.convert_to_tensor(x_test, dtype=tf.float32), tf.convert_to_tensor(y_test, dtype=tf.float32)

    return {'train':{'x': x_train,'y': y_train}, 'dev':{'x': x_dev, 'y': y_dev}, 'test':{'x': x_test, 'y': y_test}}


if __name__ == '__main__':
    path = r'C:\Users\Andres Felipe Suarez\Documents\GitHub\cs230antitrust\Data\diary21'
    data = pd.read_csv(f'{path}\diary21_merged.csv')
    dictionary = train_dev_test_datasets(path, data, 0.8, 0.5)
    print(dictionary['train']['x'])
