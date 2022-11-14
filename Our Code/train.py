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


class Normalize(tf.Module):
    def __init__(self, x):
        #Initialize the mean and standard deviation for normalization
        self.mean = tf.Variable(tf.math.reduce_mean(x, axis = 0))
        self.std = tf.Variable(tf.math.reduce_std(x, axis=0))

    def norm(self, x):
        #Normalize the input
        return (x - self.mean)/self.std
    
    def unnorm(self, x):
        #Unnormalize the input
        return (x * self.std) + self.mean



class LogisticRegression(tf.Module):

  def __init__(self):
    self.built = False

  def __call__(self, x, train=True):
    # Initialize the model parameters on the first call
    if not self.built:
      # Randomly generate the weights and the bias term
      rand_w = tf.random.uniform(shape=[x.shape[-1], 1], seed=22)
      rand_b = tf.random.uniform(shape=[], seed=22)
      self.w = tf.Variable(rand_w)
      self.b = tf.Variable(rand_b)
      self.built = True
    # Compute the model output
    z = tf.add(tf.matmul(x, self.w), self.b)
    z = tf.squeeze(z, axis=1)
    if train:
      return z
    return tf.sigmoid(z)



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


def normalize_feature_datasets(dictionary):
    """
    Desc: Normalizes features over the train, dev, and test sets.
    """
    #
    norm_x = Normalize(dictionary['train']['x'])
    
    #
    x_train_norm, x_dev_norm, x_test_norm = norm_x.norm(dictionary['train']['x']), norm_x.norm(dictionary['dev']['x']), norm_x.norm(dictionary['test']['x'])
    
    return {'train':{'x': x_train_norm,'y': dictionary['train']['y']}, 'dev':{'x': x_dev_norm, 'y': dictionary['dev']['y']}, 'test':{'x': x_test_norm, 'y': dictionary['test']['y']}}


def log_loss(y_pred, y):
  # Compute the log loss function
  ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_pred)
  return tf.reduce_mean(ce)


def train():
    pass 




if __name__ == '__main__':
    path = r'C:\Users\Andres Felipe Suarez\Documents\GitHub\cs230antitrust\Data\diary21'
    data = pd.read_csv(f'{path}\diary21_merged.csv')
    dictionary = train_dev_test_datasets(path, data, 0.8, 0.5)
    print(dictionary['train']['x'])
    dictionary = normalize_feature_datasets(dictionary)
    print('dictionary[train][x]:',dictionary['train']['x'])
    log_reg = LogisticRegression()
    y_pred = log_reg(dictionary['train']['x'][:5], train=False)
    print('a')
    y_pred.numpy()
    print(y_pred)

