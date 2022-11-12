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




if __name__ == '__main__':
    path = r'C:\Users\Andres Felipe Suarez\Documents\GitHub\cs230antitrust\Data\diary21'
    data = pd.read_csv(f'{path}\compiled.csv')
    print(data.head())
    print(data.columns)

    train_dataset = data.sample(frac = 0.75, random_state=1)
    test_dataset = data.drop(train_dataset.index)
    x_train, y_train = train_dataset.iloc[:, 2:], train_dataset.iloc[:, 1]
    x_test, y_test = test_dataset.iloc[:, 2:], test_dataset.iloc[:, 1]



