"""
Name: process.py
Author(s): Jared Azevedo & Andres Suarez
Desc: merge CSV files, purge duplicate records, and convert to numerical format
"""
import pandas as pd
import os


def step1(path, *quarters):
    """
    Merges expd and fmld files for each quarter individually.
    """
    for elem in quarters:
        data = pd.read_csv(f'{path}\expd{elem}.csv')
        #path = r'C:\Users\Andres Felipe Suarez\Documents\GitHub\cs230antitrust\Data\diary21'
        file = 'fmld'+str(elem)+'.csv'
        right = pd.read_csv(f'{path}\{file}')
        data = pd.merge(data, right, how='left', on='NEWID')
        data.to_csv(path_or_buf=f'{path}\dairy21_merged{elem}.csv', index = False)

    return data


def step2(path):
    """
    Appends the files that resulted from the previous step.
    """
    files = []

    for elem in os.listdir(path):
        if 'dairy21_merged' in str.lower(elem):
            files.append(elem)

    data = pd.read_csv(f'{path}\{files[0]}')
    files.pop()

    for elem in files:
        additional_data = pd.read_csv(f'{path}\{elem}')
        data = data.append(additional_data)

    data.to_csv(path_or_buf=f'{path}\compiled.csv', index = False)
    return data



if __name__ == '__main__':
    path = r'C:\Users\Andres Felipe Suarez\Documents\GitHub\cs230antitrust\Data\diary21'
    data = step1(path, 211, 212, 213, 214)
    print(data.head())
    print(data.shape)
    data = step2(path)
    print(data.head())
    print(data.shape)


