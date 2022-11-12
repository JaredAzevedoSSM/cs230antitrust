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


def weekly_files(path):
files = []
for elem in os.listdir(path):
if 'expd' in str.lower(elem):
files.append(elem)

data = pd.read_csv(f'{path}\{files[0]}')
files.pop()

for elem in files:
additional_data = pd.read_csv(f'{path}\{elem}')
data = data.append(additional_data)
print(data.shape)

data.to_csv(path_or_buf=f'{path}\data_dairy21.csv', index = False)
return data

def annual_files(path, data):
files = []
for elem in os.listdir(path):
if 'fmld' in str.lower(elem):
files.append(elem)

for elem in files:
right = pd.read_csv(f'{path}\{elem}')
data = pd.merge(data, right, how = 'left', on='NEWID')

data.to_csv(path_or_buf=f'{path}\data_dairy21_merged.csv', index = False)
return data

def files_fn(list):
files = []
for i in list:
if '.csv' in i:
files.append(i)
return files

def process_directory_version2(path, file_name):
folders = folder(os.listdir(path))
print(folders)
data = pd.read_csv(f'{path}\{folders[0]}\{file_name}.csv')
print(data)
folders.pop(0)
print(folders)

for elem in folders:
print(elem)
path = f'{path}'
print(f'{path}\{elem}')
print(elem[-2:])
data = pd.merge(data, pd.read_csv(f'{path}\{elem}\dtbd{elem[-2:]}1.csv'), on="NEWID", how="inner")
print(data)

print(data)
data.drop_duplicates()

data_head = data.head()
drop_columns = []
print(data)

for col in data_head.columns:
if data_head.at[1, col].strip().isalpha():
drop_columns.append(col)

data = data.drop(drop_columns, axis="columns")

return data

def process_directory(path):
files = files_fn(os.listdir(path))
print(files)
data = pd.read_csv(f'{path}\{files[0]}')

print('data type:', type(data))
files.pop(0)

for file in files:
add = pd.read_csv(f'{path}\{file}')
print(add, add.shape)
data = pd.merge(data, add, on="NEWID", how="outer")
print('data.shape:',data.shape)


data.drop_duplicates()

data_head = data.head()
drop_columns = []

#  for col in data_head.columns:
#      if data_head.at[1, col].strip().isalpha():
#          drop_columns.append(col)

#  data = data.drop(drop_columns, axis="columns")

return data


if __name__ == '__main__':
path = r'C:\Users\Andres Felipe Suarez\Documents\GitHub\cs230antitrust\Data\diary21'
data = step1(path, 211, 212, 213, 214)
print(data.head())
print(data.shape)
data = step2(path)
print(data.head())
print(data.shape)


