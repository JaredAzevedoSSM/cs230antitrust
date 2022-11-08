"""
    Name: process.py
    Author(s): Jared Azevedo & Andres Suarez
    Desc: merge CSV files, purge duplicate records, and convert to numerical format
"""
import pandas as pd
import os

def process_directory(path):
    files = os.listdir(path)
    data = files[0]
    files.pop(0)

    for file in files:
        data = pd.merge(data, pd.read_csv(file), on="NEWID", how="inner")

    data.drop_duplicates()

    data_head = data.head()
    drop_columns = []

    for col in data_head.columns:
        if data_head.at[1, col].strip().isalpha():
            drop_columns.append(col)
    
    data = data.drop(drop_columns, axis="columns")

    return data
