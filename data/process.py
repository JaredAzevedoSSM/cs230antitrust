"""
File: process.py

Authors: Jared Azevedo & Andres Suarez

Desc: merge CSV files, purge duplicate records, and convert to numerical format
"""
import pandas as pd
import sys
import os

def merge_quarter(directory, quarter):
    """
    Desc: merges expd and fmld files for specified quarter
    """
    expd_data = pd.read_csv(f'{directory}/expd{quarter}.csv')
    fmld_data = pd.read_csv(f'{directory}/fmld{quarter}.csv')
    merged_data = pd.merge(expd_data, fmld_data, how="left", on="NEWID")

    return merged_data

def merge_directory(directory, quarters):
    """
    Desc: merges all expd and fmld files in a given directory for the desired quarters
    """
    merged_data_per_quarter = []

    # Merge the necessary files for each quarter
    for quarter in quarters:
        merged_data_per_quarter.append(merge_quarter(directory, quarter))

    # Merge all quarters together
    merged_data_per_year = pd.concat(merged_data_per_quarter)

    # Remove any duplicates
    merged_data_per_year.drop_duplicates()

    head = merged_data_per_year.head()
    drop_columns = []

    # Find the columns that contain letters
    for col in head.columns:
        if str(head.at[1, col]).isalpha():
            drop_columns.append(col)
    
    # Remove columns that contain letters instead of numbers
    merged_data_per_year = merged_data_per_year.drop(drop_columns, axis="columns")

    # Reorder columns such that our label column is at the front
    label = merged_data_per_year["AMOUNT"]
    merged_data_per_year = merged_data_per_year.drop(columns=["AMOUNT"])
    merged_data_per_year.insert(loc=0, column="AMOUNT", value=label)

    return merged_data_per_year


def main(args):
    """
    Desc: process the quarters in the given directory and save it back to a new file so that we do not need to
    process our data every time we want to run the model
    """
    directory = args[0]
    quarters = args[1:]

    merged_directory = merge_directory(directory, quarters)

    merged_directory.to_csv(path_or_buf=f'{directory}/{directory}_merged.csv', index = False)


if __name__ == '__main__':
    # Capture directory and desired quarters to process as input argument from command line
    args = sys.argv[1:]

    # Check we have right number of args before proceeding
    if len(args) < 2:
        print("Please enter the path to a directory and the desired quarters (ex: ./diary21 211 212)")
    else:
        # Run processing
        main(args)
