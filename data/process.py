"""
File: process.py

Authors: Jared Azevedo & Andres Suarez

Desc: merge CSV files, purge duplicate records, and convert to numerical format
"""
import pandas as pd
import numpy as np
import sys


def fill_missing_data(merged_data_per_year, newids, desired_goods):
    """
    Desc: fill any missing goods for newids
    """
    # Newids as a list so they can be used in DataFrame column
    newids_as_list = np.repeat(list(newids), len(desired_goods))
    # Desired goods repeated for each newid so they can be used in DataFrame column
    all_desired_goods = desired_goods * len(newids)
    # Purchased binary value as a list (defaults to 0 for every good) so they can be used in DataFrame column
    all_purchased = [0] * (len(newids) * len(desired_goods))

    # Assemble above lists into dictionary that will create DataFrame
    missing_data_info = {"PURCHASED": all_purchased, "NEWID": newids_as_list, "UCC": all_desired_goods}

    # Cast dictionary as DataFrame
    missing_data = pd.DataFrame(missing_data_info)

    # Create indices for each DataFrame 
    merged_data_per_year_index = merged_data_per_year.set_index(["NEWID", "UCC"]).index
    missing_data_index = missing_data.set_index(["NEWID", "UCC"]).index

    # Isolate which NEWID, UCC rows are NOT in merged_data_per_year (i.e. the goods were not purchased)
    mask = ~missing_data_index.isin(merged_data_per_year_index)
    missing_data = missing_data[mask]

    # Add the missing rows so every NEWID now has every UCC whether it was purchased or not
    full_data = pd.concat([missing_data, merged_data_per_year])

    # Add demographic info to the unpurchased goods
    for ind in range(len(full_data)):
        if full_data.iloc[ind].PURCHASED == 0:
            temp_ucc = full_data.iloc[ind, 2]
            full_data.iloc[ind] = merged_data_per_year.loc[merged_data_per_year["NEWID"] == full_data.iloc[ind].NEWID].head(1)
            full_data.iloc[ind, 0] = 0
            full_data.iloc[ind, 2] = temp_ucc

    return full_data


def merge_quarter(directory, quarter, newids, desired_goods):
    """
    Desc: merges expd and fmld files for specified quarter
    """
    expd_data = pd.read_csv(f'{directory}/expd{quarter}.csv')
    # Only keep the columns we are interested in (expenditure info)
    expd_data = expd_data[["NEWID", "COST", "UCC"]]
    # Only keep the rows we are interested in (goods info)
    expd_data = expd_data[expd_data["UCC"].isin(desired_goods)]

    fmld_data = pd.read_csv(f'{directory}/fmld{quarter}.csv')
    # Only keep the columns we are interested in (demographic info)
    fmld_data = fmld_data[["NEWID", "AGE_REF", "BLS_URBN", "EDUC_REF", "FAM_TYPE", "HRSPRWK1", "FAM_SIZE", "FINCBEFX"]]

    merged_data = expd_data.merge(fmld_data, how="left", on="NEWID")

    newids.update(set(merged_data["NEWID"]))

    return merged_data


def merge_directory(directory, quarters):
    """
    Desc: merges all expd and fmld files in a given directory for the desired quarters
    """
    # Set list of goods codes that we want to keep
    desired_goods = [10110, 20510, 30110, 100210, 120310]
    newids = set()
    merged_data_per_quarter = []

    # Merge the necessary files for each quarter
    for quarter in quarters:
        merged_data_per_quarter.append(merge_quarter(directory, quarter, newids, desired_goods))

    # Merge all quarters together
    merged_data_per_year = pd.concat(merged_data_per_quarter)

    # Set COST column to binary value
    merged_data_per_year["COST"] = np.where(merged_data_per_year["COST"] > 0, 1, 0)

    # Reorder columns such that PURCHASED, NEWID, and UCC appear at the front
    label = merged_data_per_year["COST"]
    merged_data_per_year = merged_data_per_year.drop(columns=["COST"])
    merged_data_per_year.insert(loc=0, column="PURCHASED", value=label)

    newid = merged_data_per_year["NEWID"]
    merged_data_per_year = merged_data_per_year.drop(columns=["NEWID"])
    merged_data_per_year.insert(loc=1, column="NEWID", value=newid)

    ucc = merged_data_per_year["UCC"]
    merged_data_per_year = merged_data_per_year.drop(columns=["UCC"])
    merged_data_per_year.insert(loc=2, column="UCC", value=ucc)

    # Fill data so unpurchased goods are present
    merged_data_per_year = fill_missing_data(merged_data_per_year, newids, desired_goods)

    # Remove any duplicates
    merged_data_per_year.drop_duplicates()

    # Replace NaN with 0
    merged_data_per_year = merged_data_per_year.fillna(0)

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
