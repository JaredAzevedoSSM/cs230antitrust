"""
File: process.py

Authors: Jared Azevedo & Andres Suarez

Desc: merge CSV files, purge duplicate records, and convert to numerical format
"""
import pandas as pd
import numpy as np
import re
import sys
import os


def cast_to_float(value):
    """
    Desc: make sure every value is a float and doesn't contain other characters
    """
    number = re.findall(r'\d+', str(value))
    return float(number[0] if len(number) != 0 else 0)


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


def merge_quarter(directory, quarter, newids):
    """
    Desc: merges expd and fmld files for specified quarter
    """
    expd_data = pd.read_csv(f'{directory}/expd{quarter}.csv')
    # Only keep the columns we are interested in (expenditure info)
    expd_data = expd_data[["NEWID", "COST", "UCC"]]
    fmld_data = pd.read_csv(f'{directory}/fmld{quarter}.csv')
    # Only keep the columns we are interested in (demographic info)
    fmld_data = fmld_data[["NEWID", "AGE_REF", "AGE2", "ALCBEV","BAKEPROD",'BEEF',"BLS_URBN",'CEREAL','CHILDAGE','CUID','CUTENURE',
                        'DESCRIP','DRUGSUPP',
                        "EDUC_REF",'EGGS','EARNCOMP', 'EDUCA2', 'EMPLTYP1', 
                        "FAM_TYPE", 'FINCBEFX', 'FINLWT21','FIRAX','FJSSDEDX', 'FPVTX', 'FSS_RRX', 'FWAGEX',  "FAM_SIZE", 'FOODTOT', 'FOODHOME', 'FRSHFRUT',
                        "HRSPRWK1",'HRSPRWK2',
                        'JFS_AMT','JGRCFDMV', 'JGRCFDWK','JGROCYMV','JGROCYWK',
                        'MARITAL1', 'MILKPROD',
                        'NO_EARNR', 'OTHDAIRY',
                        'OCCEXPNX', 'OTHMEAT', 'OCCULIS2','OTHRECX', 'OCCULIS1', 
                        'PERSLT18','PERSOT64', 'PORK','POULTRY','POPSIZE',
                        'RACE2','REC_FS','REF_RACE','REGION',
                        'SEX_REF','SEX2','SMSASTAT','STRTMNTH', 'SEAFOOD',
                        'VEHQ',
                        'WEEKI','WK_WRKD2','WTREP01','WTREP02','WTREP03','WTREP04','WTREP05','WTREP06', 'WTREP07','WTREP08','WTREP09','WTREP10',
                        ]]

    merged_data = expd_data.merge(fmld_data, how="left", on="NEWID")

    # Keep track of each unique ID (survey response) to backfill missing data later
    newids.update(set(merged_data["NEWID"]))

    return merged_data


def merge_directory(directory):
    """
    Desc: merges all expd and fmld files in a given directory for the desired quarters
    """
    # Set list of goods codes that we want to keep
    # [10110, 20510, 30110, 100210, 120310]
    desired_goods = [20510]
    newids = set()
    merged_data_per_quarter = []

    # Retrieve each year quarters' numbers
    quarters = os.listdir(directory)
    quarters =  [c[c.index('.')-3: c.index('.')] for c in quarters if 'fmld' in c]

    # Merge the necessary files for each quarter
    for quarter in quarters:
        merged_data_per_quarter.append(merge_quarter(directory, quarter, newids))

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

    # Only keep the rows we are interested in (goods info)
    merged_data_per_year = merged_data_per_year[merged_data_per_year["UCC"].isin(desired_goods)]

    # Remove any duplicates
    # merged_data_per_year.drop_duplicates()

    # Replace NaN with 0
    merged_data_per_year = merged_data_per_year.fillna(0)

    # Cast every column value as a float
    merged_data_per_year = merged_data_per_year.applymap(cast_to_float)

    return merged_data_per_year


def main():
    """
    Desc: process the quarters in desired directories and save it back to a new file so that we do not need to
    process our data every time we want to run the model
    """
    # The directories we want to merge
    directories = ["./diary21", "./diary20", "./diary19", "./diary18", "./diary17", "./diary16", "./diary15", "./diary14", "./diary13", "./diary12", "./diary11", "./diary10", "./diary09"]
    merged_directories = []

    # Merge each directory
    for directory in directories:
        merged_directories.append(merge_directory(directory))

    # Concatenate all of the merged, preprocessed directory files together
    merged_directories = pd.concat(merged_directories)

    # Save the concatenated directories to a new file
    merged_directories.to_csv(path_or_buf='./merged_directories.csv', index = False)


if __name__ == '__main__':
    main()
