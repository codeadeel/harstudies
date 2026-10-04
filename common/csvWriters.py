# This file is responsible for writing and reading result tables with one fixed number format
# %%
# Importing Libraries
import pandas as pd


# %%
# Table Helpers
floatFormat = "%.6f"


def writeCsv(tableFrame, csvPath, includeIndex=False):
    """
    This function writes a result table with a fixed float format so identical runs give identical files

    Arguments
    =========
    tableFrame : Dataframe holding the table
    csvPath : Destination csv path
    includeIndex : Whether the dataframe index is written as the first column ( default : False )

    Output
    ======
    Path of the written csv file
    """
    # Write The Table
    tableFrame.to_csv(csvPath, index=includeIndex, float_format=floatFormat, lineterminator="\n")
    return csvPath


def readCsv(csvPath):
    """
    This function reads a result table written by writeCsv

    Arguments
    =========
    csvPath : Path of the csv file

    Output
    ======
    Dataframe holding the table
    """
    # Read The Table, Treating Only Empty Cells As Missing So Labels Such As NULL Stay Text
    return pd.read_csv(csvPath, keep_default_na=False, na_values=[""])
