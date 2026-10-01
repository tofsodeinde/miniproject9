# Create .py file to edit the data to be shared by everyone

# Imports
import numpy as np
import polars as pl
import pandas as pd
#import matplotlib.pyplot as plt

# Load the data
tep_fault_free = pl.read_parquet("../data/tep_fault_free_training.parquet")
tep_faulty = pl.read_parquet("../data/tep_faulty_training_runs01-20.parquet")

# Define global variables
RUN = "simulationRun"

# Define preprocessing functions
def channels(df): # Get the list of channels
    lst = df.drop(["faultNumber", "simulationRun", "sample"])
    lst_channels = lst.columns
    return lst_channels

def training_runs(df): # faulty free
    train = df.filter(pl.col(RUN) <= 300)
    return train

def validation_runs(df): # faulty free
    validation = df.filter((pl.col(RUN) >= 301) & (pl.col(RUN) <= 400))
    return validation

def test_runs(df): # fault free
    test = df.filter((pl.col(RUN) >= 401) & (pl.col(RUN) <= 500))
    return test

def standardization(df, lst_channels): # standardize the data set
    # compute the mean for each column
    train = training_runs(df)
    mean = []
    std_dev = []
    for channel in lst_channels:
        mean.append(train.select(pl.col(channel).mean()).item()) #get the exact value
        std_dev.append(train.select(pl.col(channel).std(ddof=1)).item()) 
    return mean, std_dev

def apply_standard(df, lst_channels, mean, std_dev): # take in any df (i.e. training, validation, test)
    standardized = df
    for i in range(len(lst_channels)):
        standardized = standardized.with_columns(pl.col(lst_channels[i]).sub(mean[i]) / std_dev[i])
    return standardized



def main():
    chan = channels(tep_fault_free)
    mean, std_dev = standardization(tep_fault_free, chan)

    train = training_runs(tep_fault_free)
    val = validation_runs(tep_fault_free)
    test = test_runs(tep_fault_free)

    # Four standardized data frames for the training, validation, test, and faulty runs
    train_standardized = apply_standard(train, chan, mean, std_dev)
    val_standardized = apply_standard(val, chan, mean, std_dev)
    test_standardized = apply_standard(test, chan, mean, std_dev)
    faulty_standardized = apply_standard(tep_faulty, chan, mean, std_dev)
    


if __name__ == "__main__":
    main()