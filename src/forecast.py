# Create a "multivariate forecast-residual detector" (Detector from L8 but applied to every channel at once)

# Import Functions
import numpy as np
import polars as pl

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from preprocessing import * # import all the functions from the preprocessing file

# Load the data
tep_fault_free = pl.read_parquet("../data/tep_fault_free_training.parquet")
tep_faulty = pl.read_parquet("../data/tep_faulty_training_runs01-20.parquet")

# Create global variables
lst_channels = channels(tep_fault_free)
RUN = "simulationRun"

# Step 1: Create build table function (BUILT)
def build_table(df): # takes in standardized dataframe
    '''The forecast monitor is to predict exactly one step ahead for each channel'''
    '''Features are the standardized rows at t-1 and t-2,
    Targets are standardized rows at t, First two samples of each run have no score'''
    features = {} # initialize features column
    for k in range(1, 3): # get standardized rows at t-1 and t-2
        for Y in lst_channels:
            features[Y+f"[t-{k}]"] = pl.col(Y).shift(k).over("faultNumber", RUN)
    names = [f for f in features]
    for Y in lst_channels: # add target column per channel
        features[Y+" target"] = pl.col(Y)
    table = df.select("faultNumber", RUN, "sample", **features).drop_nulls() # drop null values
    return table, names


# Step 2: Fit one Ridge(alpha=1.0) on the training runs, predicting all 52 targets at once
def split(table, names): # Create a splitting function that takes in a built table and names
    targets = []
    for Y in lst_channels:
        targets.append(Y+" target")
    return (table.select(names).to_numpy(), table.select(targets).to_numpy())

def ridge(): # define model
    model = Ridge(alpha=1.0)
    return model

def fit(Xtr, ytr): # takes in built table
    one_step = ridge().fit(Xtr, ytr)
    return one_step


# Step 3: On the training runs, compute each channel's residual standard deviation (ddof=1)
def standard_residual(training_runs): # take in training runs dataframe
    # compute the residual for each channel
    table, names = build_table(training_runs)
    Xtr, ytr = split(table, names) 
    model = fit(Xtr, ytr)
    residual = ytr - model.predict(Xtr) # residual for each channel across all runs (numpy array)

    # take the standard deviation for each residual across all training rows
    std_dev = np.std(residual, axis=0)
    return std_dev

# Step 4: For every scored row, divide each channel's residual by that std dev, square, 
# and sum over the 52 channels (res/std_dev)**2 per channel. Sum = score
def scored(training_runs, other_runs): # Using the validation, test, and faulty runs standardized dataframes
    # train 
    train_table, train_names = build_table(training_runs)
    Xtr, ytr = split(train_table, train_names)
    model = fit(Xtr, ytr) # fit the model

    # Divide each channel's residual by the residual standard deviation
    table, names = build_table(other_runs)
    X, y = split(table, names)
    residual = y - model.predict(X)
    res_std_dev = standard_residual(training_runs)

    square = np.square(np.divide(residual, res_std_dev))
    score = np.sum(square, axis=1) # sum over the 52 channels

    scores_df = table.select(pl.col("faultNumber"), pl.col(RUN), pl.col("sample"))
    scores_df = scores_df.with_columns(score=pl.Series(score)) # add column
    return scores_df

def concatenate(lst): #takes in list of scored dataframes
    return pl.concat(lst)

#---------------------------------------------------------------------------------------------------------------
def main():
    mean, std_dev = standardization(tep_fault_free, lst_channels)
    
    train = training_runs(tep_fault_free)
    val = validation_runs(tep_fault_free)
    test = test_runs(tep_fault_free)
    
    # Four standardized data frames for the training, validation, test, and faulty runs
    train_standardized = apply_standard(train, lst_channels, mean, std_dev)
    val_standardized = apply_standard(val, lst_channels, mean, std_dev)
    test_standardized = apply_standard(test, lst_channels, mean, std_dev)
    faulty_standardized = apply_standard(tep_faulty, lst_channels, mean, std_dev)
    
    # Write results/scores_ridge.parquet
    v = scored(train_standardized, val_standardized)
    t = scored(train_standardized, test_standardized)
    f = scored(train_standardized, faulty_standardized)
    scores_ridge = concatenate([v, t, f])

    print(scores_ridge) #PRINT FINAL SCORES RIDGE TABLE
    
    #scores_ridge.write_parquet("../results/scores_ridge.parquet") # UNCOMMENT AFTER CHECKING

    

if __name__ == "__main__":
    main()