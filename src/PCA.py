# Create a PCA forecast monitor
# It works by defining a "shape" of regular operation in a high-D space
# When the shape of operation deviates from that, it can flag an error 

# Imports 
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import polars.selectors as cs
#import seaborn as sns #only needed for covariance plot 

from preprocessing import * # import all the functions from the preprocessing file

# Load the data
tep_fault_free = pl.read_parquet("../data/tep_fault_free_training.parquet")
tep_faulty = pl.read_parquet("../data/tep_faulty_training_runs01-20.parquet")

# Create global variables
lst_channels = channels(tep_fault_free)
RUN = "simulationRun"
chan = channels(tep_fault_free)
metadata_cols = {"faultNumber", "simulationRun", "sample"}
feature_cols = [col for col in chan if col not in metadata_cols]

def PCA_model(df):
    "PCA model fitted on training data. Also visualizes covariance matrix"
    
    Z = df.select(feature_cols) # only want covariance under normal operation of the data, not the run numbers

    pca = PCA() # use PCA on the shortened test data
    pca.fit(Z.to_numpy()) # fit without column names, doesn't change anything on the training but avoids an error later
   
    """
    covariance = pca.get_covariance() # used to examine and see what variables are related 
    
    corr = pd.DataFrame(covariance, index=chan, columns=chan )
    fig, ax = plt.subplots(figsize=(14, 12))

    sns.heatmap(corr, cmap='coolwarm', xticklabels=True, yticklabels=True)
    ax.tick_params(labelsize=7)
    plt.title('Covariance Matrix Heatmap')
    plt.tight_layout()
    plt.show()
    """
    return(pca)



def score(pca, df):
    """Calculate PCA T2 and SPE scores for each row in df."""
    metadata = df.select(metadata_cols)
    Z = df.select(feature_cols)
    z = Z.to_numpy()

    # Find the smallest number of components explaining at least 90% of variance.
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.argmax(cumulative_variance >= 0.90) + 1)

    # sklearn's transform centers each row using pca.mean_ before projecting it.
    t_k = pca.transform(z)[:, :k]
    eigenvalues = pca.explained_variance_[:k]

    T2 = ((t_k ** 2) / eigenvalues).sum(axis=1)

    # Reconstruct each row using only the first k components.
    reconstruction = t_k @ pca.components_[:k, :] + pca.mean_
    SPE = ((z - reconstruction) ** 2).sum(axis=1)

    t_k_scores = pl.DataFrame(t_k, schema=[f"PC{i+1}" for i in range(k)])
    T2_score = pl.DataFrame(T2, schema=['T2'])
    SPE_score = pl.DataFrame(SPE, schema=['SPE'])

    Z = pl.concat([Z, t_k_scores, T2_score, SPE_score], how="horizontal_extend")
    
    return Z, metadata



def write_scores_pca():
    chan = channels(tep_fault_free) # list of channel names for tracked variables
    mean, std_dev = standardization(tep_fault_free, chan)
    fault_free_standardized = apply_standard(tep_fault_free, chan, mean, std_dev)
    faulty_standardized = apply_standard(tep_faulty, chan, mean, std_dev)

    train = training_runs(tep_fault_free) # define training set (1-300)
    train_standardized = apply_standard(train, chan, mean, std_dev)

    pca = PCA_model(train_standardized) # fit PCA on training data 

    # score both fault free AND faulty data
    fault_free_standardized_scored, fault_free_metadata = score(pca, fault_free_standardized)
    faulty_standardized_scored, faulty_metadata = score(pca, faulty_standardized)

    # now add back metadata, remove PCA and data columns
    parquet_cols = ["faultNumber", "simulationRun", "sample", "T2", "SPE"]
    fault_free_pca_data_df = pl.concat([fault_free_metadata, fault_free_standardized_scored], how="horizontal_extend")
    faulty_pca_data_df = pl.concat([faulty_metadata, faulty_standardized_scored], how="horizontal_extend")

    # if needed, this is all of the information
    # includes metadata, data, PCA and scores
    fault_free_pca_df = fault_free_pca_data_df.select(parquet_cols)
    faulty_pca_df = faulty_pca_data_df.select(parquet_cols)

    # write parquet file
    scores_pca = pl.concat([fault_free_pca_df,faulty_pca_df], how="vertical")
    scores_pca.write_parquet(r"..\results\scores_pca.parquet")
    return scores_pca

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

    scores_pca = write_scores_pca()
