import pandas as pd

from Data.data_prep import split_data
from training.chemBerta_mlc import train_chemBerta

DATA_PATH = r'Data\Multi-Labelled_Smiles_Odors_dataset.csv'
if __name__ == "__main__":

    data = pd.read_csv(DATA_PATH)
    ## visualize_data(data)
    #split_data(data, smiles_col="nonStereoSMILES")
    train_chemBerta(data, smiles_col="nonStereoSMILES")