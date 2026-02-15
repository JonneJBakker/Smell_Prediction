from Data.data_prep import stratified_train_val_test_split
#from models.mpnn import train_mpnn
from training.training import train_mlc
import pandas as pd


DATA_PATH = r'Data/Multi-Labelled_Smiles_Odors_dataset.csv'
if __name__ == "__main__":
    data = pd.read_csv(DATA_PATH)
    #stratified_train_val_test_split(data, smiles_col='nonStereoSMILES', test_size=0.1, val_size=0.1)

    train_mlc()
    #train_mpnn(DATA_PATH)


