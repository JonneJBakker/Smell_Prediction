import pandas as pd
import torch
from Data.data_prep import split_data, augment_train_csv
from training.chemBerta_mlc import train_chemBerta
from utils.data_visualizer import visualize_data

DATA_PATH = r'Data\splits\augmented_train.csv'
if __name__ == "__main__":
    data = pd.read_csv(DATA_PATH)
    print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
    #smiles_train, smiles_val, smiles_test, labels_train, labels_val, labels_test, label_cols = split_data(data, smiles_col="nonStereoSMILES")
    visualize_data(data)
    #split_data(data, smiles_col="nonStereoSMILES")
    #augment_train_csv("Data/splits/smiles_train.csv", "Data/splits/labels_train.csv", out_csv_path="Data/splits/augmented_train.csv", percentile=95, k=10)
    #train_chemBerta(data, smiles_col="nonStereoSMILES")