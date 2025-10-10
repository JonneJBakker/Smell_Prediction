import pandas as pd
from utils.data_visualizer import visualize_data


DATA_PATH = r'C:\Users\annad\PycharmProjects\Smell_Prediction\Data\Multi-Labelled_Smiles_Odors_dataset.csv'
if __name__ == "__main__":

    visualize_data(pd.read_csv(DATA_PATH))