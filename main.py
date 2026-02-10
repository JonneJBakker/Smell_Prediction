#from Data.analysis import SmellFragmentAnalyzer
#from Data.rdkit_pca import RDKitPCA
#from Data.structure_odor_visualizer import visualize_structure_odor
#from Data.data_prep import split_data
#from Data.data_prep import stratified_train_val_test_split
#from models.mpnn import train_mpnn
from training.training import train_mlc
#from utils.chemberta_workflows import sweep_thresholds_from_saved_results

#from training.chemberta_to_ffn import FrozenChemBERTaMultiLabel
#from Data.data_prep import stratified_train_val_test_split
#from Data.data_prep import augment_train_csv
#from Data.analysis import plot_per_label_metrics
#from models.mpnn import train_mpnn
import pandas as pd

from utils.chemberta_workflows_lora import sweep_thresholds_from_saved_results

DATA_PATH = r'Data/Multi-Labelled_Smiles_Odors_dataset.csv'
if __name__ == "__main__":
    data = pd.read_csv(DATA_PATH)
    #stratified_train_val_test_split(data, smiles_col='nonStereoSMILES', test_size=0.1, val_size=0.1)

    train_mlc()


    #train_mpnn(DATA_PATH)


