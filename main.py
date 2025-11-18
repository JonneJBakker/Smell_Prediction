#from Data.analysis import SmellFragmentAnalyzer
#from Data.rdkit_pca import RDKitPCA
#from Data.structure_odor_visualizer import visualize_structure_odor
#from Data.data_prep import split_data
from training.training import train_mlc
#from training.chemberta_to_ffn import FrozenChemBERTaMultiLabel
#from Data.data_prep import stratified_train_val_test_split
#from Data.data_prep import augment_train_csv
#from Data.analysis import plot_per_label_metrics
#from models.mpnn import train_mpnn
import pandas as pd

DATA_PATH = r'Data/Multi-Labelled_Smiles_Odors_dataset.csv'
if __name__ == "__main__":
    data = pd.read_csv(DATA_PATH)
    #stratified_train_val_test_split(data, smiles_col='nonStereoSmiles', test_size=0.1, val_size=0.1)
    #augment_train_csv('Data/splits/train_stratified.csv', smiles_col="nonStereoSMILES", out_csv_path=r'Data/splits/augment_train.csv')
    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3]
    best_f1 = 0
    for threshold in thresholds:
        f1_macro = train_mlc(threshold)
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_f1_threshold = threshold

    print("Best F1-score threshold:", best_f1_threshold)
    print("Best F1-score:", f1_macro)
    #train_mpnn(DATA_PATH)
    #plot_per_label_metrics(datapath="Data/Metrics/focal_loss_60epoch.csv")
    #print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
    #smiles_train, smiles_val, smiles_test, labels_train, labels_val, labels_test, label_cols = split_data(data, smiles_col="nonStereoSMILES")
    #visualize_data(data)
    #split_data(data, smiles_col="nonStereoSMILES")
    #augment_train_csv("Data/splits/smiles_train.csv", "Data/splits/labels_train.csv", out_csv_path="Data/splits/augmented_train.csv", percentile=95, k=10)
    #train_chemBerta(data, smiles_col="nonStereoSMILES")
    #func_detector = FunctionalGroupDetector()
    #func_detector.detect_from_csv(input_csv=DATA_PATH, output_csv="Data/smiles_with_func_groups.csv")
    #visualize_structure_odor(input_csv="Data/smiles_with_func_groups.csv", descriptor_column="descriptors")
    #an = SmellFragmentAnalyzer(smell_col="descriptors", frag_prefix="fr_")
    #an.full_analysis("Data/smiles_with_func_groups.csv", n_clusters=3)
    #df = pd.read_csv("Data/smiles_with_func_groups.csv")


