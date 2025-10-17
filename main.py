import pandas as pd

from Data.analysis import SmellFragmentAnalyzer
from Data.structure_odor_visualizer import visualize_structure_odor

DATA_PATH = r'Data/Multi-Labelled_Smiles_Odors_dataset.csv'
if __name__ == "__main__":
    data = pd.read_csv(DATA_PATH)
    #print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
    #smiles_train, smiles_val, smiles_test, labels_train, labels_val, labels_test, label_cols = split_data(data, smiles_col="nonStereoSMILES")
    #visualize_data(data)
    #split_data(data, smiles_col="nonStereoSMILES")
    #augment_train_csv("Data/splits/smiles_train.csv", "Data/splits/labels_train.csv", out_csv_path="Data/splits/augmented_train.csv", percentile=95, k=10)
    #train_chemBerta(data, smiles_col="nonStereoSMILES")
    #func_detector = FunctionalGroupDetector()
    #func_detector.detect_from_csv(input_csv=DATA_PATH, output_csv="Data/smiles_with_func_groups.csv")
    #visualize_structure_odor(input_csv="Data/smiles_with_func_groups.csv", descriptor_column="descriptors")
    an = SmellFragmentAnalyzer(smell_col="descriptors", frag_prefix="fr_")
    an.full_analysis("Data/smiles_with_func_groups.csv", n_clusters=4)