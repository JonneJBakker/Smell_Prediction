from Data.analysis import SmellFragmentAnalyzer
from Data.rdkit_pca import RDKitPCA
from Data.structure_odor_visualizer import visualize_structure_odor
from Data.data_prep import split_data
from training.training import train_mlc
from training.chemberta_to_ffn import FrozenChemBERTaMultiLabel
from Data.data_prep import stratified_train_val_test_split
from Data.data_prep import augment_train_csv
from Data.analysis import plot_per_label_metrics
import pandas as pd

DATA_PATH = r'Data/Multi-Labelled_Smiles_Odors_dataset.csv'
if __name__ == "__main__":
    data = pd.read_csv(DATA_PATH)
    #stratified_train_val_test_split(data, smiles_col='nonStereoSmiles', test_size=0.1, val_size=0.1)
    #augment_train_csv('Data/splits/train_stratified.csv', smiles_col="nonStereoSMILES", out_csv_path=r'Data/splits/augment_train.csv')
    train_mlc()
    #plot_per_label_metrics(datapath="Data/Metrics/per_label_metrics_cls801010.csv")
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

    # 1) Descriptors only
    #rp = RDKitPCA(smiles_col="nonStereoSMILES", add_morgan=False)
    ##scores = rp.fit_transform(df, n_components=10, scale=True, extra_meta_cols=["smell_classes"])
    #display(scores.head())
    #display(rp.explained_variance())
    #display(rp.top_loadings(pc=1, top_k=15))

    #rp.plot_scree()
    #rp.plot_scores(pcx=1, pcy=2)


    ''''
    trainer = FrozenChemBERTaMultiLabel(
        csv_path=DATA_PATH,
        smiles_col="nonStereoSMILES",
        backbone="DeepChem/ChemBERTa-77M-MLM",
        n_labels=138,
        epochs=100,
        batch_train=32,
        batch_val=64,
        dropout=0.3,
        lr_head=1e-3,
        weight_decay=1e-2,
        warmup_frac=0.1,
        seed=1999,
    )

    #best_map = trainer.fit()
    #print(f"Best validation micro-mAP: {best_map:.4f}")
    test_smiles = [
        "CCO",  # ethanol
        "c1ccccc1O",  # phenol
    ]

    names, probs = trainer.predict_labels(test_smiles, return_probs=True)

    for smi, lbls, p in zip(test_smiles, names, probs):
        print(f"\nSMILES: {smi}")
        print(f"Predicted labels: {lbls}")
        print(f"Top 5 probabilities: {[(trainer.label_cols[i], round(float(p[i]), 3)) for i in np.argsort(-p)[:5]]}")
        '''
