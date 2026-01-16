import deepchem as dc
from openpom.feat.graph_featurizer import GraphFeaturizer, GraphConvConstants
from openpom.utils.data_utils import get_class_imbalance_ratio, IterativeStratifiedSplitter
from openpom.models.mpnn_pom import MPNNPOMModel
from datetime import datetime
import os
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    hamming_loss,
    jaccard_score,
    roc_auc_score,
    precision_score,
    recall_score,
)
import pandas as pd

def train_mpnn(filepath = 'Data/Multi-Labelled_Smiles_Odors_dataset.csv'):
    df = pd.read_csv(filepath)

    TASKS = [col for col in df.columns if col not in ['nonStereoSMILES', 'descriptors']]



    featurizer = GraphFeaturizer()
    smiles_field = 'nonStereoSMILES'
    loader = dc.data.CSVLoader(tasks=TASKS,
                       feature_field=smiles_field,
                       featurizer=featurizer)
    dataset = loader.create_dataset(inputs=[filepath])
    n_tasks = len(dataset.tasks)

    randomstratifiedsplitter = dc.splits.RandomStratifiedSplitter()
    train_dataset, test_dataset, valid_dataset = randomstratifiedsplitter.train_valid_test_split(dataset, frac_train = 0.8, frac_valid = 0.1, frac_test = 0.1, seed = 1)

    train_ratios = get_class_imbalance_ratio(train_dataset)
    assert len(train_ratios) == n_tasks

    learning_rate = dc.models.optimizers.ExponentialDecay(initial_rate=0.001, decay_rate=0.5, decay_steps=32*20, staircase=True)


    model = MPNNPOMModel(n_tasks = n_tasks,
                                batch_size=128,
                                learning_rate=learning_rate,
                                class_imbalance_ratio = train_ratios,
                                loss_aggr_type = 'sum',
                                node_out_feats = 100,
                                edge_hidden_feats = 75,
                                edge_out_feats = 100,
                                num_step_message_passing = 5,
                                mpnn_residual = True,
                                message_aggregator_type = 'sum',
                                mode = 'classification',
                                number_atom_features = GraphConvConstants.ATOM_FDIM,
                                number_bond_features = GraphConvConstants.BOND_FDIM,
                                n_classes = 1,
                                readout_type = 'set2set',
                                num_step_set2set = 3,
                                num_layer_set2set = 2,
                                ffn_hidden_list= [392, 392],
                                ffn_embeddings = 256,
                                ffn_activation = 'relu',
                                ffn_dropout_p = 0.12,
                                ffn_dropout_at_input_no_act = False,
                                weight_decay = 1e-5,
                                self_loop = False,
                                optimizer_name = 'adam',
                                log_frequency = 32,
                                model_dir = './examples/experiments',
                                device_name='cuda')

    nb_epoch = 62

    metric_roc_auc = dc.metrics.Metric(
        dc.metrics.roc_auc_score,
        mode="classification",
        name="roc_auc_score",
    )


    metrics = [metric_roc_auc]

    start_time = datetime.now()
    for epoch in range(1, nb_epoch+1):
            loss = model.fit(
                  train_dataset,
                  nb_epoch=1,
                  max_checkpoints_to_keep=1,
                  deterministic=False,
                  restore=epoch>1)
            train_scores = model.evaluate(train_dataset, metrics)
            valid_scores = model.evaluate(valid_dataset, metrics)

            print(
                f"epoch {epoch}/{nb_epoch} ; loss = {loss}; "
                f"train_roc_auc = {train_scores['roc_auc_score']:.4f}; "
                f"valid_roc_auc = {valid_scores['roc_auc_score']:.4f}; "
            )
    model.save_checkpoint()
    end_time = datetime.now()

    test_scores = model.evaluate(test_dataset, metrics)
    print("time_taken: ", str(end_time - start_time))
    print("test_roc_auc: ", test_scores['roc_auc_score'])

    # DeepChem dataset labels: shape (N, T)
    y_true = test_dataset.y.astype(int)

    # Get model predictions on test set
    preds_ds = model.predict(test_dataset)
    # For DeepChem models, predictions are usually stored in .y
    if hasattr(preds_ds, "y"):
        y_prob = preds_ds.y  # shape (N, T)
    else:
        y_prob = preds_ds  # fallback, if it’s already a raw array

    # Use the SAME threshold as ChemBERTa
    threshold = 0.4  # or 0.25 if you want to match your trainer/eval

    y_pred = (y_prob >= threshold).astype(int)

    # --- ChemBERTa-style metrics (same as in chemberta_workflows_lora.py) ---

    micro_accuracy = accuracy_score(y_true, y_pred)
    auroc_macro = roc_auc_score(y_true, y_prob, average="macro")
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    samples_f1 = f1_score(y_true, y_pred, average="samples", zero_division=0)
    hamming = hamming_loss(y_true, y_pred)
    jaccard_samples = jaccard_score(y_true, y_pred, average="samples", zero_division=0)

    print("\nChemBERTa-style multi-label metrics on MPNN:")
    print(f"  micro_accuracy: {micro_accuracy:.4f}")
    print(f"  macro_auroc:    {auroc_macro:.4f}")
    print(f"  micro_f1:       {micro_f1:.4f}")
    print(f"  macro_f1:       {macro_f1:.4f}")
    print(f"  samples_f1:     {samples_f1:.4f}")
    print(f"  hamming_loss:   {hamming:.4f}")
    print(f"  jaccard_samp:   {jaccard_samples:.4f}")

    # List of label/odor names (same order as dataset.columns)
    target_cols = TASKS  # tasks already defined in your train_mpnn file

    # Per-label metrics
    precisions = precision_score(y_true, y_pred, average=None, zero_division=0)
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, average=None, zero_division=0)
    supports = y_true.sum(axis=0)
    freqs = y_true.mean(axis=0)

    df_metrics = pd.DataFrame({
        "label": target_cols,
        "precision": precisions,
        "recall": recalls,
        "f1": f1s,
        "support": supports,
        "frequency": freqs,
    }).sort_values("f1", ascending=False).reset_index(drop=True)

    print("\n=== Per-label metrics (MPNN, threshold = {:.2f}) ===".format(threshold))
    print(df_metrics.head(10))
    print(df_metrics.tail(10))
    output_dir = "../Data/Metrics/"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "mpnn_per_label_metrics.csv")
    df_metrics.to_csv(csv_path, index=False)
    print(f"\nSaved per-label metrics to: {csv_path}")