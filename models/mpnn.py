import deepchem as dc
from openpom.feat.graph_featurizer import GraphFeaturizer, GraphConvConstants
from openpom.utils.data_utils import get_class_imbalance_ratio, IterativeStratifiedSplitter
from openpom.models.mpnn_pom import MPNNPOMModel
from datetime import datetime
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np

def train_mpnn(filepath = 'Data/Multi-Labelled_Smiles_Odors_dataset.csv'):
    TASKS = [
    'alcoholic', 'aldehydic', 'alliaceous', 'almond', 'amber', 'animal',
    'anisic', 'apple', 'apricot', 'aromatic', 'balsamic', 'banana', 'beefy',
    'bergamot', 'berry', 'bitter', 'black currant', 'brandy', 'burnt',
    'buttery', 'cabbage', 'camphoreous', 'caramellic', 'cedar', 'celery',
    'chamomile', 'cheesy', 'cherry', 'chocolate', 'cinnamon', 'citrus', 'clean',
    'clove', 'cocoa', 'coconut', 'coffee', 'cognac', 'cooked', 'cooling',
    'cortex', 'coumarinic', 'creamy', 'cucumber', 'dairy', 'dry', 'earthy',
    'ethereal', 'fatty', 'fermented', 'fishy', 'floral', 'fresh', 'fruit skin',
    'fruity', 'garlic', 'gassy', 'geranium', 'grape', 'grapefruit', 'grassy',
    'green', 'hawthorn', 'hay', 'hazelnut', 'herbal', 'honey', 'hyacinth',
    'jasmin', 'juicy', 'ketonic', 'lactonic', 'lavender', 'leafy', 'leathery',
    'lemon', 'lily', 'malty', 'meaty', 'medicinal', 'melon', 'metallic',
    'milky', 'mint', 'muguet', 'mushroom', 'musk', 'musty', 'natural', 'nutty',
    'odorless', 'oily', 'onion', 'orange', 'orangeflower', 'orris', 'ozone',
    'peach', 'pear', 'phenolic', 'pine', 'pineapple', 'plum', 'popcorn',
    'potato', 'powdery', 'pungent', 'radish', 'raspberry', 'ripe', 'roasted',
    'rose', 'rummy', 'sandalwood', 'savory', 'sharp', 'smoky', 'soapy',
    'solvent', 'sour', 'spicy', 'strawberry', 'sulfurous', 'sweaty', 'sweet',
    'tea', 'terpenic', 'tobacco', 'tomato', 'tropical', 'vanilla', 'vegetable',
    'vetiver', 'violet', 'warm', 'waxy', 'weedy', 'winey', 'woody'
    ]


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
        name="roc_auc_score"
    )

    metric_f1_micro = dc.metrics.Metric(
        lambda y_true, y_pred, w: f1_score(
            y_true,
            (y_pred > 0.5).astype(int),
            average="micro"
        ),
        mode="classification",
        name="f1_micro"
    )

    metric_f1_macro = dc.metrics.Metric(
        lambda y_true, y_pred, w: f1_score(
            y_true,
            (y_pred > 0.5).astype(int),
            average="macro"
        ),
        mode="classification",
        name="f1_macro"
    )

    metrics = [metric_roc_auc, metric_f1_micro, metric_f1_macro]

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
                f"train_f1_micro = {train_scores['f1_micro']:.4f}; "
                f"train_f1_macro = {train_scores['f1_macro']:.4f}; "
                f"valid_roc_auc = {valid_scores['roc_auc_score']:.4f}; "
                f"valid_f1_micro = {valid_scores['f1_micro']:.4f}; "
                f"valid_f1_macro = {valid_scores['f1_macro']:.4f}"
            )
    model.save_checkpoint()
    end_time = datetime.now()

    test_scores = model.evaluate(test_dataset, metrics)
    print("time_taken: ", str(end_time - start_time))
    print("test_roc_auc: ", test_scores['roc_auc_score'])
    print("test_f1_micro: ", test_scores['f1_micro'])
    print("test_f1_macro: ", test_scores['f1_macro'])