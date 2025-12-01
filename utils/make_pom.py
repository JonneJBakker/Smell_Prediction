import os
import torch
import numpy as np
import pandas as pd
from PIL._imaging import display
from transformers import RobertaTokenizerFast
from torch.utils.data import DataLoader
import torch.nn.functional as F
import umap
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
from IPython.display import Image

from utils.chemberta_workflows import (  # <- change to your actual filename/module
    ChembertaMultiLabelClassifier,
    ChembertaDataset,
    DEFAULT_PRETRAINED_NAME,
    mean_pool
)

def chemberta_predict_embedding(model, dataset, device, batch_size=256):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_embeds = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Get RoBERTa outputs
            outputs = model.roberta(input_ids=input_ids, attention_mask=attention_mask)
            token_embs = outputs.last_hidden_state  # (B, L, H)

            # Use the same pooling as in your forward (e.g. "max_mean")
            mean_pooled = mean_pool(token_embs, attention_mask)  # from your utility file

            cls_emb = outputs.last_hidden_state[:, 0, :]

            mean_pooled = mean_pooled
            pooled = torch.cat([mean_pooled, cls_emb], dim=1)
            all_embeds.append(pooled.cpu().numpy())

    return np.concatenate(all_embeds, axis=0)  # (N, D)


def get_probabilities(model, dataset, device, batch_size=256):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_probs = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits                 # (B, num_labels)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    return np.concatenate(all_probs, axis=0)

def plot_pca(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- config you already know from training ---
    smiles_col = args.smiles_column               # same as args.smiles_column
    target_cols = args.target_columns             # same list as args.target_columns
    output_dir = "runs/optuna_focal_2/trial_11/chemberta_multilabel_model_final.bin"  # where Trainer saved the model

    # Load data you want to visualize (can be train+val+test or full dataset)
    df = pd.read_csv("Data/splits/train_stratified80.csv")

    num_labels = len(target_cols)

    tokenizer = RobertaTokenizerFast.from_pretrained(DEFAULT_PRETRAINED_NAME)

    # Dummy targets just to satisfy ChembertaDataset (we don't use them)
    dummy_targets = df[target_cols].values.astype(np.float32)

    dataset = ChembertaDataset(
        texts=df[smiles_col].tolist(),
        targets=dummy_targets,
        tokenizer=tokenizer,
    )

    # Recreate model with same hyperparams you used for training
    model = ChembertaMultiLabelClassifier(
        pretrained=DEFAULT_PRETRAINED_NAME,
        num_labels=num_labels,
        dropout=args.dropout,           # set to your args.dropout
        hidden_channels=args.hidden_channels,   # your args.hidden_channels
        num_mlp_layers=args.num_mlp_layers,      # your args.num_mlp_layers
        gamma=args.gamma,            # whatever you used
        alpha=args.alpha,            # whatever you used
        pooling_strat=args.pooling_strat,  # or "cls_mean" / whatever you trained with
    )

    state_dict = torch.load(os.path.join(output_dir),
                            map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    pom_frame(model, dataset, device, target_cols=target_cols)


def pom_frame(model, dataset, device, target_cols, is_preds=False, threshold=0.35):
    pom_embeds = chemberta_predict_embedding(model, dataset, device)
    y_preds = get_probabilities(model, dataset, device)  # the sigmoid(logits) part we wrote earlier
    required_desc = list(target_cols)
    type1 = {'floral': '#F3F1F7', 'subs': {'muguet': '#FAD7E6', 'lavender': '#8883BE', 'jasmin': '#BD81B7'}}
    type2 = {'meaty': '#F5EBE8', 'subs': {'savory': '#FBB360', 'beefy': '#7B382A', 'roasted': '#F7A69E'}}
    type3 = {'ethereal': '#F2F6EC', 'subs': {'cognac': '#BCE2D2', 'fermented': '#79944F', 'alcoholic': '#C2DA8F'}}

    # Assuming you have your features in the 'features' array
    pca = PCA(n_components=2,
              iterated_power=10)  # You can choose the number of components you want (e.g., 2 for 2D visualization)
    reduced_features = pca.fit_transform(pom_embeds)  # try different variations

    variance_explained = pca.explained_variance_ratio_

    # Variance explained by PC1 and PC2
    variance_pc1 = variance_explained[0]
    variance_pc2 = variance_explained[1]

    if is_preds:
        # y_preds is already a numpy array in your Kaggle-style code
        y = np.where(y_preds > threshold, 1.0, 0.0)
    else:
        # ChembertaDataset stores labels in .targets (a torch tensor)
        if isinstance(dataset.targets, torch.Tensor):
            y = dataset.targets.cpu().numpy()
        else:
            y = np.asarray(dataset.targets)

    # Right after you define `y` and `required_desc`
    labels_of_interest = [
        "floral", "muguet", "lavender", "jasmin",
        "meaty", "savory", "beefy", "roasted",
        "ethereal", "cognac", "fermented", "alcoholic",
    ]

    print("\n[make_pom] Positive counts per label (after threshold):")
    for lbl in labels_of_interest:
        if lbl in required_desc:
            idx = required_desc.index(lbl)
            n_pos = int(y[:, idx].sum())
            print(f"  {lbl:10s}: {n_pos}")
        else:
            print(f"  {lbl:10s}: NOT IN target_cols")
    print()

    # Generate grid points to evaluate the KDE on (try kernel convolution)
    x_grid, y_grid = np.meshgrid(np.linspace(reduced_features[:, 0].min(), reduced_features[:, 0].max(), 500),
                                 np.linspace(reduced_features[:, 1].min(), reduced_features[:, 1].max(), 500))
    grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()])

    def get_kde_values(label):
        plot_idx = required_desc.index(label)
        label_indices = np.where(y[:, plot_idx] == 1)[0]
        if label_indices.size < 2:
            print(f"Skipping label not enough points for KDE.")
            return None
        kde_label = gaussian_kde(reduced_features[label_indices].T)
        kde_values_label = kde_label(grid_points)
        kde_values_label = kde_values_label.reshape(x_grid.shape)
        return kde_values_label

    def plot_contours(type_dictionary, bbox_to_anchor):
        main_label = list(type_dictionary.keys())[0]

        # main “type” (e.g. "floral")
        main_kde = get_kde_values(main_label)
        legend_elements = []

        if main_kde is not None:
            plt.contourf(
                x_grid,
                y_grid,
                main_kde,
                levels=1,
                colors=['#00000000', type_dictionary[main_label], type_dictionary[main_label]]
            )

        # sublabels (e.g. "muguet", "lavender", ...)
        for label, color in type_dictionary['subs'].items():
            sub_kde = get_kde_values(label)
            if sub_kde is None:
                # optionally, print a warning once:
                print(f"Skipping label {label}: not enough points for KDE.")
                continue

            plt.contour(
                x_grid,
                y_grid,
                sub_kde,
                levels=1,
                colors=color,
                linewidths=2
            )
            legend_elements.append(Patch(facecolor=color, label=label))

        if legend_elements:
            legend = plt.legend(
                handles=legend_elements,
                title=main_label,
                bbox_to_anchor=bbox_to_anchor,
            )
            legend.get_frame().set_facecolor(type_dictionary[main_label])
            plt.gca().add_artist(legend)

    plt.figure(figsize=(15, 10))
    plt.title('KDE Density Estimation with Contours in Reduced Space')
    plt.xlabel(f'Principal Component 1 ({round(variance_pc1 * 100, ndigits=2)}%)')
    plt.ylabel(f'Principal Component 2 ({round(variance_pc2 * 100, ndigits=2)}%)')
    plot_contours(type_dictionary=type1, bbox_to_anchor=(0.2, 0.8))
    plot_contours(type_dictionary=type2, bbox_to_anchor=(0.9, 0.4))
    plot_contours(type_dictionary=type3, bbox_to_anchor=(0.3, 0.1))
    # plt.colorbar(label='Density')
    plt.show()
    png_file = os.path.join("Data/", f'pom_frame.png')
    plt.savefig(png_file)
    plt.close()
    pil_img = Image(filename=png_file)
    display(pil_img)




