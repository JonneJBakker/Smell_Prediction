import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

def visualize_data(df):
    df = pd.read_csv(r'C:\Users\annad\PycharmProjects\Smell_Prediction\Data\Multi-Labelled_Smiles_Odors_dataset.csv')
    label_cols = df.columns[2:]

    # Count number of samples with label = 1 for each label
    label_counts = df[label_cols].sum().sort_values(ascending=False)

    # Compute percentages
    label_percentages = (label_counts / len(df)) * 100

    # Combine into one DataFrame
    label_stats = pd.DataFrame({
        'count': label_counts,
        'percentage': label_percentages.round(2)
    })

    label_stats['count'].plot(kind='bar')
    plt.title('Label Frequency Distribution')
    plt.ylabel('Number of Samples')
    plt.show()

    print(label_stats)

    imbalance_ratio = label_stats['count'].max() / label_stats['count'].min()
    print(f"Overall imbalance ratio: {imbalance_ratio:.2f}")

    threshold = 0.05 * len(df)  # labels appearing in less than 5% of samples
    rare_labels = label_stats[label_stats['count'] < threshold]
    print("Underrepresented labels:")
    print(rare_labels)

    X = df[label_cols]

    # 2. Standardize (optional, but good practice)
    #scaler = StandardScaler()
    #X_scaled = scaler.fit_transform(X)

    # 3. Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # 4. Convert to DataFrame for plotting
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    print(pca_df.head())

    # 5. Visualize
    plt.scatter(pca_df['PC1'], pca_df['PC2'])
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA on Multi-label Dataset')
    plt.show()

    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Total variance explained:", pca.explained_variance_ratio_.sum())

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(df[label_cols])

    rgb_labels = ['fruity', 'green', 'sweet']
    colors = df[rgb_labels].values  # shape: (n_samples, 3)
    colors = np.clip(colors, 0, 1)  # ensure values are between 0-1

    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], color=colors, s=15)
    plt.title("t-SNE on Multi-label Data")
    plt.show()