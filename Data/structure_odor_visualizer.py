import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

def top_fragments_for_smell(frag_smell_df, smell, top_n=10):
    return frag_smell_df[smell].sort_values(ascending=False).head(top_n)

def smell_cooc(df, descriptor_column):
    
    # Get all unique smells
    all_smells = sorted({s for lst in df[descriptor_column] for s in lst})
    co_matrix = pd.DataFrame(0, index=all_smells, columns=all_smells)

    # Count co-occurrences
    for smells in df[descriptor_column]:
        for s1, s2 in combinations(smells, 2):
            co_matrix.loc[s1, s2] += 1
            co_matrix.loc[s2, s1] += 1
        # Also increment diagonal (self)
        for s in smells:
            co_matrix.loc[s, s] += 1

    # Heatmap of smell co-occurrence
    plt.figure(figsize=(10, 8))
    sns.heatmap(co_matrix, cmap="viridis", square=True)
    plt.title("Smell Co-occurrence Matrix", fontsize=14)
    plt.xlabel("Smell")
    plt.ylabel("Smell")
    plt.tight_layout()
    plt.show()

def heatmap(frag_smell_df):
    plt.figure(figsize=(14, 8))
    sns.heatmap(frag_smell_df, cmap="magma", linewidths=0.5)
    plt.title("Functional Groups vs. Smell Classes", fontsize=14)
    plt.xlabel("Smell Class")
    plt.ylabel("Functional Group")
    plt.tight_layout()
    plt.show()

def visualize_structure_odor(input_csv, descriptor_column):
    df = pd.read_csv(input_csv)
    df[descriptor_column] = df[descriptor_column].apply(lambda x: [s.strip() for s in x.split(";")])

    frag_cols = [c for c in df.columns if c.startswith("fr_")]

    # Initialize mapping
    smell_to_frags = defaultdict(lambda: defaultdict(int))

    # Count fragment occurrences by smell
    for _, row in df.iterrows():
        for smell in row[descriptor_column]:
            for frag in frag_cols:
                if row[frag] > 0:
                    smell_to_frags[smell][frag] += 1

    # Convert to DataFrame
    frag_smell_df = pd.DataFrame(smell_to_frags).fillna(0).astype(int)
    print(frag_smell_df.head())
    frag_smell_df.to_csv("Data/frag_smell.csv", index=False)

    print(top_fragments_for_smell(frag_smell_df, "fruity"))
    heatmap(frag_smell_df)
    smell_cooc(df, descriptor_column)

