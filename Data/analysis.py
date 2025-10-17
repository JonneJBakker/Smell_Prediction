import os
from typing import Optional, List, Tuple, Union, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


class SmellFragmentAnalyzer:
    """
    Analyze relationships between RDKit functional-group fragments (fr_*)
    and multi-labeled smell classes.

    Features:
      - Build fragment x smell frequency matrix
      - Smell co-occurrence matrix
      - Fragment co-occurrence/correlation
      - Clustering of smells by fragment patterns (PCA + KMeans)
      - Matplotlib heatmaps and scatter plots
      - Export tables and figures

    Expected columns:
      - fragments: columns starting with frag_prefix (default 'fr_')
      - smell labels: a column with multi-label text (default 'smell_classes')
      - smiles: optional, not used for analysis but kept (default 'smiles')
    """

    def __init__(
        self,
        smiles_col: str = "smiles",
        smell_col: str = "smell_classes",
        frag_prefix: str = "fr_",
        smell_separator: str = ",",
    ):
        self.smiles_col = smiles_col
        self.smell_col = smell_col
        self.frag_prefix = frag_prefix
        self.smell_separator = smell_separator

        self.df: Optional[pd.DataFrame] = None
        self.frag_cols: List[str] = []
        self.frag_smell_df: Optional[pd.DataFrame] = None   # rows: fragments, cols: smells
        self.smell_co_matrix: Optional[pd.DataFrame] = None # smells x smells
        self.frag_corr: Optional[pd.DataFrame] = None       # fragments x fragments
        self.pca_df: Optional[pd.DataFrame] = None          # smells x [PC1, PC2, Cluster]
        self.pca_model: Optional[PCA] = None
        self.kmeans_model: Optional[KMeans] = None

    # ---------- Data loading & preparation ----------

    def load(
        self,
        data: Union[str, pd.DataFrame],
        infer_frag_cols: bool = True,
        copy_df: bool = True,
    ) -> "SmellFragmentAnalyzer":
        """
        Load data from a CSV path or an existing DataFrame.
        """
        if isinstance(data, str):
            df = pd.read_csv(data)
        else:
            df = data.copy() if copy_df else data

        # Ensure smell column is list-like
        if self.smell_col not in df.columns:
            raise ValueError(f"Missing required column '{self.smell_col}'")
        df[self.smell_col] = df[self.smell_col].apply(self._parse_smell_labels)

        # Identify fragment columns
        if infer_frag_cols:
            self.frag_cols = [c for c in df.columns if c.startswith(self.frag_prefix)]
            if not self.frag_cols:
                raise ValueError(
                    f"No fragment columns found with prefix '{self.frag_prefix}'."
                )

        # Clean fragment columns to numeric (handle missing/strings)
        for c in self.frag_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

        self.df = df
        return self

    def _parse_smell_labels(self, x) -> List[str]:
        if isinstance(x, list):
            return [str(s).strip() for s in x]
        if pd.isna(x):
            return []
        return [s.strip() for s in str(x).split(self.smell_separator) if s.strip()]

    # ---------- Core tables ----------

    def build_fragment_smell_matrix(
        self,
        binary_frag_presence: bool = True,
        min_count: int = 0,
    ) -> pd.DataFrame:
        """
        Build a matrix: rows = fragments, cols = smells, values = frequency of co-occurrence.
        If binary_frag_presence=True, count a fragment if its value > 0 in a molecule.
        Otherwise sum the integer fragment counts.
        min_count filters fragments by total across all smells.
        """
        if self.df is None:
            raise RuntimeError("No data loaded. Call .load(...) first.")

        smell_to_frag_counts: Dict[str, Dict[str, int]] = {}

        for _, row in self.df.iterrows():
            smells = row[self.smell_col]
            if not smells:
                continue
            for frag in self.frag_cols:
                val = int(row[frag])
                if binary_frag_presence:
                    val = 1 if val > 0 else 0
                if val == 0:
                    continue
                for smell in smells:
                    smell_to_frag_counts.setdefault(smell, {})
                    smell_to_frag_counts[smell][frag] = smell_to_frag_counts[smell].get(frag, 0) + val

        if not smell_to_frag_counts:
            self.frag_smell_df = pd.DataFrame(columns=[], index=self.frag_cols)
            return self.frag_smell_df

        mat = pd.DataFrame(smell_to_frag_counts).fillna(0).astype(int)
        # Reindex to ensure all fragments present as rows
        mat = mat.reindex(index=sorted(set(self.frag_cols) & set(mat.index)), fill_value=0)

        # Optional filter by min_count (row-wise)
        if min_count > 0:
            keep = mat.sum(axis=1) >= min_count
            mat = mat.loc[keep]

        self.frag_smell_df = mat
        return mat

    def build_smell_cooccurrence(self) -> pd.DataFrame:
        """
        Build symmetric smell x smell co-occurrence matrix.
        Diagonal = number of molecules labeled with that smell.
        """
        if self.df is None:
            raise RuntimeError("No data loaded. Call .load(...) first.")

        all_smells = sorted({s for lbls in self.df[self.smell_col] for s in lbls})
        co = pd.DataFrame(0, index=all_smells, columns=all_smells, dtype=int)

        for smells in self.df[self.smell_col]:
            # increment diagonal
            for s in smells:
                co.loc[s, s] += 1
            # off-diagonals
            for i in range(len(smells)):
                for j in range(i + 1, len(smells)):
                    s1, s2 = smells[i], smells[j]
                    co.loc[s1, s2] += 1
                    co.loc[s2, s1] += 1

        self.smell_co_matrix = co
        return co

    def build_fragment_correlation(self, method: str = "pearson") -> pd.DataFrame:
        """
        Correlation between fragment columns across molecules.
        method ∈ {'pearson','kendall','spearman'}
        """
        if self.df is None:
            raise RuntimeError("No data loaded. Call .load(...) first.")
        corr = self.df[self.frag_cols].corr(method=method)
        self.frag_corr = corr
        return corr

    # ---------- Clustering (PCA + KMeans) ----------

    def cluster_smells(
        self,
        n_clusters: int = 4,
        scale: bool = True,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Cluster smells using their fragment-frequency vectors (from frag_smell_df).
        Returns a DataFrame indexed by smell with columns: PC1, PC2, Cluster.
        """
        if self.frag_smell_df is None:
            raise RuntimeError("Call .build_fragment_smell_matrix(...) first.")

        X = self.frag_smell_df.T  # smells x fragments
        X_mat = X.values

        if scale:
            scaler = StandardScaler()
            X_mat = scaler.fit_transform(X_mat)

        # Keep 2 PCs for visualization
        self.pca_model = PCA(n_components=2, random_state=random_state)
        X_pca = self.pca_model.fit_transform(X_mat)

        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        labels = self.kmeans_model.fit_predict(X_mat)

        out = pd.DataFrame(X_pca, index=X.index, columns=["PC1", "PC2"])
        out["Cluster"] = labels
        self.pca_df = out
        return out

    # ---------- Plotting (matplotlib only) ----------

    def _heatmap(
        self,
        data: pd.DataFrame,
        title: str,
        xlabel: str,
        ylabel: str,
        figsize: Tuple[int, int] = (12, 7),
        xtick_rotation: int = 90,
        ytick_rotation: int = 0,
        sort_rows_by_sum: bool = True,
        sort_cols_by_sum: bool = True,
        clip_top_k_rows: Optional[int] = None,
        clip_top_k_cols: Optional[int] = None,
        annotate: bool = False,
    ) -> None:
        mat = data.copy()
        if sort_rows_by_sum and mat.shape[0] > 1:
            mat = mat.loc[mat.sum(axis=1).sort_values(ascending=False).index]
        if sort_cols_by_sum and mat.shape[1] > 1:
            mat = mat[mat.sum(axis=0).sort_values(ascending=False).index]
        if clip_top_k_rows is not None:
            mat = mat.iloc[:clip_top_k_rows, :]
        if clip_top_k_cols is not None:
            mat = mat.iloc[:, :clip_top_k_cols]

        plt.figure(figsize=figsize)
        im = plt.imshow(mat.values, aspect="auto")
        plt.colorbar(im)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(ticks=np.arange(mat.shape[1]), labels=mat.columns, rotation=xtick_rotation)
        plt.yticks(ticks=np.arange(mat.shape[0]), labels=mat.index, rotation=ytick_rotation)

        if annotate and (mat.shape[0] * mat.shape[1] <= 2000):
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    plt.text(j, i, str(mat.iat[i, j]), ha="center", va="center")

        plt.tight_layout()
        plt.show()

    def plot_fragment_smell_heatmap(
        self,
        top_k_frags: Optional[int] = 40,
        top_k_smells: Optional[int] = None,
        annotate: bool = False,
    ) -> None:
        """
        Heatmap of fragments (rows) vs smells (cols).
        """
        if self.frag_smell_df is None:
            raise RuntimeError("Call .build_fragment_smell_matrix(...) first.")
        self._heatmap(
            self.frag_smell_df,
            title="Functional Groups vs Smell Classes",
            xlabel="Smell Class",
            ylabel="Functional Group",
            clip_top_k_rows=top_k_frags,
            clip_top_k_cols=top_k_smells,
            annotate=annotate,
        )

    def plot_smell_cooccurrence_heatmap(self, annotate: bool = False) -> None:
        """
        Heatmap of smell co-occurrence.
        """
        if self.smell_co_matrix is None:
            raise RuntimeError("Call .build_smell_cooccurrence() first.")
        self._heatmap(
            self.smell_co_matrix,
            title="Smell Co-occurrence Matrix",
            xlabel="Smell",
            ylabel="Smell",
            xtick_rotation=90,
            ytick_rotation=0,
            annotate=annotate,
        )

    def plot_fragment_correlation_heatmap(
        self,
        top_k_frags: Optional[int] = 40,
        annotate: bool = False,
    ) -> None:
        """
        Heatmap of fragment-fragment correlation.
        """
        if self.frag_corr is None:
            raise RuntimeError("Call .build_fragment_correlation() first.")
        self._heatmap(
            self.frag_corr,
            title="Correlation Between Functional Groups",
            xlabel="Functional Group",
            ylabel="Functional Group",
            clip_top_k_rows=top_k_frags,
            clip_top_k_cols=top_k_frags,
            annotate=annotate,
        )

    def plot_smell_clusters(
        self,
        label_points: bool = True,
        figsize: Tuple[int, int] = (8, 6),
    ) -> None:
        """
        Scatter plot of smells in PCA space, colored by cluster.
        """
        if self.pca_df is None or self.pca_model is None:
            raise RuntimeError("Call .cluster_smells(...) first.")

        plt.figure(figsize=figsize)
        clusters = sorted(self.pca_df["Cluster"].unique())
        for cl in clusters:
            sub = self.pca_df[self.pca_df["Cluster"] == cl]
            plt.scatter(sub["PC1"], sub["PC2"], label=f"Cluster {cl}", s=60)

        if label_points:
            for name, row in self.pca_df.iterrows():
                plt.text(row["PC1"] + 0.02, row["PC2"], name, fontsize=8)

        var = self.pca_model.explained_variance_ratio_
        plt.title("Smell Clusters Based on Functional Group Patterns")
        plt.xlabel(f"PC1 ({var[0]*100:.1f}% var)")
        plt.ylabel(f"PC2 ({var[1]*100:.1f}% var)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ---------- Exports ----------

    def export_tables(self, out_dir: str) -> None:
        """
        Save computed tables (if available) to CSV files.
        """
        os.makedirs(out_dir, exist_ok=True)
        if self.frag_smell_df is not None:
            self.frag_smell_df.to_csv(os.path.join(out_dir, "fragment_smell_matrix.csv"))
        if self.smell_co_matrix is not None:
            self.smell_co_matrix.to_csv(os.path.join(out_dir, "smell_cooccurrence.csv"))
        if self.frag_corr is not None:
            self.frag_corr.to_csv(os.path.join(out_dir, "fragment_correlation.csv"))

    # ---------- Convenience pipeline ----------

    def full_analysis(
        self,
        data: Union[str, pd.DataFrame],
        binary_frag_presence: bool = True,
        min_count: int = 0,
        n_clusters: int = 4,
        correlation_method: str = "pearson",
        plot: bool = True,
    ) -> None:
        """
        Run the full pipeline:
          load -> fragment-smell matrix -> co-occurrence -> fragment correlation -> clustering -> plots
        """
        self.load(data)
        self.build_fragment_smell_matrix(binary_frag_presence=binary_frag_presence, min_count=min_count)
        self.build_smell_cooccurrence()
        self.build_fragment_correlation(method=correlation_method)
        self.cluster_smells(n_clusters=n_clusters)

        if plot:
            self.plot_fragment_smell_heatmap()
            self.plot_smell_cooccurrence_heatmap()
            self.plot_fragment_correlation_heatmap()
            self.plot_smell_clusters()
