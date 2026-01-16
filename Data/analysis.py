# %%
import os
from typing import Optional, List, Tuple, Union, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib.colors import LogNorm
from itertools import combinations
import matplotlib as mpl
k = 1.25
mpl.rcParams.update({
    "figure.titlesize": 18 * k,
    "axes.titlesize": 18*k,
    "axes.labelsize": 16*k,
    "xtick.labelsize": 14*k,
    "ytick.labelsize": 14*k,
    "legend.fontsize": 14*k,
    "font.size": 14*k
})

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
        smell_separator: str = ";",
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

    def get_top_smells(self, n: int = 20) -> list:
        """Return top-N most frequent smells by occurrence count."""
        smell_counts = {}
        for smells in self.df[self.smell_col]:
            for s in smells:
                smell_counts[s] = smell_counts.get(s, 0) + 1
        return sorted(smell_counts, key=smell_counts.get, reverse=True)[:n]

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

    def build_fragment_cooccurrence(
            self,
            mode: str = "binary",  # "binary", "min", "product"
            min_total: int = 0,  # drop fragments whose total diag count < min_total
    ) -> pd.DataFrame:
        """
        Build a fragment x fragment co-occurrence matrix.
        - mode="binary": counts molecules where both fragments are present (off-diagonal += 1);
                         diagonal = # molecules where fragment is present
        - mode="min":    off-diagonal += min(count_i, count_j); diagonal += count_i
        - mode="product":off-diagonal += count_i*count_j; diagonal += count_i

        Returns: DataFrame (index=columns=fragment names) with integer counts.
        """
        if self.df is None:
            raise RuntimeError("Call .load(...) first.")
        if not self.frag_cols:
            raise RuntimeError("No fragment columns found.")

        frags = self.frag_cols
        M = pd.DataFrame(0, index=frags, columns=frags, dtype=float)

        for _, row in self.df.iterrows():
            # grab fragment counts for this molecule
            counts = row[frags].astype(int)
            present = counts[counts > 0]

            if present.empty:
                continue

            # diagonal updates
            if mode == "binary":
                for f in present.index:
                    M.loc[f, f] += 1
            elif mode == "min" or mode == "product":
                for f, c in present.items():
                    M.loc[f, f] += c
            else:
                raise ValueError("mode must be 'binary', 'min', or 'product'.")

            # off-diagonals
            idxs = present.index.tolist()
            for f1, f2 in combinations(idxs, 2):
                if mode == "binary":
                    inc = 1
                elif mode == "min":
                    inc = min(int(present[f1]), int(present[f2]))
                else:  # product
                    inc = int(present[f1]) * int(present[f2])
                M.loc[f1, f2] += inc
                M.loc[f2, f1] += inc

        # optional: drop very rare fragments by diagonal (total presence)
        if min_total > 0:
            keep = M.index[M.values.diagonal() >= min_total]
            M = M.loc[keep, keep]

        # store for plotting
        self.frag_co_matrix = M.astype(float)
        return self.frag_co_matrix

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
        figsize: Tuple[int, int] = (20, 20),
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
            top_k_frags: int = 100,
            top_k_smells: int = 80,
            scale: str = "log",  # "log", "percentile", "normalize", "linear"
            normalize: str = "by_smell",  # used only if scale == "normalize"; "by_smell" or "by_fragment"
            log_pseudocount: float = 0.0,  # added before LogNorm
            exclude_smells: list = None,
            annotate: bool = False,
            hide_zero_rows: bool = True,
    ) -> None:
        """
        Heatmap of fragments (rows) vs smells (cols) with robust scaling/normalization.

        - top_k_*: limit rows/cols for readability
        - scale="log": uses LogNorm with +pseudocount so zeros are visible
        - scale="percentile": 5–95% clipping for contrast
        - scale="normalize": per-column ("by_smell") or per-row ("by_fragment") 0–1 scaling
        - exclude_smells: remove dominant or unwanted smells from columns
        """

        if self.frag_smell_df is None:
            raise RuntimeError("Call .build_fragment_smell_matrix(...) first.")

        # --- start from the computed matrix
        mat = self.frag_smell_df.copy()  # rows: fragments, cols: smells

        # Optionally exclude smells
        if exclude_smells:
            keep_cols = [c for c in mat.columns if c not in set(exclude_smells)]
            mat = mat.loc[:, keep_cols]

        # Limit to top smells by frequency in the original data
        if top_k_smells is not None:
            # get_top_smells uses the raw dataset; intersect with existing columns
            top_smells = [s for s in self.get_top_smells(top_k_smells) if s in mat.columns]
            if top_smells:
                mat = mat.loc[:, top_smells]

        # Drop zero-only rows (optional; helps declutter)
        if hide_zero_rows:
            nonzero = (mat.sum(axis=1) > 0)
            mat = mat.loc[nonzero]

        # If too many fragments, keep the most informative (highest total across smells)
        if top_k_frags is not None and mat.shape[0] > top_k_frags:
            mat = mat.loc[mat.sum(axis=1).sort_values(ascending=False).head(top_k_frags).index]

        # --- choose scaling
        title_suffix = f"{scale}"
        cmap = "plasma"

        if scale == "log":
            # + pseudocount to avoid log(0)
            plot_vals = mat.values.astype(float) + float(log_pseudocount)
            # vmin: smallest positive
            positive = plot_vals[plot_vals > 0]
            vmin = positive.min() if positive.size else 1e-6
            vmax = np.nanmax(plot_vals) if np.isfinite(np.nanmax(plot_vals)) else vmin * 10

            plt.figure(figsize=(15, 15))
            im = plt.imshow(plot_vals, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10)))
            cbar = plt.colorbar(im)
            cbar.set_label(f"Count (log)")

        elif scale == "percentile":
            vals = mat.values.flatten()
            low, high = np.percentile(vals, [5, 95]) if vals.size else (0, 1)
            if low == high:
                low, high = 0, max(high, 1)
            plt.figure(figsize=(12, 7))
            im = plt.imshow(mat.values, cmap=cmap, vmin=low, vmax=high)
            plt.colorbar(im)

        elif scale == "normalize":
            # column-wise (per smell) or row-wise (per fragment) normalization to [0,1]
            m = mat.astype(float).copy()
            if normalize == "by_smell":  # divide each column by its max
                col_max = m.max(axis=0)
                m = m.divide(col_max.replace(0, np.nan), axis=1).fillna(0.0)
                title_suffix += " (by_smell)"
            elif normalize == "by_fragment":  # divide each row by its max
                row_max = m.max(axis=1)
                m = m.divide(row_max.replace(0, np.nan), axis=0).fillna(0.0)
                title_suffix += " (by_fragment)"
            else:
                raise ValueError("normalize must be 'by_smell' or 'by_fragment' when scale='normalize'.")

            plt.figure(figsize=(15, 15))
            im = plt.imshow(m.values, cmap="viridis", vmin=0, vmax=1)
            plt.colorbar(im, label="Normalized (0–1)")
            mat = m  # for labels/annotations below

        else:  # linear
            vmin, vmax = mat.values.min(), mat.values.max()
            if vmin == vmax:
                vmin, vmax = 0, vmin + 1
            plt.figure(figsize=(12, 7))
            im = plt.imshow(mat.values, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(im)

        # --- axes, labels, annotations
        plt.title(f"Fragment and Smell Label co-occurence ({scale} scale)")
        plt.xlabel(f"Smell Label (Top {top_k_smells})")
        plt.ylabel(f"Fragment (Top {top_k_frags})")
        plt.xticks(ticks=np.arange(mat.shape[1]), labels=mat.columns, rotation=90)
        plt.yticks(ticks=np.arange(mat.shape[0]), labels=mat.index)

        if annotate and (mat.shape[0] * mat.shape[1] <= 2000):
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    val = mat.values[i, j]
                    # show integers where sensible; otherwise 2 decimals
                    text = f"{int(val)}" if float(val).is_integer() else f"{val:.2f}"
                    plt.text(j, i, text, ha="center", va="center", fontsize=7)

        plt.tight_layout()
        plt.savefig(
            "figures/fragment_smell_heatmap.pdf",
            dpi=300,
            bbox_inches="tight"
        )
        plt.show()

    def plot_smell_cooccurrence_heatmap(
            self,
            top_k_smells: int = 80,
            hide_diagonal: bool = True,
            scale: str = "log",  # "percentile", "log", "normalize", or "linear"
            annotate: bool = False,
    ) -> None:
        """
        Heatmap of smell co-occurrence with adjustable color scaling.
        """
        if self.smell_co_matrix is None:
            raise RuntimeError("Call .build_smell_cooccurrence() first.")

        top_smells = self.get_top_smells(top_k_smells)
        mat = self.smell_co_matrix.loc[top_smells, top_smells].copy()

        if hide_diagonal:
            np.fill_diagonal(mat.values, 0)

        if scale == "normalize":
            mat = mat.div(np.diag(mat), axis=1).fillna(0)
            vmin, vmax = 0, 1
            cmap = "viridis"
        elif scale == "log":
            from matplotlib.colors import LogNorm
            mat[mat <= 0] = np.nan
            norm = LogNorm()
            cmap = "plasma"
            vmin = vmax = None
        elif scale == "percentile":
            vals = mat.values.flatten()
            low, high = np.percentile(vals, [5, 95])
            vmin, vmax = low, high
            cmap = "plasma"
        else:  # linear
            vmin, vmax = mat.values.min(), mat.values.max()
            cmap = "plasma"

        plt.figure(figsize=(15, 15))
        if scale == "log":
            im = plt.imshow(mat.values, cmap=cmap, norm=norm)
        else:
            im = plt.imshow(mat.values, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im)
        plt.title(f"Smell Co-occurrence ({scale} scale)")
        plt.xlabel(f"Smell Label (Top {top_k_smells})")
        plt.ylabel(f"Smell Label (Top {top_k_smells})")
        plt.xticks(ticks=np.arange(mat.shape[1]), labels=mat.columns, rotation=90)
        plt.yticks(ticks=np.arange(mat.shape[0]), labels=mat.index)
        plt.tight_layout()
        plt.savefig(
            "figures/smell_cooccurrence_heatmap.pdf",
            dpi=300,
            bbox_inches="tight"
        )
        plt.show()

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    def plot_fragment_correlation_heatmap(
            self,
            top_k_frags: int = 90,
            scale: str = "log",  # "log", "percentile", "normalize", "linear"
            log_pseudocount: float = 0.0,  # avoid log(0)
            hide_diagonal: bool = True,
            annotate: bool = False,
    ) -> None:
        """
        Heatmap of fragment-fragment correlation (or co-occurrence) with log scaling options.
        """

        if self.frag_corr is None:
            raise RuntimeError("Call .build_fragment_correlation() first.")

        # Limit to top fragments (by variance or absolute correlation magnitude)
        mat = self.frag_corr.copy()
        if mat.shape[0] > top_k_frags:
            # pick top fragments with highest mean absolute correlation
            scores = mat.abs().mean(axis=1).sort_values(ascending=False)
            top = scores.head(top_k_frags).index
            mat = mat.loc[top, top]

        # Optionally hide diagonal (always 1 for correlation)
        if hide_diagonal:
            np.fill_diagonal(mat.values, 0)

        cmap = "plasma"
        title_suffix = scale

        if scale == "log":
            # shift to positive domain: correlation in [-1,1] → take abs, + pseudocount
            plot_vals = np.abs(mat.values) + log_pseudocount
            positive = plot_vals[plot_vals > 0]
            vmin = positive.min() if positive.size else 1e-6
            vmax = np.nanmax(plot_vals) if np.isfinite(np.nanmax(plot_vals)) else vmin * 10

            plt.figure(figsize=(10, 8))
            im = plt.imshow(plot_vals, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10)))
            plt.colorbar(im, label=f"|Correlation| (log)")

        elif scale == "percentile":
            vals = mat.values.flatten()
            low, high = np.percentile(vals, [5, 95])
            if low == high:
                low, high = 0, max(1, high)
            plt.figure(figsize=(10, 8))
            im = plt.imshow(mat.values, cmap=cmap, vmin=low, vmax=high)
            plt.colorbar(im, label="Correlation")

        elif scale == "normalize":
            # rescale each column to 0–1 range
            m = mat.astype(float)
            m = (m - m.min()) / (m.max() - m.min() + 1e-9)
            plt.figure(figsize=(10, 8))
            im = plt.imshow(m.values, cmap="viridis", vmin=0, vmax=1)
            plt.colorbar(im, label="Normalized 0–1")
            mat = m
            title_suffix += " (normalized)"

        else:  # linear
            vmin, vmax = mat.values.min(), mat.values.max()
            if vmin == vmax:
                vmin, vmax = 0, vmin + 1
            plt.figure(figsize=(15, 15))
            im = plt.imshow(mat.values, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(im, label="Correlation")

        plt.title(f"Fragment–Fragment correlation ((Top {top_k_frags}, {scale} scale))")
        plt.xlabel("Functional Group")
        plt.ylabel("Functional Group")
        plt.xticks(ticks=np.arange(mat.shape[1]), labels=mat.columns, rotation=90)
        plt.yticks(ticks=np.arange(mat.shape[0]), labels=mat.index)

        if annotate and mat.shape[0] <= 30:
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    plt.text(j, i, f"{mat.values[i, j]:.2f}", ha="center", va="center", fontsize=7)

        plt.tight_layout()
        plt.savefig(
            "figures/fragment_correlation_heatmap.pdf",
            dpi=300,
            bbox_inches="tight"
        )
        plt.show()

    def plot_fragment_cooccurrence_heatmap(
            self,
            top_k_frags: int = 90,
            scale: str = "log",  # "log", "percentile", "normalize", "linear"
            log_pseudocount: float = 0.0,  # added before LogNorm so zeros show up
            hide_diagonal: bool = True,  # hide big diagonals to emphasize co-occur
            min_co: float = 0.0,  # zero-out tiny co-occurrences (< min_co)
            annotate: bool = False,
    ) -> None:
        """
        Heatmap of fragment–fragment *co-occurrence counts* (not correlation).
        Supports log scaling with a pseudocount.
        """
        if not hasattr(self, "frag_co_matrix") or self.frag_co_matrix is None:
            raise RuntimeError("Call .build_fragment_cooccurrence(...) first.")

        mat = self.frag_co_matrix.copy()

        # keep most 'connected' fragments to reduce clutter
        if mat.shape[0] > top_k_frags:
            # score by total off-diagonal strength
            offdiag_sum = mat.sum(axis=1) - np.diag(mat.values)
            top = offdiag_sum.sort_values(ascending=False).head(top_k_frags).index
            mat = mat.loc[top, top]

        # threshold tiny counts (optional)
        if min_co > 0:
            mat = mat.mask(mat < min_co, 0)

        # optionally remove diagonal
        if hide_diagonal:
            np.fill_diagonal(mat.values, 0)

        cmap = "plasma"
        title_suffix = scale

        if scale == "log":
            vals = mat.values.astype(float) + float(log_pseudocount)
            # guard rails for LogNorm
            positive = vals[vals > 0]
            vmin = positive.min() if positive.size else 1e-6
            vmax = np.nanmax(vals) if np.isfinite(np.nanmax(vals)) else vmin * 10

            plt.figure(figsize=(15, 15))
            im = plt.imshow(vals, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10)))
            cbar = plt.colorbar(im)
            cbar.set_label(f"Count (log)")

        elif scale == "percentile":
            flat = mat.values.flatten()
            low, high = np.percentile(flat, [5, 95]) if flat.size else (0, 1)
            if low == high:
                low, high = 0, max(1, high)
            plt.figure(figsize=(10, 8))
            im = plt.imshow(mat.values, cmap=cmap, vmin=low, vmax=high)
            plt.colorbar(im, label="Co-occurrence")

        elif scale == "normalize":
            # 0–1 normalization by global min/max (on selected submatrix)
            m = mat.astype(float)
            mn, mx = m.values.min(), m.values.max()
            if mn == mx:
                mn, mx = 0, mn + 1
            m = (m - mn) / (mx - mn)
            plt.figure(figsize=(10, 8))
            im = plt.imshow(m.values, cmap="viridis", vmin=0, vmax=1)
            plt.colorbar(im, label="Normalized (0–1)")
            mat = m
            title_suffix += " (normalized)"

        else:  # linear
            vmin, vmax = mat.values.min(), mat.values.max()
            if vmin == vmax:
                vmin, vmax = 0, vmin + 1
            plt.figure(figsize=(15, 15))
            im = plt.imshow(mat.values, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(im, label="Co-occurrence")

        plt.title(f"Fragment Co-occurrence ({scale} scale)")
        plt.xlabel(f"Fragment (Top {top_k_frags})")
        plt.ylabel(f"Fragment (Top {top_k_frags})")
        plt.xticks(ticks=np.arange(mat.shape[1]), labels=mat.columns, rotation=90)
        plt.yticks(ticks=np.arange(mat.shape[0]), labels=mat.index)

        if annotate and mat.shape[0] <= 30:
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    plt.text(j, i, f"{mat.values[i, j]:.0f}", ha="center", va="center", fontsize=7)

        plt.tight_layout()
        plt.savefig(
            "figures/fragment_cooccurrence_heatmap.pdf",
            dpi=300,
            bbox_inches="tight"
        )
        plt.show()

    def print_summary_stats(self, top_n: int = 10) -> None:
        """
        Print useful statistics about fragments, smells, and co-occurrences.
        Requires that .load() and .build_fragment_smell_matrix() were run.
        Optionally includes fragment–fragment co-occurrence stats if available.
        """
        if self.df is None:
            raise RuntimeError("Load your dataset first with .load(...)")

        print(" === DATASET SUMMARY ===")
        print(f"Total molecules: {len(self.df):,}")

        # --- Fragments ---
        if self.frag_cols:
            frag_counts = (self.df[self.frag_cols] > 0).sum()
            n_frags = len(self.frag_cols)
            avg_frags = frag_counts.sum() / len(self.df)
            print(f"Total fragment types: {n_frags:,}")
            print(f"Average fragments per molecule: {avg_frags:.2f}")
            print("\nTop fragments:")
            print(frag_counts.sort_values(ascending=False).head(top_n))

            print(f"Total fragment types: {n_frags:,}")
            print(f"Average fragments per molecule: {avg_frags:.2f}")
            print("\nBottom fragments:")
            print(frag_counts.sort_values(ascending=True).head(top_n))
        else:
            print("No fragment columns detected.")

        # --- Smells ---
        if self.smell_col in self.df.columns:
            smells_flat = [s for lst in self.df[self.smell_col] for s in lst]
            if len(smells_flat) == 0:
                print("\nNo smell data found.")
            else:
                smell_counts = pd.Series(smells_flat).value_counts()
                avg_smells = smell_counts.sum() / len(self.df)
                print(f"\nTotal unique smell labels: {len(smell_counts):,}")
                print(f"Average smells per molecule: {avg_smells:.2f}")
                print("\nTop smell labels:")
                print(smell_counts.head(top_n))
                print("\nBottom smell labels:")
                print(smell_counts.tail(top_n))
        else:
            print("\nNo smell column found.")

        # --- Fragment–Smell matrix ---
        if hasattr(self, "frag_smell_df") and self.frag_smell_df is not None:
            total_links = int(self.frag_smell_df.values.sum())
            print(f"\nFragment–Smell matrix: {self.frag_smell_df.shape[0]} fragments × "
                  f"{self.frag_smell_df.shape[1]} smells "
                  f"({total_links:,} total fragment–smell links)")
            # Show top fragments associated with most smells
            frag_link_counts = (self.frag_smell_df > 0).sum(axis=1)
            print("\nFragments associated with most smell classes:")
            print(frag_link_counts.sort_values(ascending=False).head(top_n))

        # --- Fragment–Fragment co-occurrence ---
        if hasattr(self, "frag_co_matrix") and self.frag_co_matrix is not None:
            n = self.frag_co_matrix.shape[0]
            total_pairs = int(self.frag_co_matrix.values.sum() / 2)
            print(f"\nFragment–Fragment co-occurrence matrix: {n}×{n}, "
                  f"{total_pairs:,} total co-occurrence counts")
            offdiag = self.frag_co_matrix.copy()
            np.fill_diagonal(offdiag.values, 0)
            top_pairs = (
                offdiag.stack()
                .sort_values(ascending=False)
                .reset_index(name="count")
                .head(top_n)
            )
            print("\nMost frequent fragment pairs:")
            print(top_pairs)
        else:
            print("\nNo fragment–fragment co-occurrence matrix built yet.")

        print("\n✅ Summary complete.\n")

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
        self.build_fragment_cooccurrence()
        self.build_fragment_correlation(method=correlation_method)
        self.cluster_smells(n_clusters=n_clusters)

        if plot:
            self.plot_fragment_smell_heatmap()
            self.plot_smell_cooccurrence_heatmap()
            self.plot_fragment_cooccurrence_heatmap()
            self.plot_fragment_correlation_heatmap()
            self.plot_smell_clusters()
            self.print_summary_stats()



TARGET_COLUMN = [
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

analyzer = SmellFragmentAnalyzer(smell_col="descriptors", frag_prefix="fr_")
analyzer.load("Data/smiles_with_func_groups.csv")

analyzer.build_fragment_smell_matrix(binary_frag_presence=True)
analyzer.build_smell_cooccurrence()
analyzer.build_fragment_cooccurrence(mode="binary")
analyzer.build_fragment_correlation()



analyzer.plot_fragment_smell_heatmap(
    top_k_frags=30,
    top_k_smells=30,
    scale="log",
    log_pseudocount=1
)

analyzer.plot_smell_cooccurrence_heatmap(
    top_k_smells=30,
    scale="log"
)

analyzer.plot_fragment_cooccurrence_heatmap(
    top_k_frags=30,
    scale="log",
    log_pseudocount=1,
    hide_diagonal=True
)

analyzer.print_summary_stats(top_n=10)
