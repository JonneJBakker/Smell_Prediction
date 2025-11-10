# pip install rdkit-pypi scikit-learn pandas numpy matplotlib
from typing import List, Optional, Tuple, Dict, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class RDKitPCA:
    """
    Build numerical features from SMILES using RDKit, then run PCA.

    Features:
      - RDKit 'Descriptors' (continuous/numeric)
      - Optional Morgan fingerprint counts (folded to nBits)
      - Feature cleaning: drop constant/NaN columns
      - Scaling (StandardScaler) + PCA
      - Scree + 2D scores plot
      - Loadings table for interpretation

    Parameters
    ----------
    smiles_col : str
        Column name containing SMILES.
    add_morgan : bool
        If True, appends folded Morgan fingerprint counts as features.
    morgan_radius : int
        Radius for Morgan fingerprints.
    morgan_nbits : int
        Number of bits for folded fingerprint.
    """

    def __init__(
        self,
        smiles_col: str = "smiles",
        add_morgan: bool = False,
        morgan_radius: int = 2,
        morgan_nbits: int = 2048,
    ):
        self.smiles_col = smiles_col
        self.add_morgan = add_morgan
        self.morgan_radius = morgan_radius
        self.morgan_nbits = morgan_nbits

        # Will be set after fit/transform
        self.feature_names_: List[str] = []
        self.scaler_: Optional[StandardScaler] = None
        self.pca_: Optional[PCA] = None
        self.loadings_: Optional[pd.DataFrame] = None
        self.scores_: Optional[pd.DataFrame] = None
        self.cleaned_X_: Optional[pd.DataFrame] = None
        self.valid_idx_: Optional[pd.Index] = None

    # ---------- Feature construction ----------

    def _smiles_to_mol(self, s: str):
        try:
            return Chem.MolFromSmiles(s) if pd.notnull(s) else None
        except Exception:
            return None

    def _rdkit_descriptor_dict(self, mol: Chem.Mol) -> Dict[str, float]:
        # Pull all descriptors from rdkit.Chem.Descriptors
        out = {}
        for name, func in Descriptors.descList:
            try:
                out[name] = float(func(mol)) if mol is not None else np.nan
            except Exception:
                out[name] = np.nan
        return out

    def _morgan_counts(self, mol: Chem.Mol) -> Dict[str, int]:
        """
        Return a dict of bit counts for a folded Morgan fingerprint.
        Using counts (not bits) gives some gradation for PCA.
        """
        if mol is None:
            return {f"Morgan{self.morgan_radius}_{i}": 0 for i in range(self.morgan_nbits)}
        info = {}
        fp = AllChem.GetHashedMorganFingerprint(
            mol,
            radius=self.morgan_radius,
            nBits=self.morgan_nbits,
            useCounts=True,
            bitInfo=info,
        )
        # Convert sparse to dense dict
        d = {f"Morgan{self.morgan_radius}_{i}": 0 for i in range(self.morgan_nbits)}
        on_bits = fp.GetNonzeroElements()  # dict: bit->count
        for bit, cnt in on_bits.items():
            key = f"Morgan{self.morgan_radius}_{bit}"
            d[key] = int(cnt)
        return d

    def featurize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build numeric features for each SMILES row.
        Returns a DataFrame aligned to input index, with NaNs where mol parse failed.
        """
        mols = df[self.smiles_col].apply(self._smiles_to_mol)
        desc_rows = [self._rdkit_descriptor_dict(m) for m in mols]

        X = pd.DataFrame(desc_rows, index=df.index)

        if self.add_morgan:
            m_rows = [self._morgan_counts(m) for m in mols]
            M = pd.DataFrame(m_rows, index=df.index)
            X = pd.concat([X, M], axis=1)

        return X

    # ---------- Cleaning & PCA ----------

    def _clean_features(self, X: pd.DataFrame) -> pd.DataFrame:
        # Drop columns entirely NaN
        X = X.dropna(axis=1, how="all")
        # Fill remaining NaNs with column medians
        X = X.apply(lambda c: c.fillna(c.median()), axis=0)
        # Drop constant columns (zero variance)
        nunique = X.nunique(dropna=False)
        X = X.loc[:, nunique > 1]
        return X

    def fit_transform(
        self,
        df: pd.DataFrame,
        n_components: int = 10,
        scale: bool = True,
        keep_valid_only: bool = True,
        extra_meta_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Build features, clean them, then fit PCA and return PCA scores.

        Returns a DataFrame with columns ["PC1","PC2",...,"PCk"] and carries over
        any extra_meta_cols from the original df (for plotting labels/hover).
        """
        # Build features
        X = self.featurize(df)

        # Optionally drop rows where everything is NaN (failed mol)
        if keep_valid_only:
            valid_mask = ~X.isna().all(axis=1)
            self.valid_idx_ = df.index[valid_mask]
            X = X.loc[self.valid_idx_]
            df_meta = df.loc[self.valid_idx_]
        else:
            self.valid_idx_ = df.index
            df_meta = df

        # Clean features (NaN/constant)
        X = self._clean_features(X)
        self.cleaned_X_ = X.copy()
        self.feature_names_ = list(X.columns)

        # Scale
        if scale:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X.values)
        else:
            self.scaler_ = None
            X_scaled = X.values

        # PCA
        self.pca_ = PCA(n_components=n_components, random_state=42)
        Z = self.pca_.fit_transform(X_scaled)

        # Scores
        score_cols = [f"PC{i+1}" for i in range(self.pca_.n_components_)]
        scores = pd.DataFrame(Z, index=self.valid_idx_, columns=score_cols)

        # Attach meta columns if requested
        if extra_meta_cols:
            for c in extra_meta_cols:
                if c in df_meta.columns:
                    scores[c] = df_meta[c].values

        self.scores_ = scores

        # Loadings (features x PCs)
        loadings = pd.DataFrame(
            self.pca_.components_.T,
            index=self.feature_names_,
            columns=score_cols,
        )
        self.loadings_ = loadings

        return scores

    # ---------- Reporting ----------

    def explained_variance(self) -> pd.DataFrame:
        if self.pca_ is None:
            raise RuntimeError("Run fit_transform first.")
        return pd.DataFrame({
            "PC": [f"PC{i+1}" for i in range(self.pca_.n_components_)],
            "ExplainedVarianceRatio": self.pca_.explained_variance_ratio_
        })

    def top_loadings(self, pc: int = 1, top_k: int = 15) -> pd.DataFrame:
        """
        Return top |loadings| features for a given PC (1-based).
        """
        if self.loadings_ is None:
            raise RuntimeError("Run fit_transform first.")
        col = f"PC{pc}"
        vec = self.loadings_[col].abs().sort_values(ascending=False).head(top_k).index
        out = self.loadings_.loc[vec, [col]].sort_values(by=col, key=np.abs, ascending=False)
        return out

    # ---------- Plots (matplotlib; single-plot, no custom colors) ----------

    def plot_scree(self) -> None:
        if self.pca_ is None:
            raise RuntimeError("Run fit_transform first.")
        evr = self.pca_.explained_variance_ratio_
        plt.figure(figsize=(6,4))
        plt.plot(range(1, len(evr)+1), evr, marker="o")
        plt.title("PCA Scree Plot")
        plt.xlabel("Principal Component")
        plt.ylabel("Explained Variance Ratio")
        plt.tight_layout()
        plt.show()

    def plot_scores(
        self,
        pcx: int = 1,
        pcy: int = 2,
        label_col: Optional[str] = None,
        s: int = 40,
    ) -> None:
        """
        Scatter of PCx vs PCy. If label_col provided, add text labels for points.
        (No explicit colors are set; matplotlib chooses defaults.)
        """
        if self.scores_ is None:
            raise RuntimeError("Run fit_transform first.")
        x = f"PC{pcx}"
        y = f"PC{pcy}"
        plt.figure(figsize=(6,5))
        plt.scatter(self.scores_[x], self.scores_[y], s=s)
        if label_col and label_col in self.scores_.columns:
            for idx, row in self.scores_.iterrows():
                plt.text(row[x], row[y], str(row[label_col]), fontsize=7)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(f"{x} vs {y}")
        plt.tight_layout()
        plt.show()
