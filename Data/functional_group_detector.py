from rdkit import Chem
from rdkit.Chem import Fragments
import pandas as pd

class FunctionalGroupDetector:
    """
    Detects RDKit-defined functional groups in molecules from SMILES strings.
    """

    def __init__(self, smiles_column: str = "nonStereoSMILES"):
        self.smiles_column = smiles_column
        self.frag_funcs = [f for f in dir(Fragments) if f.startswith("fr_")]

    def _smiles_to_mol(self, smiles: str):
        """Convert a SMILES string to an RDKit Mol object."""
        try:
            return Chem.MolFromSmiles(smiles) if pd.notnull(smiles) else None
        except Exception:
            return None

    def _get_functional_groups(self, mol):
        """Compute all functional group counts for a molecule."""
        if mol is None:
            return {}
        result = {}
        for func in self.frag_funcs:
            frag_func = getattr(Fragments, func)
            result[func] = frag_func(mol)
        return result

    def detect(self, df: pd.DataFrame, expand: bool = True) -> pd.DataFrame:
        """
        Detects functional groups in all molecules of the dataframe.

        Args:
            df (pd.DataFrame): Input dataframe containing a SMILES column.
            expand (bool): If True, expand fragments into individual columns.
                           If False, keep as a single 'functional_groups' column (dict).

        Returns:
            pd.DataFrame: Updated dataframe with functional group information.
        """
        df = df.copy()
        df["mol"] = df[self.smiles_column].apply(self._smiles_to_mol)
        df["functional_groups"] = df["mol"].apply(self._get_functional_groups)

        if expand:
            frag_df = df["functional_groups"].apply(pd.Series)
            df = pd.concat([df.drop(columns=["functional_groups", "mol"]), frag_df], axis=1)
        else:
            df.drop(columns=["mol"], inplace=True)

        return df

    def detect_from_csv(self, input_csv: str, output_csv: str = None, expand: bool = True) -> pd.DataFrame:
        """
        Load molecules from CSV, detect functional groups, and optionally save.

        Args:
            input_csv (str): Path to input CSV file (must contain SMILES column).
            output_csv (str): Optional path to save results.
            expand (bool): Whether to expand functional groups into columns.

        Returns:
            pd.DataFrame: Dataframe with detected functional groups.
        """
        df = pd.read_csv(input_csv)
        df_out = self.detect(df, expand=expand)
        print(df_out.head())
        if output_csv:
            df_out.to_csv(output_csv, index=False)
        return df_out
