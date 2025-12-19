from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from chemprop import data, featurizers


def df_to_datapoints(
    df_input: pd.DataFrame,
    smiles_column: str,
    target_columns: Sequence[str] | None
) -> Tuple[List[data.MoleculeDatapoint], np.ndarray]:
    """
    Convert a dataframe to Chemprop datapoints.
    If target_columns is None, create datapoints without labels (inference mode).
    """
    smis = df_input[smiles_column].tolist()
    if target_columns is None:
        datapoints = [data.MoleculeDatapoint.from_smi(smi) for smi in smis]
    else:
        ys = df_input.loc[:, target_columns].to_numpy()
        datapoints = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)]
    mols = [d.mol for d in datapoints]
    molecule_featurizer = featurizers.MorganCountFeaturizer()
    extra_mol_features = np.array([molecule_featurizer(mol) for mol in mols])
    return datapoints, extra_mol_features