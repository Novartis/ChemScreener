from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from sklearn.metrics import roc_auc_score


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_smiles(smiles: str) -> bool:
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


def prop_to_logits(prob_N_K: torch.Tensor) -> torch.Tensor:
    prob_N_K = torch.clip(prob_N_K, 1e-6, 1 - 1e-6)
    pos_logit = torch.logit(prob_N_K)
    neg_logit = torch.logit(1 - prob_N_K)
    logits = torch.stack([pos_logit, neg_logit], dim=2)
    return logits


def logit_mean(logits_N_K_C: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    return torch.logsumexp(logits_N_K_C, dim=dim, keepdim=keepdim) - math.log(logits_N_K_C.shape[dim])


def entropy(logits_N_K_C: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    probs = torch.exp(logits_N_K_C)
    return -(probs * logits_N_K_C).double().sum(dim=dim, keepdim=keepdim)


def mean_sample_entropy(logits_N_K_C: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    sample_entropies_N_K = entropy(logits_N_K_C, dim=dim, keepdim=keepdim)
    return sample_entropies_N_K.mean(dim=1)


def mutual_information(logits_N_K_C: torch.Tensor) -> torch.Tensor:
    entropy_mean_N = mean_sample_entropy(logits_N_K_C)
    mean_entropy_N = entropy(logit_mean(logits_N_K_C, dim=1), dim=-1)
    return mean_entropy_N - entropy_mean_N


def get_rank(tensor: torch.Tensor) -> torch.Tensor:
    sorted_indices = torch.argsort(tensor, dim=0)
    ranks = torch.zeros_like(tensor, dtype=torch.long)
    ranks.scatter_(0, sorted_indices, torch.arange(tensor.size(0)).unsqueeze(1).expand_as(tensor))
    return ranks


def cal_auc(df_ref: pd.DataFrame, target_columns: Sequence[str], preds_ref: torch.Tensor) -> List[float]:
    auc = []
    probs = preds_ref.detach().cpu().numpy()
    for i, col in enumerate(target_columns):
        y = df_ref[col].astype(int).to_numpy()
        try:
            auc.append(roc_auc_score(y, probs[:, i]))
        except ValueError:
            auc.append(float("nan"))
    return auc


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2))