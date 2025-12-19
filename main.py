from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from modeling import chemprop_ensemble
from utils import (
    set_seed,
    validate_smiles,
    prop_to_logits,
    mutual_information,
    get_rank,
    save_json,
)


def acquire_new(
    stacked_preds: torch.Tensor,
    vars_preds: torch.Tensor,
    mode: str,
    n_acquire: int = 200
) -> np.ndarray:
    """
    Select indices for acquisition based on an acquisition mode.
    - stacked_preds: [K, N, C] probabilities for positive class.
    - vars_preds: [1, N, C] variance across ensemble members.
    Modes: MI, Exploitative, Balanced_Ranking
    """
    stacked_preds = stacked_preds.permute(1, 0, 2)  # [N, K, C]
    vars_preds = vars_preds.permute(1, 0, 2)        # [N, 1, C]
    N, K, C = stacked_preds.size()

    if mode == "MI":
        score = torch.zeros(N, dtype=torch.float32)
        for i in range(C):
            logits = prop_to_logits(stacked_preds[:, :, i])
            score += mutual_information(logits).float()
        sorted_indices = torch.argsort(score, dim=0, descending=True)
    elif mode == "Exploitative":
        avg_rank = get_rank(stacked_preds.mean(1)).sum(1)
        sorted_indices = torch.argsort(avg_rank, dim=0, descending=True)
    else:
        rank_pred = get_rank(stacked_preds.mean(1))
        rank_uncertainty = get_rank(vars_preds[:, 0, :])
        avg_rank = (rank_pred + rank_uncertainty).sum(1)
        sorted_indices = torch.argsort(avg_rank, dim=0, descending=True)

    n_take = min(int(n_acquire), N)
    pick = sorted_indices[:n_take].cpu().numpy()
    mask = np.zeros(N, dtype=bool)
    mask[pick] = True
    return mask


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Active learning with Chemprop ensembles")
    p.add_argument("--assay_csv", type=str, required=True, help="Path to labeled assay CSV (with targets)")
    p.add_argument("--test_csv", type=str, required=True, help="Path to unlabeled test/library CSV (SMILES only required)")
    p.add_argument("--output_dir", type=str, required=True, help="Directory to write results and checkpoints")
    p.add_argument("--smiles_column", type=str, default="smiles", help="Name of the SMILES column")
    p.add_argument("--target_columns", type=str, nargs="+", required=True, help="Target column names (binary)")
    p.add_argument("--split_col", type=str, default=None, help="Name of split column in assay CSV if present")
    p.add_argument("--split_train", type=str, default=None, help="Split value for train rows in assay CSV")
    p.add_argument("--split_val", type=str, default=None, help="Split value for validation rows in assay CSV (optional)")
    p.add_argument("--n_ensemble", type=int, default=10, help="Number of ensembles")
    p.add_argument("--batch_size", type=int, default=512, help="Batch size")
    p.add_argument("--max_epochs", type=int, default=200, help="Max training epochs")
    p.add_argument("--accelerator", type=str, default="auto", help="Trainer accelerator (auto, cpu, gpu)")
    p.add_argument("--devices", type=str, default="1", help="Devices (e.g., 1, or 0,1 for multi-GPU)")
    p.add_argument("--mode", type=str, default="Balanced_Ranking", choices=["MI", "Exploitative", "Balanced_Ranking"])
    p.add_argument("--n_acquire", type=int, default=300, help="Number to acquire in the single pass")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--run_id", type=int, default=0, help="Run identifier for output file names")
    return p


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_json(out_dir / "config.json", vars(args))

    df_assay = pd.read_csv(args.assay_csv).drop_duplicates(subset=[args.smiles_column])
    df_test = pd.read_csv(args.test_csv).drop_duplicates(subset=[args.smiles_column])

    if args.split_col and args.split_train:
        df_assay = df_assay.rename(columns={args.split_col: "split"})
        train_mask = df_assay["split"] == args.split_train
        val_mask = df_assay["split"] == args.split_val if args.split_val else None

        cols_train_val = [args.smiles_column] + args.target_columns
        df_train = df_assay.loc[train_mask, cols_train_val].reset_index(drop=True)
        df_val = None
        if val_mask is not None:
            df_val = df_assay.loc[val_mask, cols_train_val].reset_index(drop=True)
    else:
        cols_train_val = [args.smiles_column] + args.target_columns
        df_train = df_assay.loc[:, cols_train_val].reset_index(drop=True)
        df_val = None

    for col in args.target_columns:
        if col in df_train.columns:
            df_train[col] = df_train[col].astype(int)
        if df_val is not None and col in df_val.columns:
            df_val[col] = df_val[col].astype(int)

    df_test = df_test[[args.smiles_column] + [c for c in args.target_columns if c in df_test.columns]].copy()
    df_test = df_test[~df_test[args.smiles_column].isin(df_train[args.smiles_column])].reset_index(drop=True)
    if df_val is not None and len(df_val) > 0:
        df_test = df_test[~df_test[args.smiles_column].isin(df_val[args.smiles_column])].reset_index(drop=True)
    df_test = df_test[df_test[args.smiles_column].apply(validate_smiles)].reset_index(drop=True)

    stacked_preds_test, vars_preds_test, auc = chemprop_ensemble(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        smiles_column=args.smiles_column,
        target_columns=args.target_columns,
        n_iter=0,
        output_dir=out_dir,
        task_tag=f"{args.mode}_iter0",
        n_ensemble=args.n_ensemble,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
    )

    if stacked_preds_test is None or vars_preds_test is None or len(df_test) == 0:
        (out_dir / "selected_none.csv").write_text(pd.DataFrame().to_csv(index=False))
        return

    new_mask = acquire_new(stacked_preds_test, vars_preds_test, args.mode, n_acquire=args.n_acquire)
    selected = df_test.loc[new_mask, [args.smiles_column]].copy()

    for j, target in enumerate(args.target_columns):
        if auc is not None and len(auc) > j:
            selected[f"auc_{target}"] = auc[j]
        mean_probs = stacked_preds_test.mean(dim=0)[:, j]
        var_probs = vars_preds_test[0, :, j]
        selected[f"pred_{target}"] = mean_probs[new_mask].detach().cpu().numpy()
        selected[f"var_{target}"] = var_probs[new_mask].detach().cpu().numpy()

    tag = f"Chemprop_{args.mode}_{'+'.join(args.target_columns)}_run{args.run_id}"
    (out_dir / f"{tag}.csv").write_text(selected.to_csv(index=False))


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)