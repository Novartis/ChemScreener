from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from chemprop import data, featurizers, models, nn 

from data_utils import df_to_datapoints
from utils import cal_auc


def chemprop_ensemble(
    df_train,
    smiles_column: str,
    target_columns: Sequence[str],
    n_iter: int,
    output_dir: Path,
    task_tag: str,
    df_val=None,
    df_test=None,
    n_ensemble: int = 10,
    batch_size: int = 512,
    max_epochs: int = 200,
    accelerator: str = "auto",
    devices: int | str | List[int] = 1,
) -> Tuple[torch.Tensor | None, torch.Tensor | None, List[float] | None]:
    train_data, _ = df_to_datapoints(df_train, smiles_column, target_columns)
    mp = nn.BondMessagePassing()
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_dset = data.MoleculeDataset(train_data, featurizer)
    train_loader = data.build_dataloader(train_dset, batch_size=batch_size, num_workers=4)

    if df_val is not None and len(df_val) > 0:
        val_data, _ = df_to_datapoints(df_val, smiles_column, target_columns)
        val_dset = data.MoleculeDataset(val_data, featurizer)
        val_loader = data.build_dataloader(val_dset, batch_size=batch_size, num_workers=4, shuffle=False)
    else:
        val_loader = None

    if df_test is not None and len(df_test) > 0:
        test_targets = [c for c in target_columns if c in df_test.columns]
        test_data, _ = df_to_datapoints(
            df_test,
            smiles_column,
            test_targets if len(test_targets) == len(target_columns) else None
        )
        test_dset = data.MoleculeDataset(test_data, featurizer)
        test_loader = data.build_dataloader(test_dset, batch_size=batch_size, num_workers=4, shuffle=False)
    else:
        test_loader = None

    ffn_input_dim = mp.output_dim
    ensemble_preds_test: List[torch.Tensor] = []
    ensemble_preds_val: List[torch.Tensor] = []
    auc_val_all: List[List[float]] = []

    ckpt_root = output_dir / "checkpoints" / task_tag
    ckpt_root.mkdir(parents=True, exist_ok=True)

    for es in range(n_ensemble):
        mp = nn.BondMessagePassing()
        ffn = nn.BinaryClassificationFFN(input_dim=ffn_input_dim, n_tasks=len(target_columns))
        mpnn = models.MPNN(mp, nn.MeanAggregation(), ffn)

        monitor_mode = "max" if mpnn.metrics[0].higher_is_better else "min"
        checkpointing = ModelCheckpoint(
            dirpath=str(ckpt_root / f"rep{es}_iter{n_iter}"),
            filename="chemprop_last-{epoch}",
            monitor='train_loss',
            mode=monitor_mode,
            save_last=True,
            auto_insert_metric_name=False,
        )

        trainer = Trainer(
            logger=False,
            enable_checkpointing=True,
            enable_progress_bar=False,
            accelerator=accelerator,
            devices=devices,
            callbacks=[checkpointing],
            max_epochs=max_epochs,
        )

        if val_loader is not None:
            trainer.fit(mpnn, train_loader, val_loader)
        else:
            trainer.fit(mpnn, train_loader)

        best_model_path = checkpointing.last_model_path
        model = mpnn.__class__.load_from_checkpoint(best_model_path)

        if test_loader is not None:
            preds_test = torch.concat(trainer.predict(model, test_loader), dim=0)
            ensemble_preds_test.append(preds_test)

        if val_loader is not None:
            preds_val = torch.concat(trainer.predict(model, val_loader), dim=0)
            ensemble_preds_val.append(preds_val)
            auc_val_all.append(cal_auc(df_val, target_columns, preds_val))

    stacked_preds_test = None
    vars_preds_test = None
    if len(ensemble_preds_test) > 0:
        stacked_preds_test = torch.stack(ensemble_preds_test).float()  # [K, N_test, C]
        vars_preds_test = torch.var(stacked_preds_test, dim=0, correction=0).unsqueeze(0)  # [1, N_test, C]

    mean_auc = None
    if len(ensemble_preds_val) > 0:
        mean_auc = list(np.nanmean(np.asarray(auc_val_all, dtype=float), axis=0))

    return stacked_preds_test, vars_preds_test, mean_auc