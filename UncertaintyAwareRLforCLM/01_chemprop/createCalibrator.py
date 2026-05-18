import logging
import sys

import torch
from lightning import pytorch as pl


import pickle
from chemprop import data, featurizers, uncertainty
from chemprop.models import load_model
import numpy as np
from glob import glob
import json
import math
import pandas as pd     
import sys
import os

calibrator_path=sys.argv[1]

featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
trainer = pl.Trainer(
    logger=False,
    enable_progress_bar=False,
    accelerator="cpu",
    devices=1
)
unc_estimator = uncertainty.MVEEstimator()

model_paths = glob(os.path.join(calibrator_path, "model_*/checkpoints/best*.ckpt"))
allModels = [load_model(p, multicomponent=False) for p in model_paths]

for m in allModels:
    m.eval()
    for p in m.parameters():
        p.requires_grad = False

val_dataset=pd.read_csv("../00_data/egfr/cleaned_egfr.csv")
val_dataset = val_dataset[val_dataset["split"] == "val"]

val_smiles = val_dataset["canonical_smiles"].tolist()
val_targets = val_dataset["pChEMBL Value"].tolist()
val_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(val_smiles, val_targets)]
val_dset = data.MoleculeDataset(val_data, featurizer=featurizer)
val_loader = data.build_dataloader(val_dset, num_workers=0, shuffle=False)


# Get predictions and uncertainties on validation set
smi_predss, smi_uncss = unc_estimator(val_loader, allModels, trainer)

# Average predictions across models: shape [batch, n_tasks]
preds = smi_predss.mean(0)

# Keep ALL model uncertainties (DON'T average): shape [n_models, batch, n_tasks]
uncs = smi_uncss

# Convert targets to tensor with shape [batch, n_tasks]
targets_tensor = torch.tensor(val_targets).unsqueeze(1) if isinstance(val_targets, list) else torch.tensor(val_targets)
if targets_tensor.dim() == 1:
    targets_tensor = targets_tensor.unsqueeze(1)

# Create mask with shape [batch, n_tasks]
cal_mask = torch.from_numpy(np.isfinite(val_targets)).unsqueeze(1)

# Ensure preds has shape [batch, n_tasks]
if preds.dim() == 1:
    preds = preds.unsqueeze(1)

# Ensure uncs has shape [n_models, batch, n_tasks]
if uncs.dim() == 2:
    uncs = uncs.unsqueeze(2)

# Create and fit the calibrator
unc_calibrator = uncertainty.MVEWeightingCalibrator()
unc_calibrator.fit(preds, uncs, targets_tensor, cal_mask)

# Save the calibrator
with open(f"{calibrator_path}/mve_calibrator.pkl", "wb") as f:
     pickle.dump(unc_calibrator, f)

print(f"Calibrator saved to {calibrator_path}/mve_calibrator.pkl")
