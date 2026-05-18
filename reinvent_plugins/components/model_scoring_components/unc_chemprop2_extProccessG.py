from copyreg import pickle
import sys
import os
from glob import glob

import numpy as np
import pickle
import math
import json
import torch
from lightning import pytorch as pl

from chemprop import data, featurizers
from chemprop.models import load_model
from chemprop.uncertainty import MVEEstimator

from rdkit import Chem

def sigmoidTrans(x, steepness=3, center=0.5, min_val=0, max_val=6):
    x = np.array(x)
    x_clipped = np.clip(x, min_val, max_val)

    result = 1 / (1 + np.exp(-steepness * (x_clipped - center)))
    sigmoid_min = 1 / (1 + np.exp(-steepness * (min_val - center)))
    sigmoid_max = 1 / (1 + np.exp(-steepness * (max_val - center)))
    result = (result - sigmoid_min) / (sigmoid_max - sigmoid_min)
    return result

model_path = sys.argv[1]
input_path=sys.argv[2]

cal_path = glob(os.path.join(model_path, "*calibrator.pkl"))[0]


with open(input_path) as f:
    smilies = f.read().splitlines()

valid_mask = [Chem.MolFromSmiles(smi) is not None for smi in smilies]
valid_smiles = [smi for smi, is_valid in zip(smilies, valid_mask) if is_valid]

if os.path.isdir(model_path):
    checkpoint_files = glob(os.path.join(model_path, "model_*/checkpoints/best*.ckpt")) + glob(os.path.join(model_path, "*.pt"))
    if checkpoint_files:
        models = [load_model(ckpt, multicomponent=False) for ckpt in checkpoint_files]

else:
    print(f"Loading single model: {model_path}")
    models = [load_model(model_path, multicomponent=False)]


for model in models:
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()



test_data = [data.MoleculeDatapoint.from_smi(smi) for smi in valid_smiles]
test_dset = data.MoleculeDataset(test_data, featurizer=featurizer)
test_loader = data.build_dataloader(test_dset, num_workers=0, shuffle=False)


unc_estimator = MVEEstimator()
trainer = pl.Trainer(
    logger=False,
    enable_progress_bar=False,
    accelerator="cuda" if torch.cuda.is_available() else "cpu",
    devices=1
)

with open(cal_path, "rb") as f:
    calibrator = pickle.load(f)

if len(smilies)>0:
    predictions, uncertainties = unc_estimator(test_loader, models, trainer)

    # Average across models if multiple
    if len(models) > 1:
        predictions = predictions.mean(0)
        uncertainties = uncertainties.mean(0)
    else:
        predictions = predictions[0]
        uncertainties = uncertainties[0]

    calibrator.apply(uncertainties)    

    calibrated_uncertainties = calibrator.apply(uncertainties.detach().clone())
    scores = predictions.squeeze().numpy()
    uncertainties = calibrated_uncertainties.squeeze().numpy()

else:
    scores=[]
    uncertainties=[]


final_scores = []
final_uncertainties = []
valid_idx = 0
for is_valid in valid_mask:
    if is_valid:
        final_scores.append(float(scores[valid_idx]))
        final_uncertainties.append(float(uncertainties[valid_idx]))
        valid_idx += 1
    else:
        final_scores.append(math.nan)
        final_uncertainties.append(math.nan)

final_distances= np.clip(1 - np.array(final_uncertainties) / 1.5, 0.001, 1)
final_distances=[float(s) for s in final_distances]

dataset = {"version": 1, "payload": {"predictions": final_distances, "uncertainties": list(final_uncertainties)}}

print(json.dumps(dataset))