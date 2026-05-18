import logging
import sys

import torch
from lightning import pytorch as pl


import pickle
from chemprop import data, featurizers
from chemprop.models import load_model
from chemprop.uncertainty import MVEEstimator
from rdkit import Chem
import numpy as np
from glob import glob
import json
import math

#smilies = sys.stdin.read().splitlines()
input_path=sys.argv[1]
with open(input_path) as f:
    smilies = f.read().splitlines()

valid_mask = [Chem.MolFromSmiles(smi) is not None for smi in smilies]
valid_smiles = [smi for smi, is_valid in zip(smilies, valid_mask) if is_valid]

#np.savetxt("/projects/mai/se_mai/users/kcgd777_borja/uncertainty/07_RL_chemprop/test.txt", smilies, fmt="%s")

featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
model_paths = glob("/home/kcgd777/projects/uncertainty/06_Chemprop2Model/chemprop_model_mve_noLimNoScaler/*/best_*epoch=19*")
allModels = [load_model(p, multicomponent=False) for p in model_paths]

for m in allModels:
    m.eval()
    for p in m.parameters():
        p.requires_grad = False

unc_estimator = MVEEstimator()


with open("/projects/mai/se_mai/users/kcgd777_borja/uncertainty/06_Chemprop2Model/noLim/mve_calibrator.pkl", "rb") as f:
    unc_calibrator = pickle.load(f)

trainer = pl.Trainer(
    logger=False,
    enable_progress_bar=False,
    accelerator="gpu",
    devices=1
)



random_data = [data.MoleculeDatapoint.from_smi(smi) for smi in valid_smiles]
random_dset = data.MoleculeDataset(random_data, featurizer=featurizer)
random_loader = data.build_dataloader(random_dset, num_workers=1, shuffle=False)
if len(smilies)>0:
    smi_predss, smi_uncss = unc_estimator(random_loader, allModels, trainer)


    preds = smi_predss.mean(0)  
    uncs = smi_uncss.mean(0)

    preds = np.clip(preds, -1e6, 1e6)
    uncs = np.clip(uncs, -1e6, 1e6)
    
    uncs=torch.tensor(uncs)
    unc_calibrator.apply(uncs)

    scores = np.array(preds.reshape(-1).tolist())
    uncertainties=np.array(uncs.reshape(-1).tolist())
else:
    scores=[]
    uncertainties=[]
    
final_scores = []
final_uncertainties = []
valid_idx = 0
for is_valid in valid_mask:
    if is_valid:
        final_scores.append(scores[valid_idx])
        final_uncertainties.append(uncertainties[valid_idx])
        valid_idx += 1
    else:
        final_scores.append(math.nan)
        final_uncertainties.append(math.nan)    
dataset = {"version": 1, "payload": {"predictions": list(final_scores), "uncertainties": list(final_uncertainties)}}

print(json.dumps(dataset))
#with open("/home/kcgd777/temp_results.json", mode="w") as json_file:
    #json.dump(dataset, json_file, indent=4)




