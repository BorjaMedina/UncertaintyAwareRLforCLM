import logging
import numpy as np
from dataclasses import dataclass
from typing import List

from ..component_results import ComponentResults
from lightning import pytorch as pl
import torch
import pandas as pd
from ..add_tag import add_tag

from chemprop import data,uncertainty, featurizers
from chemprop.models import load_model

from rdkit import Chem


from glob import glob




logger = logging.getLogger("reinvent")

@add_tag("__parameters")
@dataclass
class Parameters:
    """Parameters for the scoring component

    Note that all parameters are always lists because components can have
    multiple endpoints and so all the parameters from each endpoint is
    collected into a list.  This is also true in cases where there is only one
    endpoint.
    """

    models_path: List[str]
    train_data: List[str] 
    val_data: List[str]
    test_data: List[str]


@add_tag("__component")
class chemprop2:
    def __init__(self, params: Parameters):

        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        train_csv=pd.read_csv(params.train_data[0],header=0)
        train = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(train_csv["smiles"].tolist(), train_csv["IC50"].tolist())]
        val_csv=pd.read_csv(params.val_data[0],header=0)
        val = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(val_csv["smiles"].tolist(), val_csv["IC50"].tolist())]
        test_csv=pd.read_csv(params.test_data[0],header=0)
        test = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(test_csv["smiles"].tolist(), test_csv["IC50"].tolist())]


        val_dset = data.MoleculeDataset(val, featurizer=self.featurizer)
        test_dset = data.MoleculeDataset(test, featurizer=self.featurizer)
        train_dset = data.MoleculeDataset(train, featurizer=self.featurizer)

        val_loader = data.build_dataloader(val_dset, num_workers=1, shuffle=False)
        test_loader = data.build_dataloader(test_dset, num_workers=1, shuffle=False)

        cal_dset = val_dset
        cal_loader = data.build_dataloader(cal_dset, shuffle=False,)

        train_dset.Y = np.expand_dims(train_dset.Y, axis=1)


        self.unc_estimator = uncertainty.MVEEstimator()
        self.unc_calibrator = uncertainty.MVEWeightingCalibrator()



        unc_evaluators = [
            uncertainty.NLLRegressionEvaluator(),
            uncertainty.CalibrationAreaEvaluator(),
            uncertainty.ExpectedNormalizedErrorEvaluator(),
            uncertainty.SpearmanEvaluator(),
        ]

        self.uncTrainer = pl.Trainer(logger=False, enable_progress_bar=True, accelerator="gpu", devices=1)

        model_paths = glob(params.models_path[0])
        self.allModels = [load_model(model_path, multicomponent=False) for model_path in model_paths]

        test_predss, test_uncss = self.unc_estimator(test_loader, self.allModels, self.uncTrainer)
        test_preds = test_predss.mean(0)
        test_uncs = test_uncss.mean(0)


        unc_test = pd.DataFrame(
                {
                    "smiles": test_dset.smiles,
                    "target": test_dset.Y.reshape(-1),
                    "pred": test_preds.reshape(-1),
                    "unc": test_uncs.reshape(-1),
                }
            )
            

        cal_predss, cal_uncss = self.unc_estimator(cal_loader, self.allModels, self.uncTrainer)

        cal_targets = cal_dset.Y
        cal_mask = torch.from_numpy(np.isfinite(cal_targets)).unsqueeze(1)
        cal_targets = np.nan_to_num(cal_targets, nan=0.0)
        cal_targets = torch.from_numpy(cal_targets).unsqueeze(1)

        calibrated_test_uncs = []
        calibrated_test_preds = []
        cal_preds_mean=cal_predss.mean(0)
        cal_uncss_mean=cal_uncss.mean(0)

    
        self.unc_calibrator.fit(cal_preds_mean, cal_uncss_mean, cal_targets, cal_mask)


        calibrated_test_uncs = self.unc_calibrator.apply(test_uncs)
        

        unc_test_cal = pd.DataFrame(
            {
                "smiles": test_dset.smiles,
                "target": test_dset.Y.reshape(-1),
                "pred": test_preds.reshape(-1),
                "cal_unc": calibrated_test_uncs.reshape(-1),
            }
        )



        test_targets = test_dset.Y
        test_mask = torch.from_numpy(np.isfinite(test_targets))
        test_targets = np.nan_to_num(test_targets, nan=0.0)
        test_targets = torch.from_numpy(test_targets)


        for evaluator in unc_evaluators:
            evaluation = evaluator.evaluate(calibrated_test_preds.unsqueeze(1), calibrated_test_uncs.unsqueeze(1), test_targets.unsqueeze(1), test_mask.unsqueeze(1))
            logger.info(f"{evaluator.alias}: {evaluation.tolist()}")
            

    def __call__(self, smilies: List[str]) -> np.ndarray:
        smiles_batch = [[s] for s in smilies]
    
        valid_smiles=[]
        invalid_smiles=[]
        for smi in smiles_batch:
            if Chem.MolFromSmiles(smi) is not None:
                valid_smiles.append(smi)
            else:
                invalid_smiles.append(smi)
            
        random_data = [data.MoleculeDatapoint.from_smi(smi) for smi in valid_smiles]
        random_dset = data.MoleculeDataset(random_data, featurizer=self.featurizer)
        random_loader = data.build_dataloader(random_dset, num_workers=1, shuffle=False)

        smi_predss, smi_uncss = self.unc_estimator(random_loader, self.allModels, self.uncTrainer)

        preds = smi_predss.mean(0)
        uncs = smi_uncss.mean(0)

        self.unc_calibrator.apply(uncs)
        
        scores = np.array(preds.reshape(-1).tolist())
        uncertainties=np.array(uncs.reshape(-1).tolist())
        
        return ComponentResults(scores, uncertainty=uncertainties)
        



        

