"""Compute distances respect a specific training dataset


"""

from __future__ import annotations

__all__ = ["distances"]

from pydantic import Field
from pydantic.dataclasses import dataclass

from typing import List
import logging
import cloudpickle
import numpy as np


import reinvent_plugins.components.model_scoring_components.compute_sims as compute_sims
from ..component_results import ComponentResults
from ..add_tag import add_tag


from sklearn.decomposition import PCA


logger = logging.getLogger("reinvent")


supported_types = ["tanimoto", "euclideanPCA","tanimotoNorm", "cosine"]


@add_tag("__parameters")
@dataclass
class Parameters:
    """
    :param training_smiles: 
    :param neighbours:
    :param uncertainty_type: 
    :param radius:
    :param use_counts:
    :param use_features:
    """
    
    training_smiles: List[str] = Field()
    neighbours: List[int] = Field(default_factory=lambda: [4])
    uncertainty_type: List[str] = Field(default=["tanimoto"])
    radius: List[int] = Field(default_factory=lambda: [3])
    use_counts: List[bool] = Field(default_factory=lambda: [True])
    use_features: List[bool] = Field(default_factory=lambda: [False])
    model_path: List[str] = Field()

@add_tag("__component")
class classificationCP:
    """Compute distances as a measure of uncertainty"""
    

    def __init__(self, params: Parameters):
        ## bind the fp parameters
        logger = logging.getLogger("reinvent")

        self.radius = params.radius[0]
        self.use_counts = params.use_counts[0]
        self.use_features = params.use_features[0]
        self.uncertainty_type= params.uncertainty_type[0]
        self.neighbours=int(params.neighbours[0])

        logger.debug(self.uncertainty_type)
        self.first_epoch = True
        self.meanFirstEpoch= None
        ## Load training smiles
        ## make this safe for files that do not exist and validate all are valid RDKIT smiles
        with open(params.training_smiles[0], "r") as f:
            self.training_smiles = [line.strip() for line in f if line.strip()]


        with open(params.model_path[0], 'rb') as f:
            self.model = cloudpickle.load(f)



    def __call__(self, smiles: List[str]) -> ComponentResults:
        scores = [[]]
        if len(smiles) > 0:
            fps= compute_sims.smiles_to_fingerprints(smiles)
            fps_matrix = np.stack([np.array(fp) for fp in fps])
            preds= self.model["icp"].predict(fps_matrix, significance=self.model["significance"])
            logger.debug(len(preds))
            p_vals = np.asarray(self.model["icp"].predict(fps_matrix, significance=None))
            logger.debug(p_vals.shape)

            scores = [np.array(p_vals[:, 1].tolist())]

        
        return ComponentResults(
            scores, uncertainty_type=self.uncertainty_type
        )
