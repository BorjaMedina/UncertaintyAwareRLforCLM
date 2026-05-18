"""Compute distances respect a specific training dataset


"""

from __future__ import annotations

__all__ = ["distances"]

from pydantic import Field
from pydantic.dataclasses import dataclass

from typing import List
import logging

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import DataStructs
from sklearn.impute import SimpleImputer

import reinvent_plugins.components.model_scoring_components.compute_sims as compute_sims
from ..component_results import ComponentResults
from ..add_tag import add_tag
from reinvent.scoring.utils import suppress_output
from reinvent.chemistry.similarity import calculate_tanimoto_batch, calculate_cosine_batch

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
    transformation_D: List[bool] = Field(default_factory=lambda: [False])
    high_D: List[float] = Field(default_factory=lambda: [1.3])
    low_D: List[float] = Field(default_factory=lambda: [0.3])
    k_D: List[float] = Field(default_factory=lambda: [0.3])

@add_tag("__component")
class UNCdistances:
    """Compute distances as a measure of uncertainty"""
    

    def __init__(self, params: Parameters):
        ## bind the fp parameters
        logger = logging.getLogger("reinvent")

        self.radius = params.radius[0]
        self.use_counts = params.use_counts[0]
        self.use_features = params.use_features[0]
        self.uncertainty_type= params.uncertainty_type[0]
        self.neighbours=int(params.neighbours[0])
        self.transformation_D=params.transformation_D[0]
        self.high_D=params.high_D[0]
        self.low_D=params.low_D[0]
        self.k_D=params.k_D[0]

        logger.debug(self.uncertainty_type)
        self.first_epoch = True
        self.meanFirstEpoch= None
        ## Load training smiles
        ## make this safe for files that do not exist and validate all are valid RDKIT smiles
        with open(params.training_smiles[0], "r") as f:
            self.training_smiles = [line.strip() for line in f if line.strip()]

        with open(params.training_smiles[0], "r") as f:
            self.training_smiles = [line.strip() for line in f if line.strip()]
        

        self.training_fingerprints_rdkit = compute_sims.smiles_to_fingerprints(
            self.training_smiles,
            radius=self.radius,
            n_bits=2048,
            use_features=self.use_features,
        )

        self.training_fingerprints = compute_sims.smiles_to_dense_fingerprints(
            self.training_smiles,
            radius=self.radius,
            n_bits=2048,
            use_features=self.use_features,
        )
            
            
            
        if "euclideanPCADesc" in self.uncertainty_type:
            converted_fps = []
            seed=33

            converted_fps=self.training_fingerprints

            self.pca = PCA(n_components=10, random_state=seed)
            
            descriptors=[]
            for smi in self.training_smiles:
                values = []
                mol = Chem.MolFromSmiles(smi)
                for name, desc in Descriptors.descList:
                    try:
                        values.append(desc(mol))
                    except:
                        values.append(None)
                descriptors.append(values)
            
            descriptors = np.array(descriptors, dtype=float)
            imputer = SimpleImputer(strategy='mean')
            descriptors = imputer.fit_transform(descriptors)
            
            converted_fps = np.array(converted_fps, dtype=np.uint16)    
            
            self.combined_training_features = np.hstack([converted_fps, descriptors])
            
            self.pca = PCA(n_components=10, random_state=seed)
        
            self.pca.fit(self.combined_training_features)
            
            self.trainingPCA=self.pca.transform(self.combined_training_features)
        
            logger.debug(f"Explained PCA variance: {self.pca.explained_variance_ratio_}")
        
        elif "euclideanPCA" in self.uncertainty_type:
            converted_fps = []
            seed=33
                
            self.pca = PCA(n_components=10, random_state=seed)
            
            self.pca.fit(self.training_fingerprints)
            self.trainingPCA=self.pca.transform(self.training_fingerprints)

    def updateDist(self,smiles: List[str]):     
        initSmi=smiles
        if "tanimoto" in self.uncertainty_type:
            sims = compute_sims.compute_SimTan(self,initSmi,init=True)
            length=1-np.array(sims)
        elif "euclideanPCA" in self.uncertainty_type:
            length = compute_sims.compute_SimEuclideanPCA(self,initSmi,init=True)
            
        self.meanSim=np.mean(length)
        logger.debug("Initial dist distances:")
        logger.debug(self.meanSim)


    
    
    def __call__(self, smiles: List[str]) -> ComponentResults:
        scores=[]

        logger = logging.getLogger("reinvent")

        if "tanimoto" in self.uncertainty_type:
            sims = compute_sims.compute_SimTan(self,smiles, normalize=False, mean=False)
            lengths=1-sims
        elif "euclideanPCA" in self.uncertainty_type:
            lengths = compute_sims.compute_SimEuclideanPCA(self,smiles, trans=self.transformation_D, high=self.high_D, low=self.low_D, k=self.k_D)


        #weights=1/(lengths+1e-6)
        weights= 1-np.array(lengths)

        
        scores.append(
                np.array(
                    [
                        p if smi != "INVALID" else np.nan
                        for p, smi in zip(weights, smiles)
                    ]
                )
            )
        
        

        return ComponentResults(scores, uncertainty_type=self.uncertainty_type, metadata={"lengths": lengths})
