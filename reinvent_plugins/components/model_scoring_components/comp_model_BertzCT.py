from typing import List, Optional
import logging
import numpy as np
from sklearn.decomposition import PCA

from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit.Chem import Descriptors
from sklearn.impute import SimpleImputer

from pydantic import Field
from pydantic.dataclasses import dataclass

import reinvent_plugins.components.model_scoring_components.compute_sims as compute_sims

from ..component_results import ComponentResults
from ..add_tag import add_tag

logger = logging.getLogger("reinvent")

@add_tag("__parameters")
@dataclass
class Parameters:
    training_smiles: List[str] = Field()
    uq_parameters: List[List[float]]= Field(default_factory=lambda: [[0.2, 0.8]])
    radius: List[int] = Field(default_factory=lambda: [3])
    use_counts: List[bool] = Field(default_factory=lambda: [True])
    use_features: List[bool] = Field(default_factory=lambda: [False])
    uncertainty_type: Optional[List[str]] = Field(default=[None])
    neighbours: List[int] = Field(default_factory=lambda: [4])
    transformation_D: List[bool] = Field(default_factory=lambda: [False])
    high_D: List[float] = Field(default_factory=lambda: [1.3])
    low_D: List[float] = Field(default_factory=lambda: [0.3])
    k_D: List[float] = Field(default_factory=lambda: [0.3])
    
@add_tag("__component")
class NoisyBertz:
    """Compute noise in bertZ and measure uncertainty of the noisy samples using different measures"""

    no_cache = True  # stochastic component: disable caching so average=True works correctly

    def __init__(self, params: Parameters):
        logger = logging.getLogger("reinvent")
        self.radius = params.radius[0]
        self.use_counts = params.use_counts[0]
        self.use_features = params.use_features[0]
        self.uncertainty_type= params.uncertainty_type[0]
        self.uq_parameters = params.uq_parameters[0]
        self.neighbours=int(params.neighbours[0])
        self.transformation_D = params.transformation_D[0]
        self.high_D = params.high_D[0]
        self.low_D = params.low_D[0]
        self.k_D = params.k_D[0]

        
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
        

             

        if "euclideanPCA" in self.uncertainty_type:
            seed=33
        
    
            self.pca = PCA(n_components=10, random_state=seed)
        

            self.pca.fit(self.training_fingerprints)
            self.trainingPCA=self.pca.transform(self.training_fingerprints)
            logger.debug(f"Explained PCA variance: {self.pca.explained_variance_ratio_}")
        
    def updateDist(self,smiles: List[str]):     
        initSmi=smiles
        length = compute_sims.compute_SimEuclideanPCA(self,initSmi,init=True)
        self.meanSim=np.mean(length)
        logger.debug("Initial bertzCT distances:")
        logger.debug(self.meanSim)
        
    def uq_model(self, distance: float) -> float:
        return np.random.normal(
            loc=0, scale=self.uq_parameters[0] + distance * self.uq_parameters[1]
        )

    def __call__(self, smiles: List[str]) -> ComponentResults:
        scores = []
        logger = logging.getLogger("reinvent")

        ## compute BertzCT values
        Bertz_actual = [Descriptors.BertzCT(Chem.MolFromSmiles(smile)) for smile in smiles]

        if "tanimoto" in self.uncertainty_type:
            sims = compute_sims.compute_SimTan(self,smiles, normalize=False, mean=False)
            lengths=1-np.array(sims)
        elif "euclideanPCA" in self.uncertainty_type:
            lengths = compute_sims.compute_SimEuclideanPCA(self,smiles, trans=self.transformation_D, high=self.high_D, low=self.low_D, k=self.k_D)
  

        errors = [self.uq_model(distance) for distance in lengths]
        
        
        scores.append(
                np.array(
                    [
                        p+(200*e) if smi != "INVALID" else np.nan
                        for p, e, smi in zip(Bertz_actual, errors, smiles)
                    ]
                )
            )
        
        
        return ComponentResults(
            scores, uncertainty_type=self.uncertainty_type, metadata={"Bertz_actual": Bertz_actual, "errors": errors, "lengths": lengths}
        )
