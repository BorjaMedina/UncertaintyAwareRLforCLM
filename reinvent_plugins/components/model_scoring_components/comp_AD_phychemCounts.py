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
from rdkit.Chem import AllChem, DataStructs, QED, rdMolDescriptors, FindMolChiralCenters

from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol


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
    featurizer: List[str] = Field()

@add_tag("__component")
class ADphychemCounts:


    def physchem_and_ecfp4_counts(self, smiles):
        X = [np.concatenate((self.one_physchem(s), self.one_ecfp_counts(s, radius=2))) for s in smiles]
        return X
    
    def one_ecfp_counts(self, smile, radius=2):
        "Calculate ECFP fingerprint. If smiles is invalid return none"
        try:  
            m = Chem.MolFromSmiles(smile)
            fp = AllChem.GetHashedMorganFingerprint(m, 2, nBits=1024)
            array = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, array)
        
            
            return array
        except:
            print("Not Working")
            return None 

    def one_physchem(self, smile):
        try:
            m = Chem.MolFromSmiles(smile)
            if m is not None:
                hba = rdMolDescriptors.CalcNumHBA(m)
                hbd = rdMolDescriptors.CalcNumHBD(m)
                nrings = rdMolDescriptors.CalcNumRings(m)
                rtb = rdMolDescriptors.CalcNumRotatableBonds(m)
                psa = rdMolDescriptors.CalcTPSA(m)
                logp, mr = rdMolDescriptors.CalcCrippenDescriptors(m)
                mw = rdMolDescriptors._CalcMolWt(m)
                csp3 = rdMolDescriptors.CalcFractionCSP3(m)
                hac = m.GetNumHeavyAtoms()
                
                charges = []
                for at in m.GetAtoms():
                    charges.append(at.GetFormalCharge())

                if hac == 0:
                    fmf = 0
                else:
                    fmf = GetScaffoldForMol(m).GetNumHeavyAtoms() / hac
                ri = m.GetRingInfo()
                n_rings = len(ri.AtomRings())
                max_ring_size = len(max(ri.AtomRings(), key=len, default=()))
                min_ring_size = len(min(ri.AtomRings(), key=len, default=()))
                total_charges = sum(charges)
                min_charge = min(charges)
                max_charge = max(charges)
                n_chiral_centers = len(FindMolChiralCenters(m, includeUnassigned=True))
                return np.array([hba, hbd, hba + hbd, nrings, rtb, psa, logp, mr, mw,
                    csp3, fmf, hac, 
                    max_ring_size, min_ring_size, total_charges, min_charge, max_charge, n_chiral_centers])
        except:
            return None


    def __init__(self, params: Parameters):

        logger = logging.getLogger("reinvent")

        self.featurizer = params.featurizer[0]
        self.radius = params.radius[0]
        self.use_counts = params.use_counts[0]
        self.use_features = params.use_features[0]
        self.uncertainty_type= params.uncertainty_type[0]
        self.neighbours=int(params.neighbours[0])

        logger.debug(self.uncertainty_type)
        self.first_epoch = True
        self.meanFirstEpoch= None

        with open(params.training_smiles[0], "r") as f:
            self.training_smiles = [line.strip() for line in f if line.strip()]



    

    def __call__(self, smiles: List[str]) -> ComponentResults:
        scores=[]
                
        uncertainties = []

        if len(smiles)>0:
            fps= compute_sims.smiles_to_fingerprints(smiles)
            fps_matrix = np.stack([np.array(fp) for fp in fps])
            preds= self.model["icp"].predict(fps_matrix, significance=self.model["significance"])
            p_vals= self.model["icp"].predict(fps_matrix, significance=None)  
            
            for i, ps in enumerate(preds):
                included = [('inactive', 'active')[j] for j, v in enumerate(ps) if v]
                set_size = len(included)

                if set_size == 1:
                    uncertainties.append(self.model["significance"])
                elif set_size == 2:
                    uncertainty = 1 - abs(p_vals[i, 1] - p_vals[i, 0])
                    uncertainties.append(uncertainty)
                else:
                    uncertainties.append(1-self.model["significance"]) 
            

        elif "tanimoto" in self.uncertainty_type:
            sims, dist = compute_sims.compute_SimTan(self,smiles, normalize=False, mean=False)
        
        
        scores=np.array(p_vals[:, 1])

    

        return ComponentResults(scores, uncertainty_type=self.uncertainty_type, metadata={"uncertainties": np.array(uncertainties), "sims": np.array(sims)})
