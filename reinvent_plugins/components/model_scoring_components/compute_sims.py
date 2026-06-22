from typing import List
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors

from sklearn.impute import SimpleImputer

from reinvent.chemistry.similarity import calculate_tanimoto_batch, calculate_cosine_batch
from reinvent.scoring.transforms.sigmoid_functions import stable_sigmoid, hard_sigmoid

def smiles_to_fingerprints(
    smiles_list: List[str],
    radius: int = 3,
    n_bits: int = 2048,
    use_features: bool = False,
):
    """Return a list of RDKit ExplicitBitVect fingerprint objects."""
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    mols = [m for m in mols if m is not None]

    fps = [
        AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius,
            nBits=n_bits,
            useFeatures=use_features,
        )
        for mol in mols
    ]
    return fps


def smiles_to_dense_fingerprints(
    smiles_list: List[str],
    radius: int = 3,
    n_bits: int = 2048,
    use_features: bool = False,
):
    """Return fingerprints as a dense numpy array (for PCA/Euclidean methods)."""
    fps = smiles_to_fingerprints(smiles_list, radius, n_bits, use_features)

    arr = np.zeros((len(fps), n_bits), dtype=np.int8)

    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, arr[i])

    return arr


def compute_SimTan(self, smiles: List[str], normalize=False, mean=False,init=False) -> np.array:
    similarities = []
    query_fingerprints = smiles_to_fingerprints(
        smiles, radius=self.radius, n_bits=2048, use_features=self.use_features
    )
    for fingerprint in query_fingerprints:
        tanimoto_similarities = calculate_tanimoto_batch(
            fingerprint, self.training_fingerprints_rdkit
        )
        similarities.append(np.mean(np.partition(tanimoto_similarities, -self.neighbours)[-self.neighbours:]))

    if init:
        return np.array(similarities)
    
    else:
        return np.array(similarities)
        

    if normalize:      
        return normalizer( similarities, xmin=0.2, xmax=0.8)
    elif mean:
        k=10
        normalized= 1 / (1 + np.exp(-k * (np.array(similarities) - self.meanSim)))
        return(normalized)
    
        
    
def compute_SimCos(self, smiles: List[str], normalize=False,nBits=2048) -> np.array:
    similarities = []
    
    query_fingerprints = smiles_to_dense_fingerprints(
        smiles, radius=self.radius, n_bits=2048, use_features=self.use_features
    )
    

    for fingerprint in query_fingerprints:
        cosine_sim = calculate_cosine_batch(
            fingerprint, self.training_fingerprints
        )
        similarities.append(np.mean(np.partition(cosine_sim, -self.neighbours)[-self.neighbours:]))
        
    if normalize:      
        return normalizer(similarities, xmin=0.6, xmax=1)
    else:
        return similarities

def compute_SimEuclideanPCA(self, smiles: List[str], limit= 4, init=False, trans=False, high=1.3, low=0.3, k=0.3)-> np.array:
    
    def sigmoidTransform(values,high, low, k):
        values = np.array(values, dtype=np.float32)

        x = values - (high + low) / 2

        if (high - low) == 0:
            k = 10.0 * k
            transformed = hard_sigmoid(x, k)
        else:
            k = 10.0 * k / (high - low)
            transformed = stable_sigmoid(x, k)

        return np.array(transformed, dtype=np.float32)

    query_fingerprints = smiles_to_dense_fingerprints(
        smiles, radius=self.radius, n_bits=2048, use_features=self.use_features
    )
        
    sample_fps = self.pca.transform(query_fingerprints)
    
    lengths=[]
    for entry in sample_fps:
        dists = np.linalg.norm(self.trainingPCA - entry, axis=1)
        knn_indices = np.argpartition(dists, self.neighbours)[:self.neighbours]
        knn_dists = dists[knn_indices]
        mean_dist = np.mean(knn_dists)
        std_dist = np.std(knn_dists)

        length = mean_dist + 0.1 * std_dist
        #centroids and mean vector
        """
        knn_fps = self.trainingPCA[knn_indices]
        vectors = knn_fps - entry  
        mean_vec = np.mean(vectors, axis=0)
        length = np.linalg.norm(mean_vec)"""
        
        lengths.append(length)

    if trans:  
        lengths=sigmoidTransform(lengths, high=high, low=low, k=k)
    lengths = np.asarray(lengths, dtype=np.float32).clip(0.0, 1.0)

    return lengths



def compute_SimEuclideanPCADescriptors(self, smiles: List[str], limit= 10000)-> np.array:
    
    
    query_fingerprints = smiles_to_dense_fingerprints(
        smiles, radius=self.radius,n_bits=2048, use_features=self.use_features
    )

    
    
    descriptors=[]
    for smi in smiles:
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
    
    combined_features = np.hstack([query_fingerprints, descriptors])
    

    sample_fps = self.pca.transform(combined_features)
    
    batchSims=[]
    for entry in sample_fps:
        
        dists = np.linalg.norm(self.trainingPCA - entry, axis=1)
        knn_indices = np.argpartition(dists, self.neighbours)[:self.neighbours]
        knn_fps = self.trainingPCA[knn_indices]

        limit=4
        vectors = knn_fps - entry  
        mean_vec = np.mean(vectors, axis=0)
        length = np.linalg.norm(mean_vec)
        
        sim = (limit - length) / limit
        normSim = np.clip(sim, 0, 1) 
        
        batchSims.append(normSim)



    """ Check code with NearestNeighbors
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=self.neighbours, metric='euclidean')
    nn.fit(self.trainingPCA)

    distances, indices = nn.kneighbors(sample_fps)

    batchSims = []
    for entry, knn_indices in zip(sample_fps, indices):
        knn_fps = self.trainingPCA[knn_indices]
        mean_vec = np.mean(knn_fps - entry, axis=0)
        length = np.linalg.norm(mean_vec)
        sim = (limit - length) / limit
        batchSims.append(np.clip(sim, 0, 1))"""
        
    return np.array(batchSims)  


def normalizer( x, xmin, xmax):
    x=np.array(x)
    norm= (x - xmin) / (xmax - xmin)
    return np.clip(norm,0,1)

