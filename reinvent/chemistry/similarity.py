import numpy as np
from rdkit.Chem import DataStructs


def calculate_tanimoto_batch(fp, fps) -> np.array:
    return np.array(DataStructs.BulkTanimotoSimilarity(fp, fps))

def calculate_cosine_batch(fp, fps) -> np.array:

    
    arr_fps = np.stack(fps)

    # Normalize
    ref_norm = np.linalg.norm(fp)
    norms = np.linalg.norm(arr_fps, axis=1)

    sims = arr_fps.dot(fp) / (norms * ref_norm)
    sims[np.isnan(sims)] = 0.0
    return sims


def calculate_tanimoto(query_fps, ref_fingerprints) -> np.array:
    return np.array(
        [np.max(DataStructs.BulkTanimotoSimilarity(fp, ref_fingerprints)) for fp in query_fps]
    )
