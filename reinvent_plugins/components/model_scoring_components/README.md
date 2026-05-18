# Model Scoring Components

This folder contains two different scoring components used for modeling:

1. **`comp_model_logp.py`**: the component that models the noise.
2. **`comp_distances.py`**: the component that uses distances as a scoring mechanism to force the model closer to the training dataset.

The parameters for these components are similar because the noise is calculated using the distance to the training dataset.

---

## Components and Parameters

### `comp_distances.py`
**Description**: Uses distances to calculate the mean distance to the k-nearest neighbors.

**Parameters**:
- `training_smiles`: Path to the training SMILES file (one column containing the SMILES).
- `neighbours`: Number of KNNs to use for distance calculation (mean distance to the k-nearest neighbors).
- `uncertainty_type`: Type of uncertainty to use. Options:
  - `"tanimoto"`
  - `"tanimotoNorm"`
  - `"euclideanPCA"`
- `radius`: Fingerprint radius.
- `use_counts`: Boolean (`True`/`False`).
- `use_features`: Boolean (`True`/`False`).

---

### `comp_model_logp.py`
**Description**: Models the noise and penalizes the loss directly based on the uncertainty type.

**Parameters**:
- `training_smiles`: Path to the training SMILES file (one column containing the SMILES).
- `neighbours`: Number of KNNs to use for distance calculation (mean distance to the k-nearest neighbors).
- `uncertainty_type`: Type of uncertainty to use. Options:
  - `"tanimoto"`
  - `"tanimotoNorm"`
  - `"euclideanPCA"`
  - `"harshUnc"`
- `radius`: Fingerprint radius.
- `use_counts`: Boolean (`True`/`False`).
- `use_features`: Boolean (`True`/`False`).
- `uq_parameters`: Noise distribution.

---

## Temporal Solution for Uncertainty Types

The current implementation for handling different uncertainty types is kinda chaotic:

1. **Uncertainty Type: `harshUnc`**
   - If the `uncertainty_type` parameter starts with `"harshUnc"` (e.g., `harshUnc_tanimoto` or `harshUnc_euclideanPCA`), the distance will directly penalize the loss.
   - **Reference**: `reinvent.scoring.scorer` (line 205).

2. **Uncertainty Type: `earlyMPO`**
   - If the `uncertainty_type` contains `"earlyMPO"`, the error distribution will be sampled 1000 times, and the mean of those errors will be returned (effectively removing the error).
   - **Reference**: `reinvent.reinvent_plugins.components.model_scoring_components.com_model_logp` (line 169).

3. **Uncertainty Type: `meanPostMPO`**
   - If the `uncertainty_type` contains `"meanPostMPO"`, a true MPO will be performed by sampling multiple times and calculating the mean of the transformed scores.
   - **Reference**: `reinvent.scoring.compute_scores` (line 160).

---

## Combining Options

You can combine these uncertainty options.
