# Uncertainty-Aware reinforcement learning for chemical language models

This repository contains the code and experiments for my master's thesis on uncertainty-aware reinforcement learning for de novo molecular design.
Everything corresponding to the paper is inside of `UncertaintyAwareRLforCLM`


DISCLAIMER: If you want to check the `README.md` of REINVENT4 check `REINVENT4_README.md` 

## REINVENT4 Introduction and Credit

REINVENT4 is an open-source framework for molecular generation and multi-objective optimization. It uses a generative model (policy) to propose molecules and optimizes it with reinforcement learning based on scoring functions.

In this thesis, REINVENT4 is the baseline framework and has been extended with new uncertainty-aware functionalities. These additions are implemented on top of the REINVENT4 workflow while preserving its core training and optimization process.

This work relies on the original REINVENT4 implementation and the contributions of its authors and maintainers. Please cite the official REINVENT4 publication and repository when using or referencing this project.


## Repository Structure

- `00_data/`: dataset preparation and cleaning.
- `01_AD/`: applicability domain analysis.
- `01_cp/`: conformal prediction and random forest uncertainty modeling.
- `01_chemprop/`: Chemprop MVE training scripts.
- `02_RL/`: REINVENT4 reinforcement learning experiments and configs.
- `03_results/`: generated figures and summary plots.
- `presentationPlots.ipynb`: notebook used for thesis/presentation plots.

## Environment Setup

Create and activate the main conda environment:


```bash
conda env create -f chemprop.yaml
```
```bash
conda env create -f reinvent4.yaml
conda activate reinvent4
```


## How to Reproduce the Workflow

Run the pipeline in this order.

### 1) Data Preparation

Start from:

- `00_data/cleaningData.ipynb`

The datasets were downloaded form ChEMBL Version 34.

This step produces curated datasets under `00_data/egfr/` and `00_data/drd2/` used by downstream models.

Additionlly we also create the conda environemnts needed.

DISCLAIMER: to run reinvent you will need to install reinvent inside the reinvent4 conda env following the instructions present in `REINVENT4_README.md`

### 2) Train Uncertainty Models

#### 2.1 Chemprop MVE models

GPU cluster (SLURM) examples:

```bash
sbatch 01_chemprop/runChempropSuperSmall.sh
sbatch 01_chemprop/runChempropFull.sh
```

These scripts train MVE Chemprop regressors and save artifacts to the paths defined inside each script.

#### 2.2 Conformal prediction (RF + split conformal)

Script:

- `01_cp/conformerPredictionModel.ipynb`

Inside this notebook we create the conformal prediction model


### 3) Run REINVENT4 RL Experiments


- `02_RL/`

This folder contains all required configuration files, organized by experiment type, including Model System experiments, Chemprop experiments (point-estimation and averaged variants), and conformal prediction experiments.

For the Model System, configurations are provided for both the Unique Noisy Function and Multiple Noisy Functions settings, as well as the low-similarity and mid-similarity scenarios used in the master's thesis.

Main launcher scripts:

- `02_RL/runRL.sh` (GPU partition)
- `02_RL/runRLcpu.sh` (CPU partition)
- `02_RL/runMultiple.sh` (batch submit for a config folder)

Submit one RL run:

```bash
sbatch 02_RL/runRL.sh 02_RL/finalRL/multipleNoise_low/rl_doubleUnc_w0_SF_w1_N5.json
```

Submit multiple runs:

```bash
bash 02_RL/runMultiple.sh
```

Run locally without SLURM:

```bash
conda activate reinvent4
reinvent --log-level debug 02_RL/finalRL/multipleNoise_low/rl_doubleUnc_w0_SF_w1_N5.json
```

### 4) Collect and Plot Results

Main output locations:

- RL logs: `02_RL/00_reinventLogs/`
- RL experiment folders: `02_RL/finalRL/`
- figures used in the thesis: `03_results/`

For final figure generation and presentation plots, use:

- `presentationPlots.ipynb`

## Quick Reproduction Checklist

1. Create conda environment from `reinvent4.yaml`.
2. Prepare/clean datasets in `00_data/cleaningData.ipynb`.
3. Train Chemprop and/or conformal uncertainty models.
4. Select RL JSON configs in `02_RL/finalRL/.../csv/`.
5. Launch RL with `sbatch 02_RL/runRL.sh <config.json>`.
6. Aggregate and visualize results from `03_results/` and `presentationPlots.ipynb`.

## Thesis Context

The thesis studies how predictive uncertainty can be integrated into molecular RL in two ways:

1. As an explicit optimization objective in the scoring function.
2. As a modulation signal for policy updates.

Experiments include synthetic and real-world uncertainty settings (AD-like scenarios, Chemprop MVE, and conformal prediction), implemented in the REINVENT4 optimization pipeline.


## REINVENT4
### 1) Main changes in REINVENT4

In this project, I implemented new uncertainty-aware scoring components in REINVENT4.

Main plugin folder:

`reinvent_plugins/components/model_scoring_components`

All scripts in this folder are thesis-specific additions used to incorporate uncertainty into scoring.

#### 1.1 Core uncertainty components

- `comp_model_logp.py` (`NoisyLogP`): computes LogP and injects distance-dependent stochastic noise. The uncertainty scale increases with distance to training data.
- `comp_model_BertzCT.py` (`NoisyBertz`): same idea as `NoisyLogP`, but applied to BertzCT complexity.
- `comp_distances.py` (`UNCdistances`): computes distance-to-training (k-NN based) and returns a distance-derived weight to penalize out-of-domain molecules.

These components support tanimoto- and PCA-based distance variants through the shared helper functions in `compute_sims.py`.

#### 1.2 Chemprop uncertainty integration

- `comp_chemprop2.py` (`chemprop2`): Chemprop-based scoring component with MVE uncertainty estimation and calibration support.
- `chemprop2_extProccess.py`: external prediction script that outputs Chemprop predictions + calibrated uncertainties as JSON.
- `unc_chemprop2_extProccess.py`: external Chemprop script variant used for uncertainty-focused outputs.
- `chemprop2_extProccessG.py`: generalized external Chemprop inference script that loads checkpoints from a model folder and returns calibrated prediction/uncertainty JSON.
- `unc_chemprop2_extProccessG.py`: generalized uncertainty-focused variant that transforms uncertainty into a distance-like score.

#### 1.3 Classification conformal prediction components

- `comp_class_CP.py` (`classificationCP`): returns the active-class conformal p-value as a score.
- `comp_unc_class_CP.py` (`uncclassificationCP`): returns uncertainty-oriented weights from conformal prediction set confidence.


These components load a pre-trained conformal model artifact (`cloudpickle`) and compute conformal outputs from SMILES fingerprints.

#### 1.4 Shared utilities and examples

- `compute_sims.py`: shared similarity/distance utilities (Tanimoto, cosine, Euclidean in PCA space, optional distance transformation).


#### 1.5 Small modifcations of the whole REINVENT4 repo
- Adding the Average option - probabilistic scoring functions -  `reinvent/runmodes/RL/learning.py` line 136 and everything around average
- Adding the Loss Modulation inside REINVENT4:
    - `reinvent/runmodes/RL/reward.py`: LM (called reward in the config file)
    - `reinvent/runmodes/RL/reinvent`: weight retrieving and merging of the different scoring components
    - `reinvent/runmodes/RL/run_staged_learning.py`: sampling at the beginnign when necessary associated with each scoring component
    - `reinvent/runmodes/RL/learning.py`: minor modifications



#### 1.6 Notes

- The components above were used in the uncertainty experiments described in this thesis (model system, Chemprop, and conformal settings).
- Additional modifications were also made in other REINVENT4 files outside this folder to make the uncertainty work.




