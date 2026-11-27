# Uncertainty-Aware reinforcement learning for chemical language models

This repository contains the code and experiments for the article on uncertainty-aware reinforcement learning for de novo molecular design.
All paper-specific code, data and configs live inside the `UncertaintyAwareRLforCLM/` subdirectory (`00_data/`, `01_chemprop/`, `02_RL/`, …). Unless stated otherwise, every path in the "How to Reproduce" section below is relative to that subdirectory, **not** the repository root. The only commands run from the repository root are the package installs (`pip install -e .`) and downloading the prior (see below).

![RL Schematic](UncertaintyAwareRLforCLM/03_results/rlSchematic.png)


DISCLAIMER: If you want to check the `README.md` of REINVENT4 check `REINVENT4_README.md` 



## REINVENT4 Introduction and Credit

REINVENT4 is an open-source framework for molecular generation and multi-objective optimization. It uses a generative model (policy) to propose molecules and optimizes it with reinforcement learning based on scoring functions.

In this article, REINVENT4 is the baseline framework and has been extended with new uncertainty-aware functionalities. These additions are implemented on top of the REINVENT4 workflow while preserving its core training and optimization process.

This work relies on the original REINVENT4 implementation and the contributions of its authors and maintainers. Please cite also the official REINVENT4 publication and repository when using or referencing this project.


## Repository Structure

- `00_data/`: dataset preparation and cleaning.
- `01_cp/`: conformal prediction and random forest uncertainty modeling.
- `01_chemprop/`: Chemprop MVE training scripts.
- `02_RL/`: REINVENT4 reinforcement learning experiments and configs.
- `03_results/`: generated figures and summary plots.
- `figures.ipynb`: notebook used for article plots.

## Environment Setup

The Chemprop experiments use **two separate conda environments**:

- `reinvent4` (Python 3.10): runs REINVENT and its scoring components. REINVENT depends on legacy Chemprop 1.x; keep that dependency in this environment.
- `chemprop` (Python 3.11): runs Chemprop 2.2.1 training and the external MVE prediction scripts. Not needed for the model-system experiments.

The two are kept apart on purpose: the RL configs invoke Chemprop 2 in a separate process via `ExternalProcess`, so Chemprop 1.x and 2.x never have to share an interpreter. Do not install Chemprop 2 into the REINVENT environment.

> **Tested on:** Rocky Linux 8.10 (x86_64), conda 24.11.3, NVIDIA A100 (driver 570.x, CUDA 12.8). Any recent Linux + conda should work; match the PyTorch build to your own GPU driver (see below).

### Recommended: minimal install

This path is portable and not tied to our exact build versions. Run from the **repository root** (the directory containing this README and `pyproject.toml`):

```bash
# chemprop 2 environment
conda create -n chemprop python=3.11 pip
conda run -n chemprop python -m pip install "chemprop==2.2.1"

# Pin PyTorch < 2.6. Chemprop 2.2.1 predates the PyTorch 2.6 change that makes
# torch.load default to weights_only=True, which otherwise prevents reloading the
# trained checkpoint. Install a torch 2.5.x build that matches your hardware:
#   CPU-only (sufficient for the smoke test):
conda run -n chemprop python -m pip install "torch==2.5.1" \
    --index-url https://download.pytorch.org/whl/cpu
#   GPU: install the torch 2.5.x wheel for your CUDA driver instead (see pytorch.org).

# REINVENT environment (this repo's modified REINVENT, installed editable)
conda create -n reinvent4 python=3.10 pip
conda activate reinvent4
python -m pip install -e .
```

See [REINVENT4 installation instructions](REINVENT4_README.md) for optional dependencies; install notebook/analysis extras as needed.

### Fallback: exact environment exports

The repository also ships full conda exports of the environments we used, at `UncertaintyAwareRLforCLM/00_data/chemprop.yaml` and `UncertaintyAwareRLforCLM/00_data/reinvent4.yaml`. They pin exact build and version numbers from our system, so prefer the minimal install above for portability and reach for these only if you need our precise versions. Status on the tested system above:

- **`chemprop.yaml` — works as-is.** Reproduces the Chemprop environment we used (torch 2.9.1 / lightning 2.5.6) and installs and trains cleanly on both CPU and A100 (its pinned lightning handles the checkpoint reload, so no torch pin is needed here):
  ```bash
  conda env create -n chemprop -f UncertaintyAwareRLforCLM/00_data/chemprop.yaml
  ```
- **`reinvent4.yaml` — reference only; does not install standalone.** It pins `reinvent==4.6.27`, which is *this repository* (install it with `python -m pip install -e .` as above — it is not published on PyPI), and the licensed `openeye-toolkits` (optional — needs an OpenEye license and package channel). Use the minimal REINVENT install above and consult this file only to match specific dependency versions.

OpenEye is optional throughout; install it only if you use OpenEye-based functionality.

### Download the generative prior

Every RL config references a pretrained REINVENT prior at `priors/reinvent.prior` (relative to the repository root). The prior is **not** stored in this repository because of its size. Download it from Zenodo and save it under a `priors/` directory at the repository root:

```bash
mkdir -p priors
curl -L -o priors/reinvent.prior \
    https://zenodo.org/records/20701824/files/reinvent_pubchem.prior
```

The RL configs expect the filename `reinvent.prior`, so keep the `-o priors/reinvent.prior` target above (the Zenodo file is named `reinvent_pubchem.prior`). If you place it elsewhere, update `prior_file`/`agent_file` in the configs accordingly.

### Connect REINVENT to Chemprop 2

In **each** Chemprop `ExternalProcess` endpoint, replace the original `/home/$USER$/...` executable and machine-specific paths. Either call the Chemprop environment's Python directly:

```json
{
    "executable": "/absolute/path/to/envs/chemprop/bin/python",
    "args": "/absolute/path/to/repo/reinvent_plugins/components/model_scoring_components/chemprop2_extProccessG.py /absolute/path/to/model_directory",
    "property": "predictions"
}
```

This matches the `params` block shipped in the configs (`executable`, `args`, `property`). Prediction endpoints call `chemprop2_extProccessG.py`; uncertainty endpoints call `unc_chemprop2_extProccessG.py` on the same model directory — keep whichever script the original endpoint used.

Alternatively, set `executable` to the absolute path of your conda executable and prefix `args` with `run --no-capture-output -n chemprop python `. Quote individual paths within `args` if they contain spaces.

REINVENT appends a temporary SMILES-file path to these arguments and reads JSON predictions/uncertainties from stdout. The model directory must contain the trained checkpoints and `*calibrator.pkl` expected by the wrapper. Activate `reinvent4` to run RL; there is no need to activate `chemprop` in the same shell. Other paths in the RL config must also match your installation and working directory.

## How to Reproduce the Workflow

Run the pipeline in this order.

### 1) Data Preparation

Start from:

- `00_data/cleaningData.ipynb`

The datasets were downloaded from ChEMBL version 34 and are shipped gzip-compressed (`00_data/egfr/egfr.csv.gz`, `00_data/drd2/drd2.csv.gz`). The notebook decompresses them in its first code cell; if you run the cleaning steps outside the notebook, decompress first and keep the archive:

```bash
gunzip -k 00_data/egfr/egfr.csv.gz
gunzip -k 00_data/drd2/drd2.csv.gz
```

This step produces curated datasets under `00_data/egfr/` and `00_data/drd2/` used by downstream models. For the Chemprop EGFR experiments the key outputs are `cleaned_egfr.csv`, `cleaned_egfr_supersmall.csv`, and `cleaned_egfr_supersmall_smiles.csv` (the last is required by the `UNCdistances` scoring component).

Create the environments using the instructions above before running the notebook. Its environment-creation commands are commented out and do not install anything as supplied.

**Notebook caveats:** run the cells you need rather than the whole notebook top-to-bottom — the EGFR and DRD2 branches are independent, and a couple of intermediate `to_csv` calls (e.g. `cleaned_egfr_small.csv`) are commented out while later cells read those files. Run only the cells for the dataset you are reproducing.

### 2) Train Uncertainty Models

#### 2.1 Chemprop MVE models

GPU cluster (SLURM) examples — submit from inside `01_chemprop/` (the scripts use paths relative to that directory and write `--save-dir ./chemprop_model_*`):

```bash
cd 01_chemprop
sbatch runChempropSuperSmall.sh   # small/fast model, good for a smoke test
sbatch runChempropFull.sh         # full model used in the paper
```

These scripts train MVE Chemprop regressors and save artifacts to the paths defined inside each script. Notes:
- The scripts call `conda activate chemprop`; in a non-interactive SLURM shell you must first `source <conda>/etc/profile.d/conda.sh` (or adjust the script) for `conda activate` to work. Edit the `#SBATCH` partition/account/gres lines to match your cluster.
- CPU alternative (no GPU driver needed): run the same `chemprop train ...` command with `CUDA_VISIBLE_DEVICES=""`.

**Build the uncertainty calibrator (required).** After training, create the `mve_calibrator.pkl` that the external scoring scripts load from the model directory:

```bash
cd 01_chemprop
python createCalibrator.py ./chemprop_model_mve_egfr_supersmall   # or the full model dir
```

`createCalibrator.py` reads `../00_data/egfr/cleaned_egfr.csv` (validation split) and writes `mve_calibrator.pkl` into the model directory. The RL scoring component will fail to load if this file is missing.

#### 2.2 Conformal prediction (RF + split conformal)

Script:

- `01_cp/conformerPredictionModel.ipynb`

Inside this notebook we create the conformal prediction model


### 3) Run REINVENT4 RL Experiments


- `02_RL/`

This folder contains all required configuration files, organized by experiment type. The config subfolders are:

- `chemprop/`, `chempropAv/`, `chempropUn8/`, `chempropUn8Av/` — Chemprop experiments (point-estimate and averaged variants).
- `conformal/` — conformal prediction experiments.
- `multipleNoise_low/`, `multipleNoise_mid/`, `uniqueNoise_low/`, `uniqueNoise_mid/` — Model System experiments (Multiple vs Unique Noisy Functions, low- and mid-similarity scenarios).

**Run RL from inside `02_RL/`.** The config files use paths relative to that directory (`../../priors/reinvent.prior`, `../01_chemprop/...`, `../00_data/...`, `../../reinvent_plugins/...`), so they resolve only when `02_RL/` is the working directory. Make sure the prior has been downloaded (see *Download the generative prior*) and that the Chemprop model directories and calibrators exist.

Main launcher scripts (in `02_RL/`):

- `runRL.sh` (GPU partition) — the site-specific `#SBATCH --partition`/`--account` lines are commented placeholders; uncomment and set them (and adjust `--gres`) for your cluster. Create the log directory it writes to (`mkdir -p 02_RL/00_reinventLogs`) and ensure `conda activate` works in a batch shell (see the Chemprop training note).
- `runRLcpu.sh` (CPU partition).
- `runMultiple.sh` — batch-submit every `*.json` in a config folder: `bash runMultiple.sh <config_folder> [job_script]` (e.g. `bash runMultiple.sh chemprop runRL.sh`).

Submit one RL run (from `02_RL/`):

```bash
cd 02_RL
mkdir -p 00_reinventLogs
sbatch runRL.sh chemprop/rl_ChempropUnc_w0_SF_w1_N5.json
```

Run locally without SLURM (from `02_RL/`):

```bash
cd 02_RL
conda activate reinvent4
reinvent --log-level debug chemprop/rl_ChempropUnc_w0_SF_w1_N5.json
```

### 4) Collect and Plot Results

Main output locations:

- RL logs: `02_RL/00_reinventLogs/` (create it before the first run)
- RL experiment outputs: written next to each config under its `02_RL/<experiment>/` subfolder (e.g. `02_RL/chemprop/`), following the `tb_logdir`/`summary_csv_prefix` in each config
- figures used in the article: `03_results/`

For final figure generation and presentation plots, use:

- `figures.ipynb`

## Quick Reproduction Checklist

1. Install this repository in `reinvent4`; for Chemprop experiments, also create the separate Chemprop 2 environment (**pinning `torch<2.6`**) and update every external scorer's executable/script/model paths as described above.
2. Download the generative prior to `priors/reinvent.prior` (see *Download the generative prior*).
3. Prepare/clean datasets in `/UncertaintyAwareRLforCLM/00_data/cleaningData.ipynb` (decompress the `.gz` files first).
4. Train Chemprop and/or conformal uncertainty models, then build the calibrator (`createCalibrator.py <model_dir>`).
5. Select RL JSON configs from the subfolders of `/UncertaintyAwareRLforCLM/02_RL/` (e.g. `02_RL/chemprop/`, `02_RL/conformal/`, `02_RL/multipleNoise_low/`).
6. Launch RL from inside `02_RL/`, e.g. `sbatch runRL.sh chemprop/rl_ChempropUnc_w0_SF_w1_N5.json`.
7. Aggregate and visualize results from `03_results/` and `figures.ipynb`.

## Paper Context

The article studies how predictive uncertainty can be integrated into molecular RL in two ways:

1. As an explicit optimization objective in the scoring function.
2. As a modulation signal for policy updates.

Experiments include synthetic and real-world uncertainty settings (AD-like scenarios, Chemprop MVE, and conformal prediction), implemented in the REINVENT4 optimization pipeline.


## REINVENT4
### 1) Main changes in REINVENT4

We implemented new uncertainty-aware scoring components in REINVENT4.

Main plugin folder:

`reinvent_plugins/components/model_scoring_components`

All scripts in this folder are additions used to incorporate uncertainty into scoring.

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

- The components above were used in the uncertainty experiments described in the article (model system, Chemprop, and conformal settings).
- Additional modifications were also made in other REINVENT4 files outside this folder to make the uncertainty work.

## Citation

This work is currently submitted as a preprint on arXiv. If you use this code or build on this work, please cite: https://arxiv.org/abs/2606.24990


Please also cite the original REINVENT4 paper.




