# Inverse Coulomb Explosion Imaging via Deep Learning

Recovering initial molecular geometry from fragment velocities measured in Coulomb Explosion Imaging (CEI) experiments using deep neural networks.

## Overview

Coulomb Explosion Imaging (CEI) is an ultrafast technique that probes molecular structure by ionizing all atoms nearly instantaneously and recording the resulting fragment momenta. The **inverse problem** — reconstructing the initial 3D atomic positions from the measured velocity vectors — is highly non-trivial due to the many-body Coulomb dynamics. This repository provides two complementary deep learning approaches:

| Model | Architecture | Description |
|-------|-------------|-------------|
| **MLP** | Multi-Layer Perceptron | Deterministic feedforward regression with batch normalization |
| **VAE** | Variational Autoencoder | Encoder–decoder with a learned latent space and reparameterization trick |

Both models are trained on an oriented molecular dataset (CHBrClF) and predict 3D initial positions (12 coordinates) from 12 measured velocity components.

## Molecular System

The target molecule is **bromochlorofluoromethane (CHBrClF)**, containing five atoms bonded to a central carbon:

```
     Br
      \
  Cl — C — F
      |
      H
```

**Input features (12):** Final velocity components of each fragment  
`Br_vxf, Br_vyf, Cl_vxf, Cl_vyf, Cl_vzf, F_vxf, F_vyf, F_vzf, C_vxf, H_vxf, H_vyf, H_vzf`

**Output targets (12):** Initial atomic positions  
`Br_x0, Br_y0, Br_z0, Cl_x0, Cl_y0, Cl_z0, F_x0, F_y0, F_z0, H_x0, H_y0, H_z0`

## Repository Structure

```
CEI/
├── README.md
├── MLP/
│   ├── MLP_FineTune.py      # Hyperparameter optimization with Optuna
│   └── MLP_Train.py         # Multi-seed training with best hyperparameters
├── VAE/
│   ├── VAE_FineTune.py      # Hyperparameter optimization with Optuna
│   └── VAE_Train.py         # Multi-seed training with best hyperparameters
└── EVAL/
    ├── Metrics.ipynb         # Per-atom and overall metric computation
    └── Plots.ipynb           # 3D visualization of true vs. predicted positions
```

## Pipeline

The workflow follows a three-stage pipeline for each model:

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│   1. Fine-Tune      │     │   2. Train           │     │   3. Evaluate        │
│   (FineTune.py)     │────▶│   (Train.py)         │────▶│   (EVAL/)            │
│                     │     │                      │     │                      │
│ • Optuna HPO        │     │ • Load best params   │     │ • Metrics.ipynb      │
│ • Minimize val MSE  │     │ • 10 random seeds    │     │ • Plots.ipynb        │
│ • Save best params  │     │ • Avg ± std metrics  │     │ • Per-atom analysis  │
│   to JSON           │     │ • Save predictions   │     │ • 3D visualization   │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
```

### Stage 1 — Hyperparameter Optimization (`*_FineTune.py`)

Uses [Optuna](https://optuna.org/) to search over model architecture and training hyperparameters. The objective minimizes **validation MSE** on a held-out 15% validation split.

**MLP search space:**

| Hyperparameter | Range |
|---------------|-------|
| Number of layers | 2 – 8 |
| Hidden size | 256 – 2048 (log) |
| Dropout | 0.0 – 0.5 |
| Learning rate | 1e-4 – 1e-2 (log) |
| Weight decay | 1e-6 – 1e-3 (log) |
| Batch size | {16384, 32768, 65536, 131072} |
| Gradient clipping | 0.5 – 3.0 |
| Activation | {ReLU, GELU, SiLU} |

**VAE search space:**

| Hyperparameter | Range |
|---------------|-------|
| Number of layers | 2 – 12 |
| Hidden size | 128 – 4096 (log) |
| Latent dimension | 8 – 512 (log) |
| Dropout | 0.0 – 0.6 |
| Learning rate | 5e-6 – 5e-3 (log) |
| Weight decay | 1e-7 – 1e-1 (log) |
| Batch size | {512, 1024, ..., 16384} |
| β (KL weight) | 1e-6 – 2.0 (log) |
| Gradient clipping | 0.0 – 10.0 |
| Scaler | {MinMax, Standard, Robust} |
| Activation | {LeakyReLU, GELU, SiLU} |

**Key features:**
- TPE / Hyperband pruning for efficient search
- GPU-resident data caching (data loaded once, reused across all trials)
- Mixed-precision training (FP16/BF16) for speed
- Graceful shutdown with progress saving
- SQLite-backed Optuna storage for resumable studies
- Trial-level CSV logging for post-hoc analysis

**Outputs:**
- `best_hyperparameters.json` / `mlp_best_hyperparams.json` — best hyperparameters
- `optuna_trials/*.csv` — full trial history
- `models/` — trial checkpoints
- `results/*.db` — Optuna study database

### Stage 2 — Multi-Seed Training (`*_Train.py`)

Loads the best hyperparameters from Stage 1 and trains the model across **10 independent random seeds** to produce robust performance estimates (mean ± std).

```bash
# MLP
python MLP_Train.py \
    --data_path data/4au_data_oriented.parquet \
    --best_params_path results/mlp_best_hyperparams.json \
    --output_dir ./results \
    --n_seeds 10

# VAE
python VAE_Train.py \
    --data_path data/data_oriented.parquet \
    --best_params_path results/best_hyperparameters.json \
    --output_dir ./results \
    --n_seeds 10
```

**Key features:**
- Same data split (70/15/15) across all seeds for fair comparison
- Early stopping with patience monitoring
- Per-seed and averaged metric reporting (MSE, R², ADE, MAE, NRMSE)
- Predictions saved in evaluation-ready format

**Outputs:**
- `results/mlp_test_full_data.csv` / `results/vae_test_full_data.csv` — combined true vs. predicted values
- `results/multi_seed_summary.json` — averaged metrics across seeds
- `models/best_model.pt` — final model checkpoint

### Stage 3 — Evaluation (`EVAL/`)

Jupyter notebooks that consume the CSV outputs from Stage 2.

**Metrics.ipynb** computes per-atom and overall metrics:

| Metric | Description |
|--------|-------------|
| MSE | Mean Squared Error |
| R² | Coefficient of Determination |
| ADE | Average Displacement Error (Euclidean distance in 3D) |
| NRMSE | Normalized Root Mean Squared Error (by data range) |

**Plots.ipynb** generates interactive 3D visualizations using Plotly:
- True atom positions (scatter)
- Predicted atom positions (scatter)
- True vs. Predicted overlay
- Average molecular geometry with bonds (true vs. predicted)

## Data Format

The input data is a Parquet file where each row represents one CEI event. Columns include 12 velocity features and 12 position targets. Data is split as:

| Split | Ratio | Purpose |
|-------|-------|---------|
| Train | 70% | Model fitting |
| Validation | 15% | Hyperparameter selection & early stopping |
| Test | 15% | Final performance evaluation |

Normalization is computed on the training set only and applied to all splits to prevent data leakage.

## Requirements

```
torch>=2.0
optuna>=3.0
scikit-learn
pandas
numpy
joblib
plotly
```

Install with:

```bash
pip install torch optuna scikit-learn pandas numpy joblib plotly
```

## Quick Start

### 1. Hyperparameter search

```bash
# MLP (50 Optuna trials)
cd MLP/
python MLP_FineTune.py \
    --data_path ../data/4au_data_oriented.parquet \
    --output_dir ./results \
    --n_trials 50

# VAE (50 Optuna trials)
cd VAE/
python VAE_FineTune.py \
    --data_path ../data/data_oriented.parquet \
    --output_dir ./results \
    --n_trials 50
```

### 2. Train with best hyperparameters (10 seeds)

```bash
# MLP
cd MLP/
python MLP_Train.py \
    --data_path ../data/4au_data_oriented.parquet \
    --best_params_path ./results/results/mlp_best_hyperparams.json \
    --output_dir ./results \
    --n_seeds 10

# VAE
cd VAE/
python VAE_Train.py \
    --data_path ../data/data_oriented.parquet \
    --best_params_path ./results/results/best_hyperparameters.json \
    --output_dir ./results \
    --n_seeds 10
```

### 3. Evaluate

Open the notebooks in `EVAL/` and set the `PATH` / `csv_file` variable to point to the output CSV from Step 2:

```python
# In Metrics.ipynb
csv_file = "../MLP/results/results/mlp_test_full_data.csv"

# In Plots.ipynb
PATH = "../MLP/results/results/mlp_test_full_data.csv"
```

### 4. Manual Split

These setup instructions assume that the entire dataset is stored in a single file.  
The train/validation/test split is performed randomly within the script.

If your data is structured differently (e.g., multiple files, pre-separated splits, or grouped data), only the data loading logic needs to be modified. The rest of the pipeline remains unchanged.

This setup was specifically used for the **"isomers" experiment**, where one isomer was excluded from the dataset during training. In this case, the split was applied after filtering out the selected isomer.

#### Key Assumptions

- All data is contained in a single file.
- Random splitting is sufficient for the experiment.
- No stratification is applied unless explicitly added.
- Filtering (e.g., excluding one isomer) occurs before splitting.

#### When to Modify

You need to adjust the data loading section if:

- The dataset is distributed across multiple files.
- The split is predefined.
- You require stratified or grouped splitting.
- The exclusion logic (e.g., isomer-based filtering) changes.

## Model Details

### MLP

A standard feedforward network with configurable depth and width:

```
Input (12) → [Linear → BatchNorm → Activation → Dropout] × N → Linear → Output (12)
```

- Trained with AdamW optimizer and ReduceLROnPlateau scheduler
- Mixed-precision (FP16) via `torch.amp`
- GPU-resident data with custom `GPUDataLoader` for zero CPU overhead

### VAE

An encoder–decoder architecture with a stochastic latent bottleneck:

```
         ┌──────────── μ ─────────────┐
Input → Encoder → ─┤                  ├─ z → Decoder → Output
         └──── log(σ²) ── reparam ───┘
```

- **Loss:** MSE (reconstruction) + β · KL divergence
- At inference, the mean μ is used directly (no sampling noise)
- Supports MinMax, Standard, and Robust scaling
- Multi-GPU support with thread-safe GPU slot allocation

## Hardware

Scripts are optimized for NVIDIA A40 GPUs but will run on any CUDA-capable device. CPU fallback is supported. Key optimizations:
- TF32 matrix multiplication on Ampere+ GPUs
- `torch.compile` with `reduce-overhead` mode (when available)
- Fused AdamW optimizer kernels
- Persistent GPU data caching across Optuna trials

## Citation

If you use this work, please cite:

```bibtex
@article{ghanaatian_molecular_cei,
  author  = {Ghanaatian, A.},
  title   = {Neural Network Based Molecular Structure Retrieval from Coulomb Explosion Imaging Data},
  journal = {<Journal Name>},
  year    = {2026},
  volume  = {<Volume>},
  pages   = {<Pages>},
  doi     = {<DOI>}
}
```
