#!/usr/bin/env python3
"""
MLP Training Script with Best Hyperparameters
Trains MLP model, evaluates on multiple random seeds, and reports averaged metrics.

This script trains a Multi-Layer Perceptron (MLP) regressor using pre-optimized
hyperparameters. It performs training and evaluation across multiple random seeds
to provide robust performance estimates.

Usage:
    python MLP_Train.py --data_path path/to/data.parquet --output_dir ./results --n_seeds 10

Requirements:
    - PyTorch
    - Scikit-learn
    - Pandas
    - NumPy

The script uses fixed hyperparameters and evaluates on test sets with different splits.
"""

import os
import sys
import gc
import random
import tempfile
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch import nn
from torch.amp import autocast, GradScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 42
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15

# Hardware config
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {DEVICE}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train MLP with best hyperparameters across multiple seeds")
    parser.add_argument("--data_path", type=str, default="data/4au_data_oriented.parquet",
                        help="Path to the input parquet data file")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Output directory for results and models")
    parser.add_argument("--best_params_path", type=str, default="mlp_best_hyperparams.json",
                        help="Path to the JSON file with best hyperparameters from MLP_FineTune.py")
    parser.add_argument("--n_seeds", type=int, default=10,
                        help="Number of random seeds to evaluate")
    parser.add_argument("--start_seed", type=int, default=42,
                        help="Starting seed number")
    return parser.parse_args()

args = parse_args()

# Load best hyperparameters from FineTune output
import json as _json
print(f"Loading best hyperparameters from {args.best_params_path}...")
with open(args.best_params_path, 'r') as _f:
    BEST_PARAMS = _json.load(_f)
print(f"Best hyperparameters: {BEST_PARAMS}")

# File paths
BASE_DIR = args.output_dir
DATA_PATH = args.data_path

# Output directories
OUTPUT_DIRS = {
    "models": os.path.join(BASE_DIR, "models"),
    "results": os.path.join(BASE_DIR, "results"),
}

# Feature definitions
INPUT_FEATURES = [
    "Br_vxf", "Br_vyf",
    "Cl_vxf", "Cl_vyf", "Cl_vzf",
    "F_vxf", "F_vyf", "F_vzf",
    "C_vxf",
    "H_vxf", "H_vyf", "H_vzf",
]

OUTPUT_FEATURES = [
    "Br_x0", "Br_y0", "Br_z0",
    "Cl_x0", "Cl_y0", "Cl_z0",
    "F_x0", "F_y0", "F_z0",
    "H_x0", "H_y0", "H_z0",
]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Create output directories
for d in OUTPUT_DIRS.values():
    os.makedirs(d, exist_ok=True)


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


set_seed(SEED)

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_and_prepare_data():
    """Load data and split into train/val/test sets."""
    print(f"Loading dataset from {DATA_PATH}...")
    
    # Load parquet with only needed columns
    df = pd.read_parquet(DATA_PATH, columns=INPUT_FEATURES + OUTPUT_FEATURES)
    print(f"Dataset loaded: {df.shape[0]:,} rows")

    X = df[INPUT_FEATURES].values.astype(np.float32)
    y = df[OUTPUT_FEATURES].values.astype(np.float32)
    del df
    gc.collect()

    # Split data
    print("Splitting data: 70% train, 15% val, 15% test...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(VAL_RATIO + TEST_RATIO), random_state=SEED, shuffle=True
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, shuffle=True
    )
    del X, y, X_temp, y_temp
    gc.collect()

    print(f"Split complete: Train={X_train.shape[0]:,}, Val={X_val.shape[0]:,}, Test={X_test.shape[0]:,}")

    # Apply normalization based on selected method
    normalization = BEST_PARAMS.get("normalization", "standard")
    
    if normalization == "minmax":
        print("Using MinMax normalization...")
        x_min = X_train.min(axis=0, keepdims=True)
        x_max = X_train.max(axis=0, keepdims=True)
        x_range = x_max - x_min + 1e-8
        
        y_min = y_train.min(axis=0, keepdims=True)
        y_max = y_train.max(axis=0, keepdims=True)
        y_range = y_max - y_min + 1e-8
        
        X_train_norm = ((X_train - x_min) / x_range).astype(np.float32)
        X_val_norm = ((X_val - x_min) / x_range).astype(np.float32)
        X_test_norm = ((X_test - x_min) / x_range).astype(np.float32)
        
        y_train_norm = ((y_train - y_min) / y_range).astype(np.float32)
        y_val_norm = ((y_val - y_min) / y_range).astype(np.float32)
        y_test_norm = ((y_test - y_min) / y_range).astype(np.float32)
        
        # Save scalers
        scaler_path = os.path.join(OUTPUT_DIRS["models"], "scalers.npz")
        np.savez(scaler_path, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
        print(f"Saved MinMax scalers to {scaler_path}")
        
        scalers = {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}
        
    else:  # standard normalization
        print("Using Standard normalization...")
        x_mean = X_train.mean(axis=0, keepdims=True)
        x_std = X_train.std(axis=0, keepdims=True) + 1e-8
        y_mean = y_train.mean(axis=0, keepdims=True)
        y_std = y_train.std(axis=0, keepdims=True) + 1e-8

        X_train_norm = ((X_train - x_mean) / x_std).astype(np.float32)
        X_val_norm = ((X_val - x_mean) / x_std).astype(np.float32)
        X_test_norm = ((X_test - x_mean) / x_std).astype(np.float32)

        y_train_norm = ((y_train - y_mean) / y_std).astype(np.float32)
        y_val_norm = ((y_val - y_mean) / y_std).astype(np.float32)
        y_test_norm = ((y_test - y_mean) / y_std).astype(np.float32)

        # Save scalers
        scaler_path = os.path.join(OUTPUT_DIRS["models"], "scalers.npz")
        np.savez(scaler_path, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
        print(f"Saved normalization scalers to {scaler_path}")
        
        scalers = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}

    # Move to GPU
    print("Transferring data to GPU...")
    train_X = torch.from_numpy(X_train_norm).to(DEVICE, non_blocking=True)
    train_y = torch.from_numpy(y_train_norm).to(DEVICE, non_blocking=True)
    val_X = torch.from_numpy(X_val_norm).to(DEVICE, non_blocking=True)
    val_y = torch.from_numpy(y_val_norm).to(DEVICE, non_blocking=True)
    test_X = torch.from_numpy(X_test_norm).to(DEVICE, non_blocking=True)
    test_y = torch.from_numpy(y_test_norm).to(DEVICE, non_blocking=True)

    del X_train, X_val, X_test, y_train, y_val, y_test
    del X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm
    gc.collect()
    torch.cuda.synchronize()

    mem_used = torch.cuda.memory_allocated() / 1e9
    print(f"GPU memory used for data: {mem_used:.2f} GB")

    return (train_X, train_y, val_X, val_y, test_X, test_y, scalers)


# ============================================================================
# GPU-NATIVE DATALOADER
# ============================================================================

class GPUTensorDataset:
    """Dataset that works entirely on GPU tensors."""
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y
        self.n_samples = X.shape[0]

    def __len__(self):
        return self.n_samples


class GPUDataLoader:
    """Custom dataloader that shuffles indices on GPU - zero CPU overhead."""
    def __init__(self, dataset: GPUTensorDataset, batch_size: int, shuffle: bool = True):
        self.X = dataset.X
        self.y = dataset.y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = dataset.n_samples
        self.n_batches = (self.n_samples + batch_size - 1) // batch_size

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.n_samples, device=self.X.device)
        else:
            indices = torch.arange(self.n_samples, device=self.X.device)

        for i in range(0, self.n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self.X[batch_indices], self.y[batch_indices]

    def __len__(self):
        return self.n_batches


# ============================================================================
# MODEL DEFINITION
# ============================================================================

class MLPRegressor(nn.Module):
    """Multi-layer perceptron with batch normalization."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list, 
                 dropout: float = 0.0, activation: str = "relu"):
        super().__init__()
        layers = []
        prev_dim = input_dim

        # Activation function
        act_fn = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "leaky_relu": lambda: nn.LeakyReLU(0.2),
        }.get(activation, nn.ReLU)

        # Hidden layers
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                act_fn() if callable(act_fn) else act_fn,
            ])
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        return self.net(x)


# ============================================================================
# TRAINING ENGINE
# ============================================================================

def train_one_epoch(model, loader, optimizer, scaler, grad_clip):
    """Single training epoch with AMP."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_X, batch_y in loader:
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", dtype=torch.float16):
            preds = model(batch_X)
            loss = nn.functional.mse_loss(preds, batch_y)

        if not torch.isfinite(loss):
            continue

        scaler.scale(loss).backward()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

    if num_batches == 0:
        return float('inf')
    return total_loss / num_batches


@torch.no_grad()
def validate(model, loader):
    """Validation with AMP."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch_X, batch_y in loader:
        with autocast(device_type="cuda", dtype=torch.float16):
            preds = model(batch_X)
            loss = nn.functional.mse_loss(preds, batch_y)

        if torch.isfinite(loss):
            total_loss += loss.item()
            num_batches += 1

    if num_batches == 0:
        return float('inf')
    return total_loss / num_batches


# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_metrics_auto(pred_csv_path, true_csv_path):
    """
    Automatically calculate MSE, R², ADE, MAE, and NRMSE for all atoms from separate
    prediction and target CSV files.
    Pred columns: P_<atom>_x0, P_<atom>_y0, P_<atom>_z0
    True columns: T_<atom>_x0, T_<atom>_y0, T_<atom>_z0
    """
    # Load both files
    pred_df = pd.read_csv(pred_csv_path)
    true_df = pd.read_csv(true_csv_path)
    
    # Identify all atoms by parsing column names from true_df
    # Columns are like: T_Br_x0, T_Cl_y0, T_F_z0, etc.
    coord_cols = [col for col in true_df.columns if col.startswith("T_") and col.endswith(("_x0", "_y0", "_z0"))]
    
    # Extract unique atoms (between T_ and _x0/_y0/_z0)
    atoms = sorted(set(col.split("_")[1] for col in coord_cols))
    
    results = {}
    overall_true = []
    overall_pred = []
    
    for atom in atoms:
        # Columns for this atom
        t_cols = [f"T_{atom}_x0", f"T_{atom}_y0", f"T_{atom}_z0"]
        p_cols = [f"P_{atom}_x0", f"P_{atom}_y0", f"P_{atom}_z0"]
        
        # Get data from both dataframes
        y_true = true_df[t_cols].to_numpy()
        y_pred = pred_df[p_cols].to_numpy()
        
        # Add to overall
        overall_true.append(y_true)
        overall_pred.append(y_pred)
        
        # Calculate ADE per atom
        distances = np.linalg.norm(y_true - y_pred, axis=1)
        ade = np.mean(distances)
        
        # Calculate MSE and R² (flattened)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate MAE (flattened)
        mae = mean_absolute_error(y_true, y_pred)
        
        # Calculate NRMSE (normalized by range)
        rmse = np.sqrt(mse)
        y_range = np.ptp(y_true)  # range (max - min)
        nrmse = rmse / y_range if y_range > 0 else 0
        
        results[atom] = {"MSE": mse, "R2": r2, "ADE": ade, "MAE": mae, "NRMSE": nrmse}
    
    # Overall metrics across all atoms
    overall_true = np.concatenate(overall_true, axis=1)
    overall_pred = np.concatenate(overall_pred, axis=1)
    overall_ade = np.mean(np.linalg.norm(overall_true - overall_pred, axis=1))
    overall_mse = mean_squared_error(overall_true, overall_pred)
    overall_r2 = r2_score(overall_true, overall_pred)
    overall_mae = mean_absolute_error(overall_true, overall_pred)
    overall_rmse = np.sqrt(overall_mse)
    overall_range = np.ptp(overall_true)
    overall_nrmse = overall_rmse / overall_range if overall_range > 0 else 0
    
    results["Overall"] = {"MSE": overall_mse, "R2": overall_r2, "ADE": overall_ade, 
                          "MAE": overall_mae, "NRMSE": overall_nrmse}
    
    return results


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    seeds = list(range(args.start_seed, args.start_seed + args.n_seeds))
    all_metrics = {}
    
    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'='*70}")
        print(f"Running with seed {seed} ({seed_idx+1}/{args.n_seeds})")
        print(f"{'='*70}")
        
        # Set seed globally for this run
        global SEED
        SEED = seed
        set_seed(SEED)
        
        # Load and prepare data
        train_X, train_y, val_X, val_y, test_X, test_y, scalers = load_and_prepare_data()
        
        # Create dataloaders
        train_dataset = GPUTensorDataset(train_X, train_y)
        val_dataset = GPUTensorDataset(val_X, val_y)
        test_dataset = GPUTensorDataset(test_X, test_y)
        
        train_loader = GPUDataLoader(train_dataset, BEST_PARAMS["batch_size"], shuffle=True)
        val_loader = GPUDataLoader(val_dataset, BEST_PARAMS["batch_size"], shuffle=False)
        test_loader = GPUDataLoader(test_dataset, BEST_PARAMS["batch_size"], shuffle=False)
        
        # Build model
        hidden_dims = [BEST_PARAMS["hidden_size"]] * BEST_PARAMS["n_layers"]
        model = MLPRegressor(
            input_dim=len(INPUT_FEATURES),
            output_dim=len(OUTPUT_FEATURES),
            hidden_dims=hidden_dims,
            dropout=BEST_PARAMS["dropout"],
            activation=BEST_PARAMS["activation"],
        ).to(DEVICE)
        
        # Try torch.compile for extra speed
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("Model compiled with torch.compile")
        except Exception as e:
            print(f"torch.compile not available: {e}")
        
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=BEST_PARAMS["lr"],
            weight_decay=BEST_PARAMS["weight_decay"],
            fused=True,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        grad_scaler = GradScaler()
        
        # Training loop
        print("\nStarting training...")
        best_val_mse = float('inf')
        best_epoch = 0
        best_state = None
        epochs_no_improve = 0
        
        for epoch in range(MAX_EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, grad_scaler, BEST_PARAMS["grad_clip"]
            )
            val_mse = validate(model, val_loader)
            
            scheduler.step(val_mse)
            
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if epoch % 10 == 0 or epochs_no_improve == 0:
                print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Val MSE: {val_mse:.6f} | Best: {best_val_mse:.6f}")
            
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break
        
        print(f"\nTraining completed for seed {seed}. Best validation MSE: {best_val_mse:.6f} at epoch {best_epoch}")
        
        # Load best model
        if best_state is not None:
            model.load_state_dict(best_state)
        
        # Generate predictions on test set
        print("\nGenerating predictions on test set...")
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                preds = model(batch_X)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Denormalize predictions and targets
        normalization = BEST_PARAMS.get("normalization", "standard")
        
        if normalization == "minmax":
            y_min = scalers["y_min"]
            y_max = scalers["y_max"]
            y_range = y_max - y_min + 1e-8
            
            all_preds_denorm = all_preds * y_range + y_min
            all_targets_denorm = all_targets * y_range + y_min
        else:
            y_mean = scalers["y_mean"]
            y_std = scalers["y_std"]
            
            all_preds_denorm = all_preds * y_std + y_mean
            all_targets_denorm = all_targets * y_std + y_mean
        
        # Create DataFrames with proper column names
        pred_cols = [f"P_{feat}" for feat in OUTPUT_FEATURES]
        target_cols = [f"T_{feat}" for feat in OUTPUT_FEATURES]
        
        pred_df = pd.DataFrame(all_preds_denorm, columns=pred_cols)
        target_df = pd.DataFrame(all_targets_denorm, columns=target_cols)
        
        # Calculate metrics
        # Create temp files for calculate_metrics_auto
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f_pred, \
             tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f_target:
            pred_df.to_csv(f_pred.name, index=False)
            target_df.to_csv(f_target.name, index=False)
            metrics = calculate_metrics_auto(f_pred.name, f_target.name)
        
        # Collect metrics
        for atom, vals in metrics.items():
            if atom not in all_metrics:
                all_metrics[atom] = {k: [] for k in vals}
            for k, v in vals.items():
                all_metrics[atom][k].append(v)
        
        # Save files only for the last seed
        if seed == seeds[-1]:
            # Save model
            model_path = os.path.join(OUTPUT_DIRS["models"], "best_model.pt")
            torch.save({
                'model_state_dict': best_state,
                'hyperparameters': BEST_PARAMS,
                'best_epoch': best_epoch,
                'best_val_mse': best_val_mse,
            }, model_path)
            print(f"Model saved to {model_path}")
            
            # Save predictions and targets (separate files)
            pred_path = os.path.join(OUTPUT_DIRS["results"], "single_test_predictions.csv")
            target_path = os.path.join(OUTPUT_DIRS["results"], "single_test_targets.csv")
            
            pred_df.to_csv(pred_path, index=False)
            target_df.to_csv(target_path, index=False)
            
            # Save combined file for EVAL notebooks (Metrics.ipynb / Plots.ipynb)
            combined_df = pd.concat([target_df, pred_df], axis=1)
            combined_path = os.path.join(OUTPUT_DIRS["results"], "mlp_test_full_data.csv")
            combined_df.to_csv(combined_path, index=False)
            
            print(f"Predictions saved to {pred_path}")
            print(f"Targets saved to {target_path}")
            print(f"Combined data saved to {combined_path}")
            
            # Print metrics for last run
            print("\n" + "="*70)
            print("METRICS EVALUATION (Last Seed)")
            print("="*70)
            
            # Print header
            print(f"{'Atom':<10} {'MSE':<12} {'R2':<12} {'ADE':<12} {'MAE':<12} {'NRMSE':<12}")
            print("-" * 70)
            
            # Print metrics for each atom
            for atom, vals in metrics.items():
                print(f"{atom:<10} {vals['MSE']:<12.5f} {vals['R2']:<12.5f} {vals['ADE']:<12.5f} "
                      f"{vals['MAE']:<12.5f} {vals['NRMSE']:<12.5f}")
            
            print("="*70)
    
    # After all seeds, compute and print averages
    print("\n" + "="*140)
    print(f"AVERAGE METRICS ACROSS {args.n_seeds} SEEDS")
    print("="*140)
    print(f"{'Atom':<10} {'MSE_mean':<12} {'MSE_std':<12} {'R2_mean':<12} {'R2_std':<12} {'ADE_mean':<12} {'ADE_std':<12} {'MAE_mean':<12} {'MAE_std':<12} {'NRMSE_mean':<12} {'NRMSE_std':<12}")
    print("-" * 140)
    
    for atom, vals in all_metrics.items():
        mse_vals = np.array(vals['MSE'])
        r2_vals = np.array(vals['R2'])
        ade_vals = np.array(vals['ADE'])
        mae_vals = np.array(vals['MAE'])
        nrmse_vals = np.array(vals['NRMSE'])
        
        print(f"{atom:<10} {mse_vals.mean():<12.5f} {mse_vals.std():<12.5f} {r2_vals.mean():<12.5f} {r2_vals.std():<12.5f} {ade_vals.mean():<12.5f} {ade_vals.std():<12.5f} {mae_vals.mean():<12.5f} {mae_vals.std():<12.5f} {nrmse_vals.mean():<12.5f} {nrmse_vals.std():<12.5f}")
    
    print("="*140)
    print("Averaging completed!")
    print("="*140)


if __name__ == "__main__":
    main()
