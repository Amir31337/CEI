# ============================
# MLP_FineTune.py
# Hyperparameter optimization for MLP using Optuna
# For inverse Coulomb explosion imaging
# ============================

"""
This script performs hyperparameter optimization for a Multi-Layer Perceptron (MLP)
regressor using Optuna. It is designed for inverse problems in physics, such as
predicting initial atomic positions from velocity features.

Usage:
    python MLP_FineTune.py --data_path path/to/data.parquet --output_dir ./results --n_trials 50

Requirements:
    - PyTorch
    - Optuna
    - Scikit-learn
    - Pandas
    - NumPy

The script optimizes hyperparameters including number of layers, hidden size,
dropout, learning rate, etc., and saves the best model and results.
"""

import os
import json
import random
import time
import logging
import signal
import sys
import gc
import traceback
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.amp import autocast, GradScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# ============================================================================
# 1.CONFIGURATION AND REPRODUCIBILITY
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for MLP regressor")
    parser.add_argument("--data_path", type=str, default="data/4au_data_oriented.parquet",
                        help="Path to the input parquet data file")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Output directory for results and models")
    parser.add_argument("--n_trials", type=int, default=50,
                        help="Number of Optuna trials")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()

args = parse_args()

SEED = args.seed
N_TRIALS = args.n_trials
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15

# Hardware config
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {DEVICE}")

# File paths
BASE_DIR = args.output_dir
DATA_PATH = args.data_path

OUTPUT_DIRS = {
    "models": os.path.join(BASE_DIR, "models"),
    "results": os.path.join(BASE_DIR, "results"),
    "optuna": os.path.join(BASE_DIR, "optuna_trials"),
    "trial_models": os.path.join(BASE_DIR, "trial_models")
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

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIRS["results"], "mlp_training.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


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
# 2.GRACEFUL SHUTDOWN HANDLER
# ============================================================================

shutdown_requested = False


def signal_handler(sig, frame):
    global shutdown_requested
    signal_name = "SIGINT" if sig == signal.SIGINT else "SIGTERM"
    logger.warning(f"{signal_name} received.Requesting graceful shutdown...")
    shutdown_requested = True


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ============================================================================
# 3.GPU-RESIDENT DATA CACHE (KEY OPTIMIZATION)
# ============================================================================

_GPU_DATA_CACHE = {
    "loaded": False,
    "train_X": None,
    "train_y": None,
    "val_X": None,
    "val_y": None,
    "test_X": None,
    "test_y": None,
    "x_mean": None,
    "x_std": None,
    "y_mean": None,
    "y_std": None,
}

# ============================================================================
# 4.DATA LOADING - PRELOAD TO GPU
# ============================================================================

def preload_data_to_gpu():
    """Load data ONCE and keep it on GPU for all trials."""
    global _GPU_DATA_CACHE

    if _GPU_DATA_CACHE["loaded"]:
        return

    logger.info(f"Loading dataset from {DATA_PATH}...")

    # Load parquet with only needed columns
    df = pd.read_parquet(DATA_PATH, columns=INPUT_FEATURES + OUTPUT_FEATURES)
    logger.info(f"Dataset loaded: {df.shape[0]:,} rows")

    X = df[INPUT_FEATURES].values.astype(np.float32)
    y = df[OUTPUT_FEATURES].values.astype(np.float32)
    del df
    gc.collect()

    # Split data
    logger.info("Splitting data: 70% train, 15% val, 15% test...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(VAL_RATIO + TEST_RATIO), random_state=SEED, shuffle=True
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, shuffle=True
    )
    del X, y, X_temp, y_temp
    gc.collect()

    logger.info(f"Split complete: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")

    # Compute normalization statistics from training set ONLY
    logger.info("Computing normalization statistics from training set...")
    x_mean = X_train.mean(axis=0, keepdims=True)
    x_std = X_train.std(axis=0, keepdims=True) + 1e-8
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True) + 1e-8

    # Apply standardization
    X_train_norm = ((X_train - x_mean) / x_std).astype(np.float32)
    X_val_norm = ((X_val - x_mean) / x_std).astype(np.float32)
    X_test_norm = ((X_test - x_mean) / x_std).astype(np.float32)

    y_train_norm = ((y_train - y_mean) / y_std).astype(np.float32)
    y_val_norm = ((y_val - y_mean) / y_std).astype(np.float32)
    y_test_norm = ((y_test - y_mean) / y_std).astype(np.float32)

    # Save scalers
    scaler_path = os.path.join(OUTPUT_DIRS["models"], "scalers.npz")
    np.savez(scaler_path, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
    logger.info(f"Saved normalization scalers to {scaler_path}")

    # MOVE EVERYTHING TO GPU
    logger.info("Transferring data to GPU (this happens ONCE)...")
    _GPU_DATA_CACHE["train_X"] = torch.from_numpy(X_train_norm).to(DEVICE, non_blocking=True)
    _GPU_DATA_CACHE["train_y"] = torch.from_numpy(y_train_norm).to(DEVICE, non_blocking=True)
    _GPU_DATA_CACHE["val_X"] = torch.from_numpy(X_val_norm).to(DEVICE, non_blocking=True)
    _GPU_DATA_CACHE["val_y"] = torch.from_numpy(y_val_norm).to(DEVICE, non_blocking=True)
    _GPU_DATA_CACHE["test_X"] = torch.from_numpy(X_test_norm).to(DEVICE, non_blocking=True)
    _GPU_DATA_CACHE["test_y"] = torch.from_numpy(y_test_norm).to(DEVICE, non_blocking=True)

    # Store scalers for denormalization
    _GPU_DATA_CACHE["x_mean"] = x_mean
    _GPU_DATA_CACHE["x_std"] = x_std
    _GPU_DATA_CACHE["y_mean"] = y_mean
    _GPU_DATA_CACHE["y_std"] = y_std

    _GPU_DATA_CACHE["loaded"] = True

    del X_train, X_val, X_test, y_train, y_val, y_test
    del X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm
    gc.collect()
    torch.cuda.synchronize()

    mem_used = torch.cuda.memory_allocated() / 1e9
    logger.info(f"GPU memory used for data: {mem_used:.2f} GB")


# ============================================================================
# 5. GPU-NATIVE DATALOADER (NO CPU TRANSFERS)
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
# 6.MODEL DEFINITION
# ============================================================================

class MLPRegressor(nn.Module):
    """
    Multi-layer perceptron with batch normalization for inverse problems.
    """

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
# 7.TRAINING ENGINE (OPTIMIZED)
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
# 8.TRIAL TRACKER
# ============================================================================

class TrialResultsTracker:
    def __init__(self, output_dir: str):
        self.csv_path = os.path.join(output_dir, "mlp_trials.csv")
        self.columns = [
            "trial_number", "status", "start_time", "end_time", "duration_seconds",
            "n_layers", "hidden_size", "dropout", "lr", "weight_decay",
            "batch_size", "grad_clip", "activation",
            "best_epoch", "best_val_mse", "error_message", "model_path"
        ]
        self._init_csv()

    def _init_csv(self):
        if not os.path.exists(self.csv_path):
            pd.DataFrame(columns=self.columns).to_csv(self.csv_path, index=False)

    def log_trial(self, trial_data: Dict[str, Any]):
        row = {col: trial_data.get(col, None) for col in self.columns}
        pd.DataFrame([row]).to_csv(self.csv_path, mode='a', header=False, index=False)


trial_tracker = TrialResultsTracker(OUTPUT_DIRS["optuna"])


# ============================================================================
# 9.OPTUNA OBJECTIVE
# ============================================================================

def objective(trial: optuna.Trial) -> float:
    global shutdown_requested

    if shutdown_requested:
        raise optuna.exceptions.TrialPruned()

    start_time = datetime.now()
    trial_data = {
        "trial_number": trial.number,
        "start_time": start_time.isoformat(),
        "status": "running"
    }

    try:
        # Hyperparameter search space - LARGER batches for 45GB VRAM
        params = {
            "n_layers": trial.suggest_int("n_layers", 2, 8),
            "hidden_size": trial.suggest_int("hidden_size", 256, 2048, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16384, 32768, 65536, 131072]),
            "grad_clip": trial.suggest_float("grad_clip", 0.5, 3.0),
            "activation": trial.suggest_categorical("activation", ["relu", "gelu", "silu"]),
        }
        trial_data.update(params)

        logger.info(f"Trial {trial.number} | batch={params['batch_size']} hidden={params['hidden_size']} layers={params['n_layers']}")

        # Ensure data is on GPU
        preload_data_to_gpu()

        # Create GPU-native dataloaders
        train_dataset = GPUTensorDataset(_GPU_DATA_CACHE["train_X"], _GPU_DATA_CACHE["train_y"])
        val_dataset = GPUTensorDataset(_GPU_DATA_CACHE["val_X"], _GPU_DATA_CACHE["val_y"])
        train_loader = GPUDataLoader(train_dataset, params["batch_size"], shuffle=True)
        val_loader = GPUDataLoader(val_dataset, params["batch_size"], shuffle=False)

        # Build model
        hidden_dims = [params["hidden_size"]] * params["n_layers"]
        model = MLPRegressor(
            input_dim=len(INPUT_FEATURES),
            output_dim=len(OUTPUT_FEATURES),
            hidden_dims=hidden_dims,
            dropout=params["dropout"],
            activation=params["activation"],
        ).to(DEVICE)

        # Try torch.compile for extra speed
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception:
            pass

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            fused=True,  # Faster fused kernel
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        grad_scaler = GradScaler()

        best_val_mse = float('inf')
        best_epoch = 0
        best_state = None
        epochs_no_improve = 0

        for epoch in range(MAX_EPOCHS):
            if shutdown_requested:
                break

            train_loss = train_one_epoch(
                model, train_loader, optimizer, grad_scaler, params["grad_clip"]
            )
            val_mse = validate(model, val_loader)

            # Report to Optuna for pruning
            trial.report(val_mse, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            scheduler.step(val_mse)

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                break

            if epoch % 10 == 0:
                logger.info(f"  Epoch {epoch}: train_loss={train_loss:.6f} val_mse={val_mse:.6f}")

        # Save best model
        if best_state:
            model_path = os.path.join(OUTPUT_DIRS["trial_models"], f"mlp_trial_{trial.number}.pth")
            torch.save({'model_state_dict': best_state, 'params': params}, model_path)
            trial_data["model_path"] = model_path

        trial_data["status"] = "completed"
        trial_data["best_epoch"] = best_epoch
        trial_data["best_val_mse"] = best_val_mse

        logger.info(f"Trial {trial.number} completed: best_mse={best_val_mse:.6f} @ epoch {best_epoch}")
        return best_val_mse

    except optuna.exceptions.TrialPruned:
        trial_data["status"] = "pruned"
        raise
    except Exception as e:
        trial_data["status"] = "failed"
        trial_data["error_message"] = str(e)
        logger.error(f"Trial {trial.number} failed: {e}")
        logger.error(traceback.format_exc())
        raise optuna.exceptions.TrialPruned()
    finally:
        end_time = datetime.now()
        trial_data["end_time"] = end_time.isoformat()
        trial_data["duration_seconds"] = (end_time - start_time).total_seconds()
        trial_tracker.log_trial(trial_data)
        torch.cuda.empty_cache()


# ============================================================================
# 10. FINAL EVALUATION
# ============================================================================

def evaluate_best_model(best_params: Dict):
    """Retrain and evaluate best model on test set."""
    logger.info("Retraining best model with optimal hyperparameters...")

    preload_data_to_gpu()

    hidden_dims = [best_params["hidden_size"]] * best_params["n_layers"]
    model = MLPRegressor(
        input_dim=len(INPUT_FEATURES),
        output_dim=len(OUTPUT_FEATURES),
        hidden_dims=hidden_dims,
        dropout=best_params["dropout"],
        activation=best_params["activation"],
    ).to(DEVICE)

    train_dataset = GPUTensorDataset(_GPU_DATA_CACHE["train_X"], _GPU_DATA_CACHE["train_y"])
    val_dataset = GPUTensorDataset(_GPU_DATA_CACHE["val_X"], _GPU_DATA_CACHE["val_y"])
    train_loader = GPUDataLoader(train_dataset, best_params["batch_size"], shuffle=True)
    val_loader = GPUDataLoader(val_dataset, best_params["batch_size"], shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
        fused=True,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    grad_scaler = GradScaler()

    best_mse = float('inf')
    patience = 20
    counter = 0
    best_state = None

    for epoch in range(200):  # Extended training for final model
        train_one_epoch(model, train_loader, optimizer, grad_scaler, best_params["grad_clip"])
        val_mse = validate(model, val_loader)
        scheduler.step(val_mse)

        if val_mse < best_mse:
            best_mse = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            break

        if epoch % 10 == 0:
            logger.info(f"Retrain Epoch {epoch} - Val MSE: {val_mse:.6f}")

    # Load best weights
    if best_state:
        model.load_state_dict(best_state)

    # Test evaluation
    logger.info("Evaluating on test set...")
    model.eval()

    test_X = _GPU_DATA_CACHE["test_X"]
    test_y = _GPU_DATA_CACHE["test_y"]

    with torch.no_grad():
        with autocast(device_type="cuda", dtype=torch.float16):
            preds_norm = model(test_X).cpu().numpy()
        targets_norm = test_y.cpu().numpy()

    # Denormalize to original scale
    y_mean = _GPU_DATA_CACHE["y_mean"]
    y_std = _GPU_DATA_CACHE["y_std"]

    preds = preds_norm * y_std + y_mean
    targets = targets_norm * y_std + y_mean

    # Denormalize inputs to original scale
    x_mean = _GPU_DATA_CACHE["x_mean"]
    x_std = _GPU_DATA_CACHE["x_std"]
    test_X_orig = test_X.cpu().numpy() * x_std + x_mean

    # Compute metrics on original scale
    test_mse = mean_squared_error(targets.reshape(-1), preds.reshape(-1))
    test_r2 = r2_score(targets.reshape(-1), preds.reshape(-1))
    test_mae = mean_absolute_error(targets.reshape(-1), preds.reshape(-1))

    logger.info("=" * 60)
    logger.info("FINAL TEST RESULTS (ORIGINAL SCALE):")
    logger.info(f"  Best Val MSE (normalized): {best_mse:.6e}")
    logger.info(f"  Test MSE (original scale): {test_mse:.6e}")
    logger.info(f"  Test R² (original scale):  {test_r2:.6f}")
    logger.info(f"  Test MAE (original scale): {test_mae:.6e}")
    logger.info("=" * 60)

    # Save metrics
    metrics = {
        "best_val_mse_normalized": float(best_mse),
        "test_mse_original_scale": float(test_mse),
        "test_r2_original_scale": float(test_r2),
        "test_mae_original_scale": float(test_mae),
    }
    metrics_path = os.path.join(OUTPUT_DIRS["results"], "mlp_test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Saved test metrics to {metrics_path}")

    # Save final model checkpoint
    hidden_dims = [best_params["hidden_size"]] * best_params["n_layers"]
    checkpoint = {
        "input_dim": len(INPUT_FEATURES),
        "output_dim": len(OUTPUT_FEATURES),
        "hidden_dims": hidden_dims,
        "dropout": float(best_params["dropout"]),
        "activation": best_params["activation"],
        "state_dict": model.state_dict(),
        "hyperparams": best_params,
        "input_cols": INPUT_FEATURES,
        "target_cols": OUTPUT_FEATURES,
        "test_metrics": metrics,
    }

    model_path = os.path.join(OUTPUT_DIRS["models"], "best_mlp_model.pth")
    torch.save(checkpoint, model_path)
    logger.info(f"Saved best model checkpoint to {model_path}")

    # Save test set data: inputs, real targets, predictions
    test_df = pd.DataFrame(test_X_orig, columns=INPUT_FEATURES)
    test_df[[f"T_{c}" for c in OUTPUT_FEATURES]] = targets
    test_df[[f"P_{c}" for c in OUTPUT_FEATURES]] = preds
    test_csv_path = os.path.join(OUTPUT_DIRS["results"], "mlp_test_full_data.csv")
    test_df.to_csv(test_csv_path, index=False)
    logger.info(f"Saved test inputs, targets, and predictions to {test_csv_path}")

    logger.info("Training complete!")


# ============================================================================
# 11. MAIN
# ============================================================================

if __name__ == "__main__":
    logger.info(f"Starting Optimized MLP Training on {DEVICE}")

    # Pre-load data to GPU BEFORE optimization starts
    logger.info("Pre-loading all data to GPU...")
    preload_data_to_gpu()

    storage_path = os.path.join(OUTPUT_DIRS["results"], "mlp_optuna_study.db")

    sampler = TPESampler(multivariate=True, seed=SEED)
    pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=5)

    study = optuna.create_study(
        study_name="MLP_Inverse_Coulomb_Optimized",
        direction="minimize",
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    logger.info(f"Starting Optuna optimization with {N_TRIALS} trials.")
    logger.info(f"Existing trials: {len(study.trials)}")

    try:
        study.optimize(
            objective,
            n_trials=N_TRIALS,
            n_jobs=1,
            gc_after_trial=True,
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        logger.warning("Optimization interrupted. Saving progress...")
    except Exception as e:
        logger.exception(f"Optimization failed: {e}")

    # Save all trial results
    trials_df = study.trials_dataframe()
    trials_csv = os.path.join(OUTPUT_DIRS["optuna"], "mlp_optuna_trials.csv")
    trials_df.to_csv(trials_csv, index=False)
    logger.info(f"Saved {len(trials_df)} trials to {trials_csv}")

    if len(study.trials) == 0:
        logger.error("No trials completed.Cannot proceed.")
        sys.exit(1)

    # Best trial
    best_trial = study.best_trial
    best_params = best_trial.params
    logger.info("=" * 60)
    logger.info("BEST HYPERPARAMETERS FOUND:")
    logger.info(f"  Trial number: {best_trial.number}")
    logger.info(f"  Validation MSE: {best_trial.value:.6e}")
    for key, val in best_params.items():
        logger.info(f"  {key}: {val}")
    logger.info("=" * 60)

    # Save best hyperparameters
    hyperparam_path = os.path.join(OUTPUT_DIRS["results"], "mlp_best_hyperparams.json")
    with open(hyperparam_path, "w") as f:
        json.dump(best_params, f, indent=4)
    logger.info(f"Saved best hyperparameters to {hyperparam_path}")

    # Final evaluation
    evaluate_best_model(best_params)