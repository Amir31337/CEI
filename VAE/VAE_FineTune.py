
# ============================
# VAE_FineTune.py
# Hyperparameter optimization for VAE using Optuna
# For inverse Coulomb explosion imaging
# ============================

"""
This script performs hyperparameter optimization for a Variational Autoencoder (VAE)
regressor using Optuna. It is designed for inverse problems in physics, such as
predicting initial atomic positions from velocity features.

Usage:
    python VAE_FineTune.py --data_path path/to/data.parquet --output_dir ./results --n_trials 50

Requirements:
    - PyTorch
    - Optuna
    - Scikit-learn
    - Pandas
    - NumPy
    - Joblib

The script optimizes hyperparameters including latent dimension, hidden size,
dropout, learning rate, etc., and saves the best model and results.
"""

import os
import sys
import json
import logging
import traceback
import gc
import time
import argparse
from datetime import datetime
import threading
from contextlib import contextmanager
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
from optuna.trial import TrialState
from typing import List, Tuple, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION & GLOBAL CONSTANTS
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for VAE regressor")
    parser.add_argument("--data_path", type=str, default="data/data_oriented.parquet",
                        help="Path to the input parquet data file")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Output directory for results and models")
    parser.add_argument("--n_trials", type=int, default=50,
                        help="Number of Optuna trials")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()

args = parse_args()

# Hardware configuration - dynamic multi-GPU support (2x A100 80GB, 4 concurrent trials per GPU)
DETECTED_GPU_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0
AVAILABLE_GPU_IDS: List[int] = list(range(min(DETECTED_GPU_COUNT, 2))) if DETECTED_GPU_COUNT else []
MAX_TRIALS_PER_GPU = 4  # target 4 trials per A100 -> 8 total when 2 GPUs are present
TOTAL_PARALLEL_JOBS = max(1, len(AVAILABLE_GPU_IDS) * MAX_TRIALS_PER_GPU)
MAIN_DEVICE = torch.device(f"cuda:{AVAILABLE_GPU_IDS[0]}" if AVAILABLE_GPU_IDS else "cpu")

# Performance optimizations for high-end hardware
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # Auto-tune algorithms
    torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 on Ampere GPUs
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')  # Use TF32 for matrix multiplications

# File Paths
BASE_DIR = args.output_dir
DATA_PATH = args.data_path

OUTPUT_DIRS = {
    "models": os.path.join(BASE_DIR, "models"),
    "results": os.path.join(BASE_DIR, "results"),
    "optuna": os.path.join(BASE_DIR, "optuna_trials"),
    "trial_models": os.path.join(BASE_DIR, "trial_models")
}

# Feature Definitions
INPUT_FEATURES = [
    "Br_vxf", "Br_vyf", "Cl_vxf", "Cl_vyf", "Cl_vzf", 
    "F_vxf", "F_vyf", "F_vzf", "C_vxf", "H_vxf", "H_vyf", "H_vzf"
]

OUTPUT_FEATURES = [
    "Br_x0", "Br_y0", "Br_z0", "Cl_x0", "Cl_y0", "Cl_z0", 
    "F_x0", "F_y0", "F_z0", "H_x0", "H_y0", "H_z0"
]

# Data split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility
RANDOM_SEED = 42

# Ensure output directories exist
for d in OUTPUT_DIRS.values():
    os.makedirs(d, exist_ok=True)

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIRS["results"], "training.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ==========================================
# GLOBAL DATA CACHE (Avoid reloading 1M rows each trial)
# ==========================================

_DATA_CACHE = {
    "raw_data": None,  # (X, y) raw arrays
    "splits": None,    # (X_train, y_train, X_val, y_val, X_test, y_test) indices
    "scaled_data": {}  # {scaler_type: (X_train_s, X_val_s, X_test_s, scaler)}
}
DATA_CACHE_LOCK = threading.Lock()


# ==========================================
# GPU RESOURCE MANAGER (4 trials per GPU)
# ==========================================


class GPUResourceManager:
    """Thread-safe GPU slot allocator for Optuna parallel jobs."""

    def __init__(self, gpu_ids: List[int], max_per_gpu: int):
        self.gpu_ids = gpu_ids
        self.max_per_gpu = max_per_gpu
        self.slots = {gid: max_per_gpu for gid in gpu_ids}
        self.condition = threading.Condition()

    @contextmanager
    def reserve(self):
        # CPU fallback if no GPUs are available
        if not self.gpu_ids:
            yield None, torch.device("cpu")
            return

        with self.condition:
            while True:
                available_gpu = next((gid for gid, slots in self.slots.items() if slots > 0), None)
                if available_gpu is not None:
                    self.slots[available_gpu] -= 1
                    break
                self.condition.wait()

        device = torch.device(f"cuda:{available_gpu}")
        try:
            torch.cuda.set_device(available_gpu)
            yield available_gpu, device
        finally:
            with self.condition:
                self.slots[available_gpu] += 1
                self.condition.notify()


gpu_manager = GPUResourceManager(AVAILABLE_GPU_IDS, MAX_TRIALS_PER_GPU)

# ==========================================
# SCALER FACTORY
# ==========================================

def get_scaler(scaler_type: str):
    """Factory function to get scaler based on type."""
    scalers = {
        "minmax": MinMaxScaler(),
        "standard": StandardScaler(),
        "robust": RobustScaler()
    }
    return scalers.get(scaler_type, MinMaxScaler())


def get_activation_layer(name: str):
    """Return activation constructor based on name."""
    mapping = {
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "leaky_relu": lambda: nn.LeakyReLU(0.2)
    }
    return mapping.get(name, lambda: nn.LeakyReLU(0.2))

# ==========================================
# DATA LOADING & PROCESSING
# ==========================================

class PhysicsDataset(Dataset):
    """
    Custom Dataset for the Inverse Physics Problem.
    Optimized for large datasets with memory mapping.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_and_split_data(scaler_type: str = "minmax") -> Tuple[Tuple, Tuple, Tuple, Any]:
    """
    Loads data from parquet, splits into train/val/test (70/15/15),
    fits scaler on training data only, and transforms all sets.
    Only normalizes input features, outputs remain in physical units.
    
    Uses global cache to avoid reloading 1M rows for each Optuna trial.
    Only re-scales when scaler_type changes.
    """
    global _DATA_CACHE
    
    # Step 1: Load raw data (only once)
    if _DATA_CACHE["raw_data"] is None:
        with DATA_CACHE_LOCK:
            if _DATA_CACHE["raw_data"] is None:
                logger.info(f"Loading dataset from {DATA_PATH}...")
                try:
                    df = pd.read_parquet(DATA_PATH)
                    logger.info(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
                except Exception as e:
                    logger.error(f"Failed to load dataset: {e}")
                    sys.exit(1)

                X = df[INPUT_FEATURES].values
                y = df[OUTPUT_FEATURES].values
                _DATA_CACHE["raw_data"] = (X, y)
                logger.info(f"Input features: {len(INPUT_FEATURES)}, Output features: {len(OUTPUT_FEATURES)}")
                
                # Free the dataframe memory
                del df
                gc.collect()
    else:
        logger.info("Using cached raw data...")
    with DATA_CACHE_LOCK:
        X, y = _DATA_CACHE["raw_data"]
    
    # Step 2: Split data (only once, same split for all trials)
    if _DATA_CACHE["splits"] is None:
        with DATA_CACHE_LOCK:
            if _DATA_CACHE["splits"] is None:
                logger.info("Splitting data into train/val/test...")
                # Split: Train (70%), Temp (30%)
                X_train, X_temp, y_train, y_temp = train_test_split(
                    X, y, test_size=(VAL_RATIO + TEST_RATIO), random_state=RANDOM_SEED, shuffle=True
                )
                
                # Split Temp: Val (15%), Test (15%) - which is 50/50 of the 30%
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, shuffle=True
                )
                
                _DATA_CACHE["splits"] = (X_train, y_train, X_val, y_val, X_test, y_test)
                logger.info(f"Data split - Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
    else:
        logger.info("Using cached data splits...")
    with DATA_CACHE_LOCK:
        X_train, y_train, X_val, y_val, X_test, y_test = _DATA_CACHE["splits"]
    
    # Step 3: Scale data (cache per scaler_type)
    if scaler_type not in _DATA_CACHE["scaled_data"]:
        with DATA_CACHE_LOCK:
            if scaler_type not in _DATA_CACHE["scaled_data"]:
                logger.info(f"Fitting {scaler_type} scaler on training data...")
                scaler_X = get_scaler(scaler_type)
                
                X_train_s = scaler_X.fit_transform(X_train)
                X_val_s = scaler_X.transform(X_val)
                X_test_s = scaler_X.transform(X_test)
                
                _DATA_CACHE["scaled_data"][scaler_type] = (X_train_s, X_val_s, X_test_s, scaler_X)
                
                # Save Scaler for Inference
                scaler_path = os.path.join(OUTPUT_DIRS["models"], f"scaler_X_{scaler_type}.pkl")
                joblib.dump(scaler_X, scaler_path)
                logger.info(f"Scaler saved to {scaler_path}")
    else:
        logger.info(f"Using cached {scaler_type} scaled data...")
    with DATA_CACHE_LOCK:
        X_train_s, X_val_s, X_test_s, scaler_X = _DATA_CACHE["scaled_data"][scaler_type]
    
    # Outputs remain unnormalized (raw physical units)
    y_train_s = y_train
    y_val_s = y_val
    y_test_s = y_test

    return (X_train_s, y_train_s), (X_val_s, y_val_s), (X_test_s, y_test_s), scaler_X

# ==========================================
# MODEL ARCHITECTURE (INVERSE VAE)
# ==========================================

class InverseVAE(nn.Module):
    """
    Variational Autoencoder for Inverse Problems.
    Maps Input (Velocities) -> Latent -> Output (Positions).
    Flexible architecture with configurable layers and widths.
    """
    def __init__(self, input_dim: int, output_dim: int, latent_dim: int, 
                 hidden_size: int, n_layers: int, dropout_rate: float = 0.1,
                 activation: str = "leaky_relu"):
        super(InverseVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        activation_layer = get_activation_layer(activation)
        
        # Build hidden dims list with same size for all layers
        hidden_dims = [hidden_size] * n_layers
        
        # --- ENCODER ---
        encoder_layers = []
        in_channels = input_dim
        
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_channels, h_dim))
            encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(activation_layer())
            encoder_layers.append(nn.Dropout(dropout_rate))
            in_channels = h_dim
            
        self.encoder_body = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # --- DECODER ---
        decoder_layers = []
        # Reverse hidden dims for symmetry
        hidden_dims_rev = hidden_dims[::-1]
        in_channels = latent_dim
        
        for h_dim in hidden_dims_rev:
            decoder_layers.append(nn.Linear(in_channels, h_dim))
            decoder_layers.append(nn.BatchNorm1d(h_dim))
            decoder_layers.append(activation_layer())
            decoder_layers.append(nn.Dropout(dropout_rate))
            in_channels = h_dim
            
        self.decoder_body = nn.Sequential(*decoder_layers)
        self.fc_out = nn.Linear(hidden_dims_rev[-1], output_dim)

    def reparameterize(self, mu, logvar):
        """
        The Reparameterization Trick: z = mu + sigma * epsilon
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x):
        # Encode
        enc_hidden = self.encoder_body(x)
        mu = self.fc_mu(enc_hidden)
        logvar = self.fc_var(enc_hidden)
        
        # Sample
        z = self.reparameterize(mu, logvar)
        
        # Decode
        dec_hidden = self.decoder_body(z)
        reconstruction = self.fc_out(dec_hidden)
        
        return reconstruction, mu, logvar
    
    def predict(self, x):
        """For inference, use mean of latent space (no sampling noise)."""
        self.eval()
        with torch.no_grad():
            enc_hidden = self.encoder_body(x)
            mu = self.fc_mu(enc_hidden)
            dec_hidden = self.decoder_body(mu)
            reconstruction = self.fc_out(dec_hidden)
        return reconstruction

def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    Loss = MSE (Reconstruction) + Beta * KL Divergence
    Returns per-sample averaged loss for stability.
    """
    batch_size = recon_x.size(0)
    
    # Reconstruction loss (MSE) - per sample average
    MSE = nn.functional.mse_loss(recon_x, x, reduction='mean')
    
    # KL Divergence - per sample average with clamped logvar for stability
    logvar = torch.clamp(logvar, min=-20.0, max=10.0)
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    return MSE + (beta * KLD), MSE, KLD


# ==========================================
# TRAINING ENGINE
# ==========================================

def train_one_epoch(model, loader, optimizer, scaler, beta, grad_clip, device):
    """Train for one epoch with comprehensive error handling."""
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_kld = 0.0
    num_batches = 0
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float32
    
    for batch_X, batch_y in loader:
        try:
            batch_X, batch_y = batch_X.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Mixed Precision Forward Pass
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                recon_y, mu, logvar = model(batch_X)
                loss, mse, kld = vae_loss_function(recon_y, batch_y, mu, logvar, beta=beta)
            
            # Check for NaN/Inf in loss
            if not torch.isfinite(loss):
                logger.warning("NaN/Inf loss detected in batch, skipping...")
                continue

            # Backward Pass with Scaling
            scaler.scale(loss).backward()
            
            # Gradient Clipping
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            total_mse += mse.item()
            total_kld += kld.item()
            num_batches += 1
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("OOM in batch, clearing cache and continuing...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            else:
                raise e
    
    if num_batches == 0:
        return float('inf'), float('inf'), float('inf')
        
    return total_loss / num_batches, total_mse / num_batches, total_kld / num_batches


def validate(model, loader, beta, device):
    """Validate model with error handling."""
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    num_batches = 0
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float32
    
    with torch.no_grad():
        for batch_X, batch_y in loader:
            try:
                batch_X, batch_y = batch_X.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
                
                with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    recon_y, mu, logvar = model(batch_X)
                    loss, mse, _ = vae_loss_function(recon_y, batch_y, mu, logvar, beta=beta)
                
                if torch.isfinite(loss):
                    total_loss += loss.item()
                    total_mse += mse.item()
                    num_batches += 1
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
    
    if num_batches == 0:
        return float('inf'), float('inf')
        
    return total_loss / num_batches, total_mse / num_batches

# ==========================================
# TRIAL RESULTS TRACKER
# ==========================================

class TrialResultsTracker:
    """Comprehensive tracker for all trial results with CSV logging."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.csv_path = os.path.join(output_dir, "trials_comprehensive.csv")
        self.columns = [
            "trial_number", "status", "start_time", "end_time", "duration_seconds",
            "n_layers", "hidden_size", "latent_dim", "dropout", "lr", "weight_decay",
            "batch_size", "beta", "grad_clip", "scaler_type", "activation", "gpu_id",
            "best_epoch", "final_train_loss", "final_val_loss", "best_val_mse",
            "error_message", "model_path"
        ]
        self._init_csv()
    
    def _init_csv(self):
        """Initialize CSV with headers if it doesn't exist."""
        if not os.path.exists(self.csv_path):
            df = pd.DataFrame(columns=self.columns)
            df.to_csv(self.csv_path, index=False)
        else:
            existing = pd.read_csv(self.csv_path, nrows=1)
            if list(existing.columns) != self.columns:
                df = pd.DataFrame(columns=self.columns)
                df.to_csv(self.csv_path, index=False)
    
    def log_trial(self, trial_data: Dict[str, Any]):
        """Append trial results to CSV."""
        # Ensure all columns are present
        row = {col: trial_data.get(col, None) for col in self.columns}
        df = pd.DataFrame([row])
        df.to_csv(self.csv_path, mode='a', header=False, index=False)
        logger.info(f"Trial {trial_data.get('trial_number')} logged to {self.csv_path}")


# Global tracker
trial_tracker = TrialResultsTracker(OUTPUT_DIRS["optuna"])


# ==========================================
# OPTUNA OPTIMIZATION
# ==========================================

def create_data_loaders(X_train, y_train, X_val, y_val, batch_size: int):
    """Create optimized data loaders for large datasets."""
    train_dataset = PhysicsDataset(X_train, y_train)
    val_dataset = PhysicsDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=False,  # Disabled to avoid "too many open files" error
        prefetch_factor=2,
        drop_last=True  # Avoid small last batch issues with BatchNorm
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=False,  # Disabled to avoid "too many open files" error
        prefetch_factor=2
    )
    
    return train_loader, val_loader


def objective(trial):
    """Optuna objective function with comprehensive error handling."""
    start_time = datetime.now()
    trial_data = {
        "trial_number": trial.number,
        "start_time": start_time.isoformat(),
        "status": "running"
    }
    
    train_loader = None
    val_loader = None
    model = None
    
    try:
        with gpu_manager.reserve() as (gpu_id, device):
            trial_data["gpu_id"] = gpu_id

            # 1. Hyperparameter Sampling (expanded search space)
            params = {
                "n_layers": trial.suggest_int("n_layers", 2, 12),
                "hidden_size": trial.suggest_int("hidden_size", 128, 4096, log=True),
                "latent_dim": trial.suggest_int("latent_dim", 8, 512, log=True),
                "dropout": trial.suggest_float("dropout", 0.0, 0.6),
                "lr": trial.suggest_float("lr", 5e-6, 5e-3, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-1, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [512, 1024, 2048, 4096, 6144, 8192, 12288, 16384]),
                "beta": trial.suggest_float("beta", 1e-6, 2.0, log=True),
                "grad_clip": trial.suggest_float("grad_clip", 0.0, 10.0),
                "scaler_type": trial.suggest_categorical("scaler_type", ["minmax", "standard", "robust"]),
                "activation": trial.suggest_categorical("activation", ["leaky_relu", "gelu", "silu"])
            }
            
            # Update trial data with params
            trial_data.update(params)
            
            logger.info(f"Trial {trial.number} (GPU {gpu_id}) - Params: {params}")

            # 2. Load data with selected scaler
            (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler = load_and_split_data(
                scaler_type=params["scaler_type"]
            )

            # 3. Create Data Loaders
            train_loader, val_loader = create_data_loaders(
                X_train, y_train, X_val, y_val, params["batch_size"]
            )

            # 4. Model Setup
            model = InverseVAE(
                input_dim=len(INPUT_FEATURES),
                output_dim=len(OUTPUT_FEATURES),
                latent_dim=params["latent_dim"],
                hidden_size=params["hidden_size"],
                n_layers=params["n_layers"],
                dropout_rate=params["dropout"],
                activation=params["activation"]
            ).to(device)
            
            # Count parameters
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Trial {trial.number} - Model has {num_params:,} trainable parameters")
            
            optimizer = optim.AdamW(
                model.parameters(), 
                lr=params["lr"], 
                weight_decay=params["weight_decay"]
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
            )
            grad_scaler = GradScaler(enabled=(device.type == "cuda"))

            # 5. Training Loop
            best_val_mse = float('inf')
            best_epoch = 0
            early_stop_counter = 0
            early_stop_patience = 15
            epochs = 120
            best_model_state = None

            for epoch in range(epochs):
                try:
                    # Training
                    train_loss, train_mse, train_kld = train_one_epoch(
                        model, train_loader, optimizer, grad_scaler, 
                        params["beta"], params["grad_clip"], device
                    )
                    
                    # Validation
                    val_loss, val_mse = validate(model, val_loader, params["beta"], device)
                    
                    # Check for NaN/Inf
                    if not np.isfinite(val_loss) or not np.isfinite(train_loss):
                        logger.warning(f"Trial {trial.number}: NaN/Inf detected at epoch {epoch}.")
                        trial_data["status"] = "failed_nan"
                        trial_data["error_message"] = "NaN/Inf in loss"
                        raise optuna.exceptions.TrialPruned()

                    scheduler.step(val_loss)

                    # Optuna Pruning
                    trial.report(val_mse, epoch)
                    if trial.should_prune():
                        trial_data["status"] = "pruned"
                        raise optuna.exceptions.TrialPruned()

                    # Track best model
                    if val_mse < best_val_mse:
                        best_val_mse = val_mse
                        best_epoch = epoch
                        early_stop_counter = 0
                        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    else:
                        early_stop_counter += 1
                    
                    # Log progress every 10 epochs
                    if (epoch + 1) % 10 == 0:
                        logger.info(f"Trial {trial.number} Epoch {epoch+1}/{epochs} | "
                                   f"T_Loss: {train_loss:.6f} | V_MSE: {val_mse:.6f} | Best: {best_val_mse:.6f}")
                    
                    if early_stop_counter >= early_stop_patience:
                        logger.info(f"Trial {trial.number}: Early stopping at epoch {epoch+1}")
                        break
                        
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.error(f"Trial {trial.number}: CUDA OOM at epoch {epoch}")
                        trial_data["status"] = "failed_oom"
                        trial_data["error_message"] = str(e)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        raise optuna.exceptions.TrialPruned()
                    else:
                        raise e

            # Save trial model
            if best_model_state is not None:
                model_path = os.path.join(OUTPUT_DIRS["trial_models"], f"trial_{trial.number}_model.pth")
                torch.save({
                    'model_state_dict': best_model_state,
                    'params': params,
                    'best_val_mse': best_val_mse,
                    'best_epoch': best_epoch,
                    'scaler_type': params["scaler_type"]
                }, model_path)
                trial_data["model_path"] = model_path
                
            # Update trial data
            trial_data["status"] = "completed"
            trial_data["best_epoch"] = best_epoch
            trial_data["final_train_loss"] = train_loss
            trial_data["final_val_loss"] = val_loss
            trial_data["best_val_mse"] = best_val_mse
            
            return best_val_mse
        
    except optuna.exceptions.TrialPruned:
        raise
        
    except Exception as e:
        logger.error(f"Trial {trial.number} failed with error: {str(e)}")
        logger.error(traceback.format_exc())
        trial_data["status"] = "failed_error"
        trial_data["error_message"] = str(e)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise optuna.exceptions.TrialPruned()
        
    finally:
        # Record end time and duration
        end_time = datetime.now()
        trial_data["end_time"] = end_time.isoformat()
        trial_data["duration_seconds"] = (end_time - start_time).total_seconds()
        
        # Log trial to CSV
        trial_tracker.log_trial(trial_data)
        
        # Cleanup
        if train_loader is not None:
            del train_loader
        if val_loader is not None:
            del val_loader
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ==========================================
# FINAL EVALUATION FUNCTION
# ==========================================

def evaluate_best_model(best_params: Dict, output_path: str, device: torch.device = MAIN_DEVICE):
    """
    Evaluate the best model on test set and save comprehensive results to text file.
    """
    logger.info("=" * 60)
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info("=" * 60)
    
    # Load data with best scaler type
    scaler_type = best_params.get("scaler_type", "minmax")
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler = load_and_split_data(
        scaler_type=scaler_type
    )
    
    # Recreate model with best architecture
    model = InverseVAE(
        input_dim=len(INPUT_FEATURES),
        output_dim=len(OUTPUT_FEATURES),
        latent_dim=best_params["latent_dim"],
        hidden_size=best_params["hidden_size"],
        n_layers=best_params["n_layers"],
        dropout_rate=best_params["dropout"],
        activation=best_params.get("activation", "leaky_relu")
    ).to(device)
    
    # Load best model weights
    best_model_path = os.path.join(OUTPUT_DIRS["models"], "best_model_final.pth")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Prepare test data
    test_dataset = PhysicsDataset(X_test, y_test)
    test_loader = DataLoader(
        test_dataset, batch_size=4096, shuffle=False,
        num_workers=4, pin_memory=True, prefetch_factor=2
    )
    
    # Run inference
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            
            # Use predict method (uses mean, no sampling)
            preds = model.predict(batch_X)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.numpy())

    np_preds = np.vstack(all_preds)
    np_targets = np.vstack(all_targets)

    # Calculate comprehensive metrics
    overall_mse = mean_squared_error(np_targets, np_preds)
    overall_rmse = np.sqrt(overall_mse)
    overall_mae = mean_absolute_error(np_targets, np_preds)
    overall_r2 = r2_score(np_targets, np_preds)
    
    # Per-feature metrics
    feature_metrics = {}
    for i, col in enumerate(OUTPUT_FEATURES):
        mse = mean_squared_error(np_targets[:, i], np_preds[:, i])
        feature_metrics[col] = {
            "mse": mse,
            "rmse": np.sqrt(mse),
            "mae": mean_absolute_error(np_targets[:, i], np_preds[:, i]),
            "r2": r2_score(np_targets[:, i], np_preds[:, i])
        }

    # Generate comprehensive report
    report_lines = [
        "=" * 80,
        "BEST MODEL EVALUATION REPORT",
        "=" * 80,
        f"Report Generated: {datetime.now().isoformat()}",
        "",
        "=" * 80,
        "HYPERPARAMETERS",
        "=" * 80,
    ]
    
    for key, value in best_params.items():
        report_lines.append(f"  {key}: {value}")
    
    report_lines.extend([
        "",
        "=" * 80,
        "DATA SPLIT INFORMATION",
        "=" * 80,
        f"  Train samples: {len(X_train):,}",
        f"  Validation samples: {len(X_val):,}",
        f"  Test samples: {len(X_test):,}",
        f"  Input features: {len(INPUT_FEATURES)}",
        f"  Output features: {len(OUTPUT_FEATURES)}",
        f"  Scaler type: {scaler_type}",
        "",
        "=" * 80,
        "OVERALL TEST METRICS",
        "=" * 80,
        f"  MSE:  {overall_mse:.8f}",
        f"  RMSE: {overall_rmse:.8f}",
        f"  MAE:  {overall_mae:.8f}",
        f"  R2:   {overall_r2:.8f}",
        "",
        "=" * 80,
        "PER-FEATURE METRICS",
        "=" * 80,
    ])
    
    for col, metrics in feature_metrics.items():
        report_lines.extend([
            f"\n  {col}:",
            f"    MSE:  {metrics['mse']:.8f}",
            f"    RMSE: {metrics['rmse']:.8f}",
            f"    MAE:  {metrics['mae']:.8f}",
            f"    R2:   {metrics['r2']:.8f}",
        ])
    
    report_lines.extend([
        "",
        "=" * 80,
        "MODEL ARCHITECTURE",
        "=" * 80,
        f"  Number of layers: {best_params['n_layers']}",
        f"  Hidden size: {best_params['hidden_size']}",
        f"  Latent dimension: {best_params['latent_dim']}",
        f"  Dropout rate: {best_params['dropout']}",
        "",
        "=" * 80,
        "FILES SAVED",
        "=" * 80,
        f"  Best model: {best_model_path}",
        f"  Scaler: {os.path.join(OUTPUT_DIRS['models'], f'scaler_X_{scaler_type}.pkl')}",
        f"  This report: {output_path}",
        "",
        "=" * 80,
    ])
    
    # Write report
    report_text = "\n".join(report_lines)
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    logger.info(f"Evaluation report saved to: {output_path}")
    logger.info(f"\nTest Results: MSE={overall_mse:.8f}, RMSE={overall_rmse:.8f}, MAE={overall_mae:.8f}, R2={overall_r2:.8f}")
    
    # Also save as JSON for programmatic access
    metrics_json = {
        "hyperparameters": best_params,
        "overall_metrics": {
            "mse": overall_mse,
            "rmse": overall_rmse,
            "mae": overall_mae,
            "r2": overall_r2
        },
        "feature_metrics": feature_metrics,
        "data_info": {
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test)
        }
    }
    
    json_path = output_path.replace('.txt', '.json')
    with open(json_path, 'w') as f:
        json.dump(metrics_json, f, indent=4)
    
    # Save predictions
    df_pred = pd.DataFrame(np_preds, columns=[f"P_{c}" for c in OUTPUT_FEATURES])
    df_true = pd.DataFrame(np_targets, columns=[f"T_{c}" for c in OUTPUT_FEATURES])
    df_final = pd.concat([df_true, df_pred], axis=1)
    pred_path = os.path.join(OUTPUT_DIRS["results"], "test_predictions.csv")
    df_final.to_csv(pred_path, index=False)
    logger.info(f"Predictions saved to: {pred_path}")
    
    return overall_mse, overall_r2


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info(f"Starting VAE Inverse Problem Solver on {MAIN_DEVICE}")
    logger.info(f"Detected GPUs: {AVAILABLE_GPU_IDS} | Parallel slots: {TOTAL_PARALLEL_JOBS}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    if MAIN_DEVICE.type == "cuda":
        torch.cuda.set_device(MAIN_DEVICE.index)
    
    # Print configuration
    logger.info(f"Data path: {DATA_PATH}")
    logger.info(f"Train/Val/Test split: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
    logger.info(f"Input features: {len(INPUT_FEATURES)}")
    logger.info(f"Output features: {len(OUTPUT_FEATURES)}")

    # Run Optuna Optimization
    logger.info("\n" + "=" * 60)
    logger.info("Starting Hyperparameter Optimization...")
    logger.info("=" * 60)
    
    # Optuna storage path
    storage_path = os.path.join(OUTPUT_DIRS["results"], "optuna_study_vae.db")
    storage = f"sqlite:///{storage_path}"
    logger.info(f"Optuna database: {storage_path}")
    
    # Create study with pruning
    study = optuna.create_study(
        study_name="VAE_Inverse_Physics_Comprehensive",
        direction="minimize",  # Minimize validation MSE
        pruner=optuna.pruners.HyperbandPruner(min_resource=5, max_resource=100, reduction_factor=3),
        storage=storage,
        load_if_exists=True
    )
    
    # Run optimization (default to 8 trials for sanity check, override with N_TRIALS env)
    n_trials = args.n_trials
    logger.info(f"Running {n_trials} trial(s)...")
    
    study.optimize(
        objective, 
        n_trials=n_trials, 
        timeout=None, 
        n_jobs=TOTAL_PARALLEL_JOBS,  # 4 trials per GPU, adjusted to detected GPUs
        show_progress_bar=True,
        catch=(Exception,)  # Catch all exceptions to continue
    )

    logger.info("\n" + "=" * 60)
    logger.info("Optimization Finished!")
    logger.info("=" * 60)
    
    # Get completed trials
    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    logger.info(f"Completed trials: {len(completed_trials)} / {len(study.trials)}")
    
    if len(completed_trials) == 0:
        logger.error("No trials completed successfully. Check logs for errors.")
        sys.exit(1)
    
    logger.info(f"Best Trial: {study.best_trial.number}")
    logger.info(f"Best Value (Val MSE): {study.best_trial.value:.8f}")
    logger.info(f"Best Parameters: {study.best_trial.params}")
    
    # Save best hyperparameters
    best_params_dict = {
        "best_trial_number": study.best_trial.number,
        "best_value": study.best_trial.value,
        "best_params": study.best_trial.params,
        "n_trials_total": len(study.trials),
        "n_trials_completed": len(completed_trials),
        "datetime": datetime.now().isoformat()
    }
    
    with open(os.path.join(OUTPUT_DIRS["results"], "best_hyperparameters.json"), 'w') as f:
        json.dump(best_params_dict, f, indent=4)
    
    # Save full trials dataframe
    trials_df = study.trials_dataframe()
    trials_df.to_csv(os.path.join(OUTPUT_DIRS["optuna"], "all_trials_optuna.csv"), index=False)
    logger.info(f"Optuna trials saved to optuna_trials/all_trials_optuna.csv")

    # ==========================================
    # RETRAIN BEST MODEL
    # ==========================================
    logger.info("\n" + "=" * 60)
    logger.info("Retraining Best Model for Final Evaluation...")
    logger.info("=" * 60)
    
    best_params = study.best_trial.params
    
    # Load data with best scaler
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler = load_and_split_data(
        scaler_type=best_params["scaler_type"]
    )
    
    # Save the scaler used for best model
    final_scaler_path = os.path.join(OUTPUT_DIRS["models"], "best_model_scaler.pkl")
    joblib.dump(scaler, final_scaler_path)
    logger.info(f"Best model scaler saved to: {final_scaler_path}")
    
    # Create model
    final_model = InverseVAE(
        input_dim=len(INPUT_FEATURES),
        output_dim=len(OUTPUT_FEATURES),
        latent_dim=best_params["latent_dim"],
        hidden_size=best_params["hidden_size"],
        n_layers=best_params["n_layers"],
        dropout_rate=best_params["dropout"],
        activation=best_params.get("activation", "leaky_relu")
    ).to(MAIN_DEVICE)
    
    num_params = sum(p.numel() for p in final_model.parameters() if p.requires_grad)
    logger.info(f"Final model has {num_params:,} trainable parameters")
    
    # Training setup
    optimizer = optim.AdamW(
        final_model.parameters(), 
        lr=best_params["lr"], 
        weight_decay=best_params["weight_decay"]
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7
    )
    grad_scaler = GradScaler(enabled=(MAIN_DEVICE.type == "cuda"))
    
    # Data loaders
    train_loader, val_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, best_params["batch_size"]
    )
    
    # Training loop
    history = []
    best_val_mse = float('inf')
    best_model_state = None
    epochs = 300
    early_stop_patience = 30
    early_stop_counter = 0

    # Resume capability
    resume_checkpoint = os.path.join(OUTPUT_DIRS["models"], "best_model_final.pth")
    start_epoch = 0
    if os.path.exists(resume_checkpoint):
        try:
            ckpt = torch.load(resume_checkpoint, map_location=MAIN_DEVICE)
            if 'model_state_dict' in ckpt:
                final_model.load_state_dict(ckpt['model_state_dict'])
                best_model_state = ckpt['model_state_dict']
            if 'optimizer_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            best_val_mse = ckpt.get('best_val_mse', best_val_mse)
            start_epoch = ckpt.get('epoch', -1) + 1
            early_stop_counter = ckpt.get('early_stop_counter', 0)
            logger.info(f"Resuming training from epoch {start_epoch} with best_val_mse={best_val_mse:.6f}")
        except Exception as e:
            logger.warning(f"Resume failed, starting fresh: {e}")
    
    logger.info(f"Training for up to {epochs} epochs with early stopping patience {early_stop_patience}")
    
    for epoch in range(start_epoch, epochs):
        try:
            t_loss, t_mse, t_kld = train_one_epoch(
                final_model, train_loader, optimizer, grad_scaler, 
                best_params["beta"], best_params["grad_clip"], MAIN_DEVICE
            )
            v_loss, v_mse = validate(final_model, val_loader, best_params["beta"], MAIN_DEVICE)
            
            if not np.isfinite(v_loss) or not np.isfinite(t_loss):
                logger.warning(f"NaN/Inf detected at epoch {epoch}. Stopping training.")
                break
            
            scheduler.step(v_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            history.append({
                "epoch": epoch,
                "train_loss": t_loss, 
                "train_mse": t_mse, 
                "train_kld": t_kld,
                "val_loss": v_loss, 
                "val_mse": v_mse,
                "lr": current_lr
            })
            
            # Log every 10 epochs
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} | T_Loss: {t_loss:.6f} | "
                           f"V_MSE: {v_mse:.6f} | Best: {best_val_mse:.6f} | LR: {current_lr:.2e}")

            if v_mse < best_val_mse:
                best_val_mse = v_mse
                best_model_state = {k: v.cpu().clone() for k, v in final_model.state_dict().items()}
                early_stop_counter = 0
                
                # Save checkpoint
                checkpoint_path = os.path.join(OUTPUT_DIRS["models"], "best_model_final.pth")
                torch.save({
                    'model_state_dict': best_model_state,
                    'params': best_params,
                    'best_val_mse': best_val_mse,
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'early_stop_counter': early_stop_counter
                }, checkpoint_path)
            else:
                early_stop_counter += 1
            
            if early_stop_counter >= early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"CUDA OOM at epoch {epoch}. Stopping training.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                break
            else:
                raise e

    # Save training history
    history_path = os.path.join(OUTPUT_DIRS["results"], "training_history_final.csv")
    pd.DataFrame(history).to_csv(history_path, index=False)
    logger.info(f"Training history saved to: {history_path}")

    # ==========================================
    # FINAL EVALUATION
    # ==========================================
    logger.info("\n" + "=" * 60)
    logger.info("Running Final Evaluation on Test Set...")
    logger.info("=" * 60)
    
    eval_report_path = os.path.join(OUTPUT_DIRS["results"], "best_model_evaluation.txt")
    test_mse, test_r2 = evaluate_best_model(best_params, eval_report_path, MAIN_DEVICE)

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Best Validation MSE: {best_val_mse:.8f}")
    logger.info(f"Test MSE: {test_mse:.8f}")
    logger.info(f"Test R2: {test_r2:.8f}")
    logger.info(f"\nAll artifacts saved to: {BASE_DIR}")