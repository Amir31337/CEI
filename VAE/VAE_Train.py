#!/usr/bin/env python3
"""
VAE Training Script with Best Hyperparameters
Trains VAE model with pre-optimized hyperparameters and evaluates performance.

This script trains a Variational Autoencoder (VAE) regressor using pre-optimized
hyperparameters. It is designed for inverse problems in physics, such as
predicting initial atomic positions from velocity features.

Usage:
    python VAE_Train.py --data_path path/to/data.parquet --output_dir ./results --best_params_path best_hyperparameters.json

Requirements:
    - PyTorch
    - Scikit-learn
    - Pandas
    - NumPy
    - Joblib

The script loads best hyperparameters from a JSON file and trains the final model.
"""

import os
import sys
import json
import logging
import gc
import time
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION & GLOBAL CONSTANTS
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE with best hyperparameters")
    parser.add_argument("--data_path", type=str, default="data/data_oriented.parquet",
                        help="Path to the input parquet data file")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Output directory for results and models")
    parser.add_argument("--best_params_path", type=str, default="best_hyperparameters.json",
                        help="Path to the JSON file with best hyperparameters")
    parser.add_argument("--n_seeds", type=int, default=10,
                        help="Number of random seeds to run")
    parser.add_argument("--start_seed", type=int, default=42,
                        help="Starting seed number")
    return parser.parse_args()

args = parse_args()

# Hardware configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Performance optimizations for high-end hardware
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

# File Paths
BASE_DIR = args.output_dir
DATA_PATH = args.data_path
BEST_PARAMS_PATH = args.best_params_path

OUTPUT_DIRS = {
    "models": os.path.join(BASE_DIR, "models"),
    "results": os.path.join(BASE_DIR, "results")
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
# SCALER FACTORY
# ==========================================

def get_scaler(scaler_type: str):
    """Factory function to get scaler based on type."""
    scalers = {
        "minmax": MinMaxScaler(),
        "standard": StandardScaler(),
        "robust": RobustScaler()
    }
    return scalers.get(scaler_type, StandardScaler())


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
    """Custom Dataset for the Inverse Physics Problem."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_and_split_data(scaler_type: str = "standard"):
    """
    Loads data from parquet, splits into train/val/test (70/15/15),
    fits scaler on training data only, and transforms all sets.
    Only normalizes input features, outputs remain in physical units.
    """
    logger.info(f"Loading dataset from {DATA_PATH}...")
    try:
        df = pd.read_parquet(DATA_PATH)
        logger.info(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    X = df[INPUT_FEATURES].values
    y = df[OUTPUT_FEATURES].values
    logger.info(f"Input features: {len(INPUT_FEATURES)}, Output features: {len(OUTPUT_FEATURES)}")
    
    # Free the dataframe memory
    del df
    gc.collect()
    
    logger.info("Splitting data into train/val/test...")
    # Split: Train (70%), Temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(VAL_RATIO + TEST_RATIO), random_state=RANDOM_SEED, shuffle=True
    )
    
    # Split Temp: Val (15%), Test (15%) - which is 50/50 of the 30%
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, shuffle=True
    )
    
    logger.info(f"Data split - Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
    
    logger.info(f"Fitting {scaler_type} scaler on training data...")
    scaler_X = get_scaler(scaler_type)
    
    X_train_s = scaler_X.fit_transform(X_train)
    X_val_s = scaler_X.transform(X_val)
    X_test_s = scaler_X.transform(X_test)
    
    # Save Scaler for Inference
    scaler_path = os.path.join(OUTPUT_DIRS["models"], f"scaler_X_{scaler_type}.pkl")
    joblib.dump(scaler_X, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")
    
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
        """The Reparameterization Trick: z = mu + sigma * epsilon"""
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
# EVALUATION FUNCTION
# ==========================================

def evaluate_model(model, X_test, y_test, scaler, params, device):
    """Evaluate the model on test set and save comprehensive results."""
    logger.info("=" * 60)
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info("=" * 60)
    
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
    
    for key, value in params.items():
        report_lines.append(f"  {key}: {value}")
    
    report_lines.extend([
        "",
        "=" * 80,
        "DATA SPLIT INFORMATION",
        "=" * 80,
        f"  Test samples: {len(X_test):,}",
        f"  Input features: {len(INPUT_FEATURES)}",
        f"  Output features: {len(OUTPUT_FEATURES)}",
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
    ])
    
    # Write report
    report_path = os.path.join(OUTPUT_DIRS["results"], "model_evaluation.txt")
    report_text = "\n".join(report_lines)
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    logger.info(f"Evaluation report saved to: {report_path}")
    logger.info(f"\nTest Results: MSE={overall_mse:.8f}, RMSE={overall_rmse:.8f}, MAE={overall_mae:.8f}, R2={overall_r2:.8f}")
    
    # Also save as JSON for programmatic access
    metrics_json = {
        "hyperparameters": params,
        "overall_metrics": {
            "mse": float(overall_mse),
            "rmse": float(overall_rmse),
            "mae": float(overall_mae),
            "r2": float(overall_r2)
        },
        "feature_metrics": {k: {mk: float(mv) for mk, mv in v.items()} for k, v in feature_metrics.items()},
        "data_info": {
            "test_samples": len(X_test)
        }
    }
    
    json_path = os.path.join(OUTPUT_DIRS["results"], "model_evaluation.json")
    with open(json_path, 'w') as f:
        json.dump(metrics_json, f, indent=4)
    
    # Save predictions
    df_pred = pd.DataFrame(np_preds, columns=[f"P_{c}" for c in OUTPUT_FEATURES])
    df_true = pd.DataFrame(np_targets, columns=[f"T_{c}" for c in OUTPUT_FEATURES])
    df_final = pd.concat([df_true, df_pred], axis=1)
    pred_path = os.path.join(OUTPUT_DIRS["results"], "test_predictions.csv")
    df_final.to_csv(pred_path, index=False)
    logger.info(f"Predictions saved to: {pred_path}")
    
    return overall_mse, overall_r2, np_preds


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info(f"Starting VAE Training with Best Hyperparameters")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    if DEVICE.type == "cuda":
        torch.cuda.set_device(DEVICE.index)
        logger.info(f"GPU: {torch.cuda.get_device_name(DEVICE)}")
    
    # Load best hyperparameters
    logger.info(f"Loading best hyperparameters from: {BEST_PARAMS_PATH}")
    with open(BEST_PARAMS_PATH, 'r') as f:
        best_params_data = json.load(f)
    
    params = best_params_data["best_params"]
    logger.info(f"Best parameters: {params}")
    
    # Load and split data (same for all seeds)
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler = load_and_split_data(
        scaler_type=params["scaler_type"]
    )
    
    # Initialize lists to collect results across seeds
    all_test_mses = []
    all_test_r2s = []
    all_test_predictions = []
    all_best_val_mses = []
    
    n_seeds = args.n_seeds
    start_seed = args.start_seed
    
    logger.info(f"Running {n_seeds} seeds starting from {start_seed}")
    logger.info("=" * 60)
    
    for seed_idx in range(n_seeds):
        current_seed = start_seed + seed_idx
        logger.info(f"\n--- Seed {seed_idx + 1}/{n_seeds} (Seed: {current_seed}) ---")
        
        # Set seed for reproducibility
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)
        
        # Create model
        model = InverseVAE(
            input_dim=len(INPUT_FEATURES),
            output_dim=len(OUTPUT_FEATURES),
            latent_dim=params["latent_dim"],
            hidden_size=params["hidden_size"],
            n_layers=params["n_layers"],
            dropout_rate=params["dropout"],
            activation=params.get("activation", "leaky_relu")
        ).to(DEVICE)
        
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model has {num_params:,} trainable parameters")
        
        # Training setup
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=params["lr"], 
            weight_decay=params["weight_decay"]
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7
        )
        grad_scaler = GradScaler(enabled=(DEVICE.type == "cuda"))
        
        # Data loaders
        train_dataset = PhysicsDataset(X_train, y_train)
        val_dataset = PhysicsDataset(X_val, y_val)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=params["batch_size"], 
            shuffle=True, 
            num_workers=4, 
            pin_memory=True, 
            prefetch_factor=2,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=params["batch_size"], 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True, 
            prefetch_factor=2
        )
        
        # Training loop
        history = []
        best_val_mse = float('inf')
        best_model_state = None
        epochs = 300
        early_stop_patience = 30
        early_stop_counter = 0
        
        logger.info(f"Training for up to {epochs} epochs with early stopping patience {early_stop_patience}")
        
        for epoch in range(epochs):
            try:
                t_loss, t_mse, t_kld = train_one_epoch(
                    model, train_loader, optimizer, grad_scaler, 
                    params["beta"], params["grad_clip"], DEVICE
                )
                v_loss, v_mse = validate(model, val_loader, params["beta"], DEVICE)
                
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
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    early_stop_counter = 0
                    
                    # Save checkpoint
                    checkpoint_path = os.path.join(OUTPUT_DIRS["models"], f"best_model_seed_{current_seed}.pth")
                    torch.save({
                        'model_state_dict': best_model_state,
                        'params': params,
                        'best_val_mse': best_val_mse,
                        'epoch': epoch,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()
                    }, checkpoint_path)
                    logger.info(f"✓ New best model saved (Val MSE: {best_val_mse:.6f})")
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

        # Save training history for this seed
        history_path = os.path.join(OUTPUT_DIRS["results"], f"training_history_seed_{current_seed}.csv")
        pd.DataFrame(history).to_csv(history_path, index=False)
        logger.info(f"Training history saved to: {history_path}")

        # Load best model for evaluation
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            logger.info(f"Loaded best model with validation MSE: {best_val_mse:.6f}")
        
        # Final evaluation on test set
        logger.info("Running Final Evaluation on Test Set...")
        test_mse, test_r2, test_predictions = evaluate_model(model, X_test, y_test, scaler, params, DEVICE)
        
        # Collect results
        all_test_mses.append(test_mse)
        all_test_r2s.append(test_r2)
        all_test_predictions.append(test_predictions)
        all_best_val_mses.append(best_val_mse)
        
        logger.info(f"Seed {current_seed} - Val MSE: {best_val_mse:.8f}, Test MSE: {test_mse:.8f}, Test R2: {test_r2:.8f}")
    
    # Compute averages across seeds
    avg_test_mse = np.mean(all_test_mses)
    std_test_mse = np.std(all_test_mses)
    avg_test_r2 = np.mean(all_test_r2s)
    std_test_r2 = np.std(all_test_r2s)
    avg_best_val_mse = np.mean(all_best_val_mses)
    std_best_val_mse = np.std(all_best_val_mses)
    
    logger.info("\n" + "=" * 60)
    logger.info("MULTI-SEED SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Seeds: {n_seeds} (from {start_seed} to {start_seed + n_seeds - 1})")
    logger.info(f"Average Best Val MSE: {avg_best_val_mse:.8f} ± {std_best_val_mse:.8f}")
    logger.info(f"Average Test MSE: {avg_test_mse:.8f} ± {std_test_mse:.8f}")
    logger.info(f"Average Test R2: {avg_test_r2:.8f} ± {std_test_r2:.8f}")
    
    # Save averaged test predictions
    avg_predictions = np.mean(all_test_predictions, axis=0)
    df_pred = pd.DataFrame(avg_predictions, columns=[f"P_{c}" for c in OUTPUT_FEATURES])
    df_true = pd.DataFrame(y_test, columns=[f"T_{c}" for c in OUTPUT_FEATURES])
    df_final = pd.concat([df_true, df_pred], axis=1)
    pred_path = os.path.join(OUTPUT_DIRS["results"], "vae_test_full_data.csv")
    df_final.to_csv(pred_path, index=False)
    logger.info(f"Averaged test predictions saved to: {pred_path}")
    
    # Save summary
    summary = {
        "n_seeds": n_seeds,
        "start_seed": start_seed,
        "avg_best_val_mse": float(avg_best_val_mse),
        "std_best_val_mse": float(std_best_val_mse),
        "avg_test_mse": float(avg_test_mse),
        "std_test_mse": float(std_test_mse),
        "avg_test_r2": float(avg_test_r2),
        "std_test_r2": float(std_test_r2),
        "hyperparameters": params
    }
    
    summary_path = os.path.join(OUTPUT_DIRS["results"], "multi_seed_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Multi-seed summary saved to: {summary_path}")

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 60)
