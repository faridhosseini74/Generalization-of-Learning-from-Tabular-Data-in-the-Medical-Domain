"""
Comprehensive QISN training pipeline supporting:
- Individual dataset training
- Combined dataset training 
- Fine-tuning pre-trained models

Usage:
    python train.py --mode individual --dataset ohio --epochs 100
    python train.py --mode combined --datasets ohio,hupa --epochs 100
    python train.py --mode finetune --source_dataset ohio --target_dataset hupa --epochs 50
"""

import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split, DistributedSampler, ConcatDataset
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import math
import uuid
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pickle
from pathlib import Path

from data_utils import prepare_glucose_data, prepare_data_for_model, create_dataset_for_model, get_model_data_requirements, GlucoseDataset, get_available_datasets, get_dataset_info
from model import QISNForGlucose, ModelFactory

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_ddp(rank, world_size, port=None):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    if port is None:
        port = os.environ.get('MASTER_PORT', '12355')
    os.environ['MASTER_PORT'] = port
    os.environ['NCCL_TIMEOUT'] = '1800000'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def create_model_from_args(args, cat_dims, num_continuous):
    """Create model based on command line arguments."""
    
    # Prepare model configuration
    model_config = {
        'dim': args.dim,
        'depth': args.depth,
        'heads': args.heads,
        'attn_dropout': args.attn_dropout,
        'ff_dropout': args.ff_dropout
    }
    
    # Add model-specific parameters
    if args.model == 'qisn':
        model_config['n_states'] = args.n_states
    elif args.model == 'patchtst':
        model_config['patch_len'] = args.patch_len
        model_config['stride'] = args.stride
    elif args.model == 'informer':
        model_config['factor'] = args.factor
    
    # Create model using the factory
    model = ModelFactory.create_model(
        model_name=args.model,
        categories=cat_dims,
        num_continuous=num_continuous,
        config=model_config
    )
    
    logging.info(f"Created {args.model.upper()} model with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


def cleanup_ddp():
    """Clean up distributed training."""
    dist.destroy_process_group()

def configure_logging(rank, log_file):
    """Configure logging for distributed training."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler for all ranks
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler only for rank 0
    if rank == 0:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

def evaluate_model(model, dataloader, criterion, device, scaler=None):
    """Evaluate model performance."""
    model.eval()
    all_targets = []
    all_predictions = []
    total_loss = 0.0
    
    with torch.no_grad():
        for (x_cat, x_cont, c_mask, co_mask), target in dataloader:
            x_cat = x_cat.to(device, non_blocking=True)
            x_cont = x_cont.to(device, non_blocking=True)
            c_mask = c_mask.to(device, non_blocking=True)
            co_mask = co_mask.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            output = model(x_cat, x_cont, c_mask, co_mask)
            loss = criterion(output, target)
            
            total_loss += loss.item() * target.size(0)
            all_predictions.extend(output.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader.dataset) if len(dataloader.dataset) > 0 else 0.0
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Scale predictions and targets to [0,1] for RMSE calculation
    rmse_scaler = MinMaxScaler(feature_range=(0, 1))
    all_targets_scaled = rmse_scaler.fit_transform(all_targets.reshape(-1, 1)).flatten()
    all_predictions_scaled = rmse_scaler.transform(all_predictions.reshape(-1, 1)).flatten()
    
    # Calculate metrics on [0,1] scaled data
    scaled_mse = mean_squared_error(all_targets_scaled, all_predictions_scaled)
    scaled_rmse = math.sqrt(scaled_mse)
    scaled_r2 = r2_score(all_targets_scaled, all_predictions_scaled)
    
    # Calculate metrics on normalized data (original behavior)
    norm_mse = mean_squared_error(all_targets, all_predictions)
    norm_rmse = math.sqrt(norm_mse)
    norm_r2 = r2_score(all_targets, all_predictions)
    
    # Calculate metrics on original scale if scaler provided
    if scaler:
        all_predictions_orig = scaler.inverse_transform(all_predictions.reshape(-1, 1)).flatten()
        all_targets_orig = scaler.inverse_transform(all_targets.reshape(-1, 1)).flatten()
        
        orig_mse = mean_squared_error(all_targets_orig, all_predictions_orig)
        orig_rmse = math.sqrt(orig_mse)
        orig_r2 = r2_score(all_targets_orig, all_predictions_orig)
        
        return {
            'loss': avg_loss,
            'mse': orig_mse,
            'rmse': scaled_rmse,  # Use scaled RMSE as primary metric
            'r2': orig_r2,
            'norm_mse': norm_mse,
            'norm_rmse': norm_rmse,
            'norm_r2': norm_r2,
            'scaled_rmse': scaled_rmse,
            'scaled_r2': scaled_r2
        }
    else:
        return {
            'loss': avg_loss,
            'mse': norm_mse,
            'rmse': scaled_rmse,  # Use scaled RMSE as primary metric
            'r2': norm_r2,
            'scaled_rmse': scaled_rmse,
            'scaled_r2': scaled_r2
        }

def plot_training_metrics(history, model_name, save_path=None):
    """Plot training metrics."""
    plt.figure(figsize=(20, 15))
    gs = GridSpec(2, 2, figure=plt.gcf())
    
    # Plot Loss
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(history['train_loss'], label='Training Loss', color='blue', marker='o', markersize=2)
    ax1.plot(history['val_loss'], label='Validation Loss', color='red', marker='o', markersize=2)
    ax1.set_title(f'{model_name} - Loss over epochs', fontsize=16)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot RMSE
    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(history['train_rmse'], label='Training RMSE', color='blue', marker='o', markersize=2)
    ax2.plot(history['val_rmse'], label='Validation RMSE', color='green', marker='o', markersize=2)
    ax2.set_title(f'{model_name} - RMSE over epochs', fontsize=16)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('RMSE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot R 
    ax3 = plt.subplot(gs[1, 0])
    ax3.plot(history['train_r2'], label='Training R ', color='blue', marker='o', markersize=2)
    ax3.plot(history['val_r2'], label='Validation R ', color='orange', marker='o', markersize=2)
    ax3.set_title(f'{model_name} - R  over epochs', fontsize=16)
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('R ')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot Learning Rate
    ax4 = plt.subplot(gs[1, 1])
    if 'lr' in history:
        ax4.plot(history['lr'], label='Learning Rate', color='red', marker='o', markersize=2)
        ax4.set_title(f'{model_name} - Learning Rate over epochs', fontsize=16)
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Learning Rate')
        ax4.set_yscale('log')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Training plot saved to {save_path}")
    
    plt.close()

def save_checkpoint(rank, epoch, model, optimizer, scheduler, history, best_val_loss, counter, best_epoch, checkpoint_path):
    """Save training checkpoint."""
    if rank == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history,
            'best_val_loss': best_val_loss,
            'early_stopping_counter': counter,
            'best_epoch': best_epoch
        }
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Checkpoint saved at {checkpoint_path}")

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """Load training checkpoint."""
    if not os.path.exists(checkpoint_path):
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return {
            'start_epoch': checkpoint['epoch'] + 1,
            'history': checkpoint['history'],
            'best_val_loss': checkpoint['best_val_loss'],
            'early_stopping_counter': checkpoint['early_stopping_counter'],
            'best_epoch': checkpoint['best_epoch']
        }
    except Exception as e:
        logging.warning(f"Failed to load checkpoint: {e}")
        return None

def transfer_weights(model, pretrained_path, device):
    """Transfer weights from pretrained model."""
    if not os.path.exists(pretrained_path):
        logging.warning(f"Pretrained model not found: {pretrained_path}")
        return []
    
    logging.info(f"Loading pretrained weights from: {pretrained_path}")
    pretrained_state_dict = torch.load(pretrained_path, map_location=device)
    
    model_to_transfer = model.module if hasattr(model, 'module') else model
    model_state_dict = model_to_transfer.state_dict()
    transferred_layers = []
    
    # Transfer compatible layers
    for name, param in model_state_dict.items():
        if name in pretrained_state_dict:
            if param.size() == pretrained_state_dict[name].size():
                model_state_dict[name] = pretrained_state_dict[name]
                transferred_layers.append(name)
                logging.info(f"Transferred layer: {name}")
            else:
                logging.info(f"Size mismatch for {name}: model={param.size()}, pretrained={pretrained_state_dict[name].size()}")
    
    model_to_transfer.load_state_dict(model_state_dict)
    logging.info(f"Successfully transferred {len(transferred_layers)} layers")
    
    return transferred_layers

def train_traditional_model(
    datasets_info, model_name, output_dir, args,
    valid_split=0.2
):
    """Training function specifically for traditional models."""
    from model import TraditionalModelWrapper
    
    logging.info(f"Training traditional model: {model_name}")
    
    # Combine all datasets if multiple
    all_cat_features = []
    all_cont_features = []
    all_targets = []
    all_statistics = []
    
    for X_cat, X_cont, y, cat_mask, con_mask, cat_dims, scaler, dataset_name in datasets_info:
        all_cat_features.append(X_cat)
        all_cont_features.append(X_cont)
        all_targets.append(y)
        
        # Create basic statistics
        stats = torch.cat([
            torch.mean(X_cont, dim=1, keepdim=True),
            torch.std(X_cont, dim=1, keepdim=True) + 1e-8,
            torch.min(X_cont, dim=1, keepdim=True)[0],
            torch.max(X_cont, dim=1, keepdim=True)[0]
        ], dim=2)
        all_statistics.append(stats)
    
    # Concatenate all data
    X_cat_combined = torch.cat(all_cat_features, dim=0)
    X_cont_combined = torch.cat(all_cont_features, dim=0)
    y_combined = torch.cat(all_targets, dim=0)
    statistics_combined = torch.cat(all_statistics, dim=0)
    
    # Get dimensions for model creation
    cat_dims = datasets_info[0][5]  # Category dimensions from first dataset
    num_continuous = X_cont_combined.shape[-1]
    
    # Create model
    model = TraditionalModelWrapper(
        model_name=model_name,
        categories=cat_dims,
        num_continuous=num_continuous
    )
    
    # Split data
    dataset_size = len(y_combined)
    valid_size = int(dataset_size * valid_split)
    train_size = dataset_size - valid_size
    
    indices = torch.randperm(dataset_size)
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:]
    
    # Create datasets
    from data_utils import GlucoseDataset
    from torch.utils.data import DataLoader, Subset
    
    full_dataset = GlucoseDataset(
        X_cat_combined, X_cont_combined, y_combined, 
        statistics=statistics_combined
    )
    
    train_dataset = Subset(full_dataset, train_indices)
    valid_dataset = Subset(full_dataset, valid_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=1024, shuffle=False)
    
    # Train the model
    logging.info("Fitting traditional model...")
    history = model.fit(train_loader, valid_loader)
    
    # Evaluate on validation set
    model.eval()
    val_predictions = []
    val_targets = []
    
    with torch.no_grad():
        for batch in valid_loader:
            if len(batch) == 4:
                cat_features, cont_features, targets, statistics = batch
            else:
                cat_features, cont_features, targets = batch
                statistics = None
            
            predictions = model(cat_features, cont_features, statistics)
            val_predictions.append(predictions.cpu().numpy())
            val_targets.append(targets.cpu().numpy())
    
    val_predictions = np.concatenate(val_predictions)
    val_targets = np.concatenate(val_targets)
    
    # Calculate metrics
    val_mse = mean_squared_error(val_targets, val_predictions)
    val_rmse = np.sqrt(val_mse)
    val_r2 = r2_score(val_targets, val_predictions)
    
    logging.info(f"Validation RMSE: {val_rmse:.4f}")
    logging.info(f"Validation R²: {val_r2:.4f}")
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f'{model_name}_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    logging.info(f"Model saved to {model_path}")
    
    return model, {'val_rmse': val_rmse, 'val_r2': val_r2}


def train_model(
    rank, world_size, datasets_info, model_name, output_dir, args,
    batch_size=64, epochs=100, lr=1e-4, weight_decay=0.001,
    valid_split=0.2, patience=15, min_delta=0.00005,
    scheduler_patience=8, scheduler_factor=0.5, scheduler_min_lr=1e-6,
    checkpoint_freq=5, resume=True, pretrained_path=None
):
    """Main training function."""
    
    if len(datasets_info) > 1:
        # Combined training mode - need to handle multiple datasets
        setup_ddp(rank, world_size)
        device = torch.device(f'cuda:{rank}')
        run_id = str(uuid.uuid4())[:8]
        configure_logging(rank, f'{output_dir}/training_{run_id}.log')
        
        if rank == 0:
            logging.info(f"Training {model_name} on combined datasets with {world_size} GPUs")
        
        # Find max dimensions across all datasets
        max_cat_features = max(X_cat.shape[2] for X_cat, _, _, _, _, _, _, _ in datasets_info)
        max_cont_features = max(X_cont.shape[2] for _, X_cont, _, _, _, _, _, _ in datasets_info)
        max_cat_dims = []
        
        for i in range(max_cat_features):
            max_dim = max(cat_dims[i] for _, _, _, _, _, cat_dims, _, _ in datasets_info 
                         if i < len(cat_dims))
            max_cat_dims.append(max_dim)
        
        # Create combined datasets
        train_datasets = []
        valid_datasets = []
        
        for X_cat, X_cont, y, cat_mask, con_mask, cat_dims, scaler, dataset_name in datasets_info:
            # Move to CPU for dataset creation
            X_cat, X_cont, y, cat_mask, con_mask = [x.cpu() for x in [X_cat, X_cont, y, cat_mask, con_mask]]
            
            dataset_size = len(y)
            valid_size = int(dataset_size * valid_split)
            train_size = dataset_size - valid_size
            
            indices = torch.randperm(dataset_size)
            train_indices = indices[:train_size]
            valid_indices = indices[train_size:]
            
            # Split data
            train_X_cat = X_cat[train_indices]
            train_X_cont = X_cont[train_indices]
            train_y = y[train_indices]
            train_cat_mask = cat_mask[train_indices]
            train_con_mask = con_mask[train_indices]
            
            valid_X_cat = X_cat[valid_indices]
            valid_X_cont = X_cont[valid_indices]
            valid_y = y[valid_indices]
            valid_cat_mask = cat_mask[valid_indices]
            valid_con_mask = con_mask[valid_indices]
            
            # Pad to max dimensions
            if train_X_cat.shape[2] < max_cat_features:
                pad_size = max_cat_features - train_X_cat.shape[2]
                pad_cat_train = torch.zeros((train_X_cat.shape[0], train_X_cat.shape[1], pad_size), dtype=torch.long)
                pad_cat_valid = torch.zeros((valid_X_cat.shape[0], valid_X_cat.shape[1], pad_size), dtype=torch.long)
                train_X_cat = torch.cat([train_X_cat, pad_cat_train], dim=2)
                valid_X_cat = torch.cat([valid_X_cat, pad_cat_valid], dim=2)
                
                pad_cat_mask_train = torch.zeros_like(pad_cat_train, dtype=torch.bool)
                pad_cat_mask_valid = torch.zeros_like(pad_cat_valid, dtype=torch.bool)
                train_cat_mask = torch.cat([train_cat_mask, pad_cat_mask_train], dim=2)
                valid_cat_mask = torch.cat([valid_cat_mask, pad_cat_mask_valid], dim=2)
            
            if train_X_cont.shape[2] < max_cont_features:
                pad_size = max_cont_features - train_X_cont.shape[2]
                pad_cont_train = torch.zeros((train_X_cont.shape[0], train_X_cont.shape[1], pad_size), dtype=torch.float32)
                pad_cont_valid = torch.zeros((valid_X_cont.shape[0], valid_X_cont.shape[1], pad_size), dtype=torch.float32)
                train_X_cont = torch.cat([train_X_cont, pad_cont_train], dim=2)
                valid_X_cont = torch.cat([valid_X_cont, pad_cont_valid], dim=2)
                
                pad_cont_mask_train = torch.zeros_like(pad_cont_train, dtype=torch.bool)
                pad_cont_mask_valid = torch.zeros_like(pad_cont_valid, dtype=torch.bool)
                train_con_mask = torch.cat([train_con_mask, pad_cont_mask_train], dim=2)
                valid_con_mask = torch.cat([valid_con_mask, pad_cont_mask_valid], dim=2)
            
            # Create datasets
            train_dataset = GlucoseDataset(
                features=(train_X_cat, train_X_cont, train_cat_mask, train_con_mask),
                targets=train_y
            )
            valid_dataset = GlucoseDataset(
                features=(valid_X_cat, valid_X_cont, valid_cat_mask, valid_con_mask),
                targets=valid_y
            )
            
            train_datasets.append(train_dataset)
            valid_datasets.append(valid_dataset)
            
            if rank == 0:
                logging.info(f"{dataset_name} - Train: {len(train_dataset)}, Valid: {len(valid_dataset)}")
        
        # Combine datasets
        combined_train_dataset = ConcatDataset(train_datasets)
        combined_valid_dataset = ConcatDataset(valid_datasets)
        
        # Use first dataset's scaler for evaluation
        glucose_scaler = datasets_info[0][6]
        
                # Create model with max dimensions
        model = create_model_from_args(args, max_cat_dims, max_cont_features).to(device)
        
    else:
        # Single dataset training
        setup_ddp(rank, world_size)
        device = torch.device(f'cuda:{rank}')
        run_id = str(uuid.uuid4())[:8]
        configure_logging(rank, f'{output_dir}/training_{run_id}.log')
        
        if rank == 0:
            logging.info(f"Training {model_name} on single dataset with {world_size} GPUs")
        
        X_cat, X_cont, y, cat_mask, con_mask, cat_dims, glucose_scaler, dataset_name = datasets_info[0]
        
        # Create dataset
        dataset = GlucoseDataset(
            features=(X_cat, X_cont, cat_mask, con_mask),
            targets=y
        )
        
        # Split dataset
        dataset_size = len(dataset)
        valid_size = int(dataset_size * valid_split)
        train_size = dataset_size - valid_size
        
        torch.manual_seed(42)
        train_dataset, valid_dataset = random_split(
            dataset, [train_size, valid_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        combined_train_dataset = train_dataset
        combined_valid_dataset = valid_dataset
        
        # Create model
        model = create_model_from_args(args, cat_dims, X_cont.shape[-1]).to(device)
    
    # Apply DDP
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # Transfer pretrained weights if provided
    if pretrained_path:
        transfer_weights(model, pretrained_path, device)
    
    # Create dataloaders
    train_sampler = DistributedSampler(combined_train_dataset)
    train_loader = DataLoader(
        combined_train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        combined_valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    if rank == 0:
        logging.info(f"Training samples: {len(combined_train_dataset)}")
        logging.info(f"Validation samples: {len(combined_valid_dataset)}")
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience,
        min_lr=scheduler_min_lr
    )
    
    # Training setup
    best_val_loss = float('inf')
    history = {
        'train_loss': [], 'train_rmse': [], 'train_r2': [],
        'val_loss': [], 'val_rmse': [], 'val_r2': [], 'lr': []
    }
    counter = 0
    best_epoch = 0
    start_epoch = 0
    
    # Checkpoint paths
    checkpoint_path = f'{output_dir}/checkpoint.pt'
    model_path = f'{output_dir}/best_model.pt'
    
    # Try to load checkpoint
    if resume and rank == 0:
        checkpoint_data = load_checkpoint(checkpoint_path, model, optimizer, scheduler, device)
        if checkpoint_data:
            start_epoch = checkpoint_data['start_epoch']
            history = checkpoint_data['history']
            best_val_loss = checkpoint_data['best_val_loss']
            counter = checkpoint_data['early_stopping_counter']
            best_epoch = checkpoint_data['best_epoch']
            logging.info(f"Resuming training from epoch {start_epoch}")
    
    # Synchronize start epoch across all processes
    if world_size > 1:
        start_epoch_tensor = torch.tensor(start_epoch, device=device)
        dist.broadcast(start_epoch_tensor, src=0)
        start_epoch = start_epoch_tensor.item()
    
    # Training loop
    for epoch in range(start_epoch, epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        epoch_loss = 0.0
        num_samples = 0
        
        # Training
        pbar = tqdm(train_loader, unit="batch", desc=f"Rank {rank} Epoch {epoch+1}", disable=rank != 0)
        for (x_cat, x_cont, c_mask, co_mask), target in pbar:
            x_cat = x_cat.to(device, non_blocking=True)
            x_cont = x_cont.to(device, non_blocking=True)
            c_mask = c_mask.to(device, non_blocking=True)
            co_mask = co_mask.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            output = model(x_cat, x_cont, c_mask, co_mask)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            current_batch_size = target.size(0)
            epoch_loss += loss.item() * current_batch_size
            num_samples += current_batch_size
            
            pbar.set_postfix(loss=loss.item())
        
        avg_train_loss = epoch_loss / num_samples if num_samples > 0 else 0.0
        
        # Evaluation (only on rank 0)
        if rank == 0:
            train_metrics = evaluate_model(model, train_loader, criterion, device, glucose_scaler)
            val_metrics = evaluate_model(model, valid_loader, criterion, device, glucose_scaler)
            
            scheduler.step(val_metrics['loss'])
            
            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['train_rmse'].append(train_metrics['rmse'])
            history['train_r2'].append(train_metrics['r2'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_rmse'].append(val_metrics['rmse'])
            history['val_r2'].append(val_metrics['r2'])
            history['lr'].append(optimizer.param_groups[0]['lr'])
            
            logging.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, Train RMSE: {train_metrics['rmse']:.4f}, Train R : {train_metrics['r2']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, Val RMSE: {val_metrics['rmse']:.4f}, Val R : {val_metrics['r2']:.4f}, "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
            
            # Early stopping
            if val_metrics['loss'] < best_val_loss - min_delta:
                best_val_loss = val_metrics['loss']
                counter = 0
                best_epoch = epoch
                torch.save(model.module.state_dict(), model_path)
                logging.info(f"Model improved and saved")
            else:
                counter += 1
                logging.info(f"Early stopping counter: {counter}/{patience}")
                if counter >= patience:
                    logging.info(f"Early stopping triggered")
                    break
            
            # Save checkpoint
            if (epoch + 1) % checkpoint_freq == 0:
                save_checkpoint(
                    rank, epoch, model, optimizer, scheduler, history,
                    best_val_loss, counter, best_epoch, checkpoint_path
                )
    
    # Final evaluation and cleanup
    if rank == 0:
        if os.path.exists(model_path):
            model.module.load_state_dict(torch.load(model_path))
            logging.info(f"Loaded best model from epoch {best_epoch+1}")
        
        final_metrics = evaluate_model(model, valid_loader, criterion, device, glucose_scaler)
        logging.info(
            f"Final validation metrics - "
            f"Loss: {final_metrics['loss']:.4f}, "
            f"RMSE: {final_metrics['rmse']:.4f}, "
            f"R : {final_metrics['r2']:.4f}"
        )
        
        # Save training plot
        plot_path = f'{output_dir}/training_metrics.png'
        plot_training_metrics(history, model_name, plot_path)
        
        # Save history
        with open(f'{output_dir}/training_history.pkl', 'wb') as f:
            pickle.dump(history, f)
    
    cleanup_ddp()
    return model, history

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Universal Glucose Prediction Training Pipeline')
    
    # Model type selection
    parser.add_argument('--model_type', type=str, choices=['traditional', 'transformer', 'quantum'], 
                        default='quantum', help='Type of model to use')
    
    # Model selection
    from model import ModelFactory
    available_models = ModelFactory.get_available_models()
    parser.add_argument('--model', type=str, choices=available_models, default='qisn',
                        help=f'Model architecture ({", ".join(available_models)})')
    
    # Model hyperparameters
    parser.add_argument('--dim', type=int, default=128, help='Model hidden dimension')
    parser.add_argument('--depth', type=int, default=6, help='Number of model layers')
    parser.add_argument('--heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--attn_dropout', type=float, default=0.1, help='Attention dropout rate')
    parser.add_argument('--ff_dropout', type=float, default=0.1, help='Feed-forward dropout rate')
    
    # Model-specific parameters
    parser.add_argument('--n_states', type=int, default=4, help='Number of quantum states (QISN only)')
    parser.add_argument('--patch_len', type=int, default=3, help='Patch length (PatchTST only)')
    parser.add_argument('--stride', type=int, default=1, help='Patch stride (PatchTST only)')
    parser.add_argument('--factor', type=int, default=5, help='Sparse attention factor (Informer only)')
    
    # Training mode
    parser.add_argument('--mode', choices=['individual', 'combined', 'finetune'], required=True,
                        help='Training mode')
    
    # Dataset selection - now supports file paths
    parser.add_argument('--data_path', type=str, help='Path to dataset file (CSV or Parquet)')
    parser.add_argument('--dataset', type=str, help='Dataset name for backward compatibility')
    parser.add_argument('--datasets', type=str,
                        help='Comma-separated datasets for combined training')
    parser.add_argument('--source_dataset', type=str,
                        help='Source dataset for fine-tuning')
    parser.add_argument('--target_dataset', type=str,
                        help='Target dataset for fine-tuning')
    
    # Column specification (optional - will auto-detect if not provided)
    parser.add_argument('--patient_col', type=str, help='Patient ID column name')
    parser.add_argument('--time_col', type=str, help='Time column name')
    parser.add_argument('--glucose_col', type=str, help='Glucose column name')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--valid_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--window_size', type=int, default=12, help='Input sequence window size')
    parser.add_argument('--prediction_horizon', type=int, default=3, help='Prediction horizon')
    
    # SHAP analysis
    parser.add_argument('--shap', action='store_true', help='Run SHAP analysis after training')
    parser.add_argument('--shap_samples', type=int, default=1000, help='Number of samples for SHAP analysis')
    parser.add_argument('--shap_evals', type=int, default=2000, help='Maximum evaluations for SHAP')
    
    # System settings
    parser.add_argument('--world_size', type=int, default=None, help='Number of GPUs (auto-detect if None)')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'individual' and not (args.data_path or args.dataset):
        parser.error("Individual mode requires either --data_path or --dataset")
    
    # Determine world size
    if args.world_size is None:
        args.world_size = min(torch.cuda.device_count(), 3)
    
    print(f"Using {args.world_size} GPUs for training")
    
    # Prepare datasets based on mode
    if args.mode == 'individual':
        if args.data_path:
            # New format: use file path directly
            X_cat, X_cont, y, cat_mask, con_mask, cat_dims, scaler = prepare_data_for_model(
                file_path=args.data_path,
                model_name=args.model,
                window_size=args.window_size,
                prediction_horizon=args.prediction_horizon,
                patient_col=args.patient_col,
                time_col=args.time_col,
                glucose_col=args.glucose_col
            )
            dataset_name = Path(args.data_path).stem
        elif args.dataset:
            # Backward compatibility: use old dataset system
            dataset_info = get_dataset_info(args.dataset)
            X_cat, X_cont, y, cat_mask, con_mask, cat_dims, scaler = prepare_glucose_data(
                args.dataset, base_path=args.data_path
            )
            dataset_name = dataset_info['name']
        else:
            raise ValueError("Either --data_path or --dataset required for individual training")
        
        # Check if dataset is empty
        if len(y) == 0:
            raise ValueError(f"Dataset is empty. No samples found.")
            
        datasets_info = [(X_cat, X_cont, y, cat_mask, con_mask, cat_dims, scaler, dataset_name)]
        model_name = f"{args.model.upper()}-{dataset_name}"
        output_dir = f"{args.output_dir}/individual/{dataset_name}"
        
    elif args.mode == 'combined':
        if not args.datasets:
            raise ValueError("--datasets required for combined training")
        
        dataset_keys = [d.strip() for d in args.datasets.split(',')]
        datasets_info = []
        
        for key in dataset_keys:
            # FORCE file path detection for .parquet extensions to avoid any logic errors
            print(f"=== Processing dataset key: '{key}' ===")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Files in current directory: {[f for f in os.listdir('.') if f.endswith('.parquet')]}")
            
            # If the key has a known file extension, treat it as a file path regardless of other checks
            if key.endswith(('.csv', '.parquet', '.pkl', '.json')):
                print(f"DETECTED FILE EXTENSION: Treating '{key}' as file path")
                
                # Determine the actual file path
                actual_file_path = key
                if not os.path.exists(key):
                    # Try common locations
                    potential_paths = [
                        os.path.join('.', key),
                        os.path.join('traditional', key),
                        os.path.join('data', key)
                    ]
                    for path in potential_paths:
                        if os.path.exists(path):
                            actual_file_path = path
                            break
                
                print(f"Using file path: {actual_file_path}")
                print(f"File exists: {os.path.exists(actual_file_path)}")
                
                # It's a file path - use the new data loading function
                X_cat, X_cont, y, cat_mask, con_mask, cat_dims, scaler = prepare_data_for_model(
                    file_path=actual_file_path,
                    model_name=args.model,
                    window_size=args.window_size,
                    prediction_horizon=args.prediction_horizon,
                    patient_col=args.patient_col,
                    time_col=args.time_col,
                    glucose_col=args.glucose_col
                )
                dataset_name = Path(key).stem
                print(f"Successfully loaded data from file. Dataset name: {dataset_name}")
                
            else:
                # It's a predefined dataset name - use the old system
                print(f"NO FILE EXTENSION: Treating '{key}' as predefined dataset name")
                dataset_info = get_dataset_info(key)
                X_cat, X_cont, y, cat_mask, con_mask, cat_dims, scaler = prepare_glucose_data(
                    key, base_path=args.data_path
                )
                dataset_name = dataset_info['name']
                print(f"Successfully loaded predefined dataset: {dataset_name}")
            
            datasets_info.append((X_cat, X_cont, y, cat_mask, con_mask, cat_dims, scaler, dataset_name))
        
        model_name = f"QISN-Combined-{'-'.join(dataset_keys)}"
        output_dir = f"{args.output_dir}/combined/{'-'.join(dataset_keys)}"
        
    elif args.mode == 'finetune':
        if not args.source_dataset or not args.target_dataset:
            raise ValueError("--source_dataset and --target_dataset required for fine-tuning")
        
        # Load target dataset
        target_info = get_dataset_info(args.target_dataset)
        X_cat, X_cont, y, cat_mask, con_mask, cat_dims, scaler = prepare_glucose_data(
            args.target_dataset, base_path=args.data_path
        )
        datasets_info = [(X_cat, X_cont, y, cat_mask, con_mask, cat_dims, scaler, target_info['name'])]
        
        model_name = f"QISN-Finetuned-{args.source_dataset}-to-{args.target_dataset}"
        output_dir = f"{args.output_dir}/finetune/{args.source_dataset}_to_{args.target_dataset}"
        
        # Set pretrained path
        pretrained_path = f"{args.output_dir}/individual/{args.source_dataset}/best_model.pt"
    
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Training {model_name}")
    print(f"Output directory: {output_dir}")
    
    # Check if this is a traditional model
    traditional_models = ['randomforest', 'xgboost', 'lightgbm', 'catboost', 'mlp']
    is_traditional = args.model.lower() in traditional_models
    
    # Run training
    if is_traditional:
        # Use single-threaded training for traditional models
        print("Using traditional model training pipeline")
        model, history = train_traditional_model(
            datasets_info, args.model, output_dir, args, args.valid_split
        )
    elif args.mode == 'finetune':
        mp.spawn(
            train_model,
            args=(
                args.world_size, datasets_info, model_name, output_dir, args,
                args.batch_size, args.epochs, args.lr, args.weight_decay,
                args.valid_split, args.patience, 0.00005, 8, 0.5, 1e-6,
                5, args.resume, pretrained_path
            ),
            nprocs=args.world_size,
            join=True
        )
    else:
        mp.spawn(
            train_model,
            args=(
                args.world_size, datasets_info, model_name, output_dir, args,
                args.batch_size, args.epochs, args.lr, args.weight_decay,
                args.valid_split, args.patience, 0.00005, 8, 0.5, 1e-6,
                5, args.resume, None
            ),
            nprocs=args.world_size,
            join=True
        )
    
    print(f"Training complete! Results saved to {output_dir}")
    
    # Run SHAP analysis if requested
    if args.shap:
        print("Starting SHAP analysis...")
        try:
            from shap_utils import run_shap_analysis
            
            # Use the data path from arguments
            data_path = args.data_path if args.data_path else args.dataset
            model_path = f"{output_dir}/best_model.pt"
            shap_output_dir = f"{output_dir}/shap_analysis"
            
            # Run SHAP analysis
            analyzer, importance_df = run_shap_analysis(
                model_path=model_path,
                data_path=data_path,
                output_dir=shap_output_dir,
                device=args.device,
                n_background=args.shap_samples,
                n_test=args.shap_samples,
                max_evals=args.shap_evals
            )
            
            print(f"SHAP analysis complete! Results saved to {shap_output_dir}")
            print(f"Top 10 most important features:")
            print(importance_df.head(10)[['feature_name', 'mean_abs_shap']])
            
        except ImportError:
            print("SHAP not available. Install with: pip install shap")
        except Exception as e:
            print(f"SHAP analysis failed: {e}")

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()