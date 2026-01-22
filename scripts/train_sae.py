"""
SAE Training Script.

Train Sparse Autoencoders on extracted hidden states.
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any
from tqdm import tqdm
import json

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models import JumpReLUSAE
from config import load_config


class SAETrainer:
    """Trainer for Sparse Autoencoders."""

    def __init__(
        self,
        sae: JumpReLUSAE,
        layer_idx: int,
        hidden_states_path: str,
        config: Dict[str, Any],
        device: str = 'cuda'
    ):
        """
        Initialize SAE trainer.

        Args:
            sae: JumpReLUSAE model to train.
            layer_idx: Layer index being trained.
            hidden_states_path: Path to extracted hidden states.
            config: Training configuration.
            device: Device to train on.
        """
        self.sae = sae.to(device)
        self.layer_idx = layer_idx
        self.config = config
        self.device = device

        # Load hidden states
        print(f"Loading hidden states from {hidden_states_path}...")
        data = torch.load(hidden_states_path)
        self.hidden_states = data['hidden_states']  # [num_tokens, d_model]
        print(f"Loaded {self.hidden_states.shape[0]:,} tokens")

        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            self.sae.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.0)
        )

        # Setup scheduler (T_max will be set during train())
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 100)  # Default to 100 if not specified
        )

        # Training state
        self.epoch = 0
        self.best_loss = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'recon_loss': [],
            'sparsity_loss': [],
            'train_sparsity': [],
            'sparsity': [],
            'lr': []
        }

    def load_data(self):
        """Load and return hidden states."""
        return self.hidden_states

    def compute_loss(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute loss for a batch.

        Args:
            batch: Input tensor [batch_size, d_model]

        Returns:
            Dictionary with loss components.
        """
        # Add sequence dimension if needed
        if batch.ndim == 2:
            batch = batch.unsqueeze(1)  # [batch_size, 1, d_model]

        # Forward pass and compute loss
        loss_dict = self.sae.loss(batch, return_components=True)

        # Rename keys to match test expectations
        return {
            'total_loss': loss_dict['total_loss'],
            'reconstruction_loss': loss_dict['recon_loss'],
            'l1_loss': loss_dict['sparsity_loss'],
            'sparsity': loss_dict['sparsity']
        }

    def create_dataloader(self, batch_size: int, shuffle: bool = True):
        """Create DataLoader for hidden states."""
        dataset = torch.utils.data.TensorDataset(self.hidden_states)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=(self.device == 'cuda')
        )
        return dataloader

    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch."""
        self.sae.train()

        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_sparsity_loss = 0.0
        epoch_sparsity = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch+1}")

        for batch in pbar:
            x = batch[0].to(self.device)  # [batch_size, d_model]

            # Add sequence dimension: [batch_size, 1, d_model]
            x = x.unsqueeze(1)
            
            # SAE expects float32
            x = x.to(dtype=torch.float32)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass and loss
            loss_dict = self.sae.loss(x, return_components=True)

            # Backward pass
            loss_dict['total_loss'].backward()

            # Gradient clipping
            if self.config.get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.sae.parameters(),
                    self.config['gradient_clip']
                )

            # Optimizer step
            self.optimizer.step()

            # Normalize decoder periodically
            if num_batches % 100 == 0:
                self.sae.normalize_decoder_weights()

            # Track metrics
            epoch_loss += loss_dict['total_loss'].item()
            epoch_recon += loss_dict['recon_loss'].item()
            epoch_sparsity_loss += loss_dict['sparsity_loss'].item()
            epoch_sparsity += loss_dict['sparsity'].item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss'].item():.6f}",
                'recon': f"{loss_dict['recon_loss'].item():.6f}",
                'sparsity': f"{loss_dict['sparsity'].item():.4f}"
            })

        # Return average metrics
        return {
            'train_loss': epoch_loss / num_batches,
            'recon_loss': epoch_recon / num_batches,
            'sparsity_loss': epoch_sparsity_loss / num_batches,
            'sparsity': epoch_sparsity / num_batches
        }

    def validate(self, dataloader) -> Dict[str, float]:
        """Validate on held-out data."""
        self.sae.eval()

        val_loss = 0.0
        val_recon = 0.0
        val_sparsity = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                x = batch[0].to(self.device)
                x = x.unsqueeze(1)
                x = x.to(dtype=torch.float32)

                loss_dict = self.sae.loss(x, return_components=True)

                val_loss += loss_dict['total_loss'].item()
                val_recon += loss_dict['recon_loss'].item()
                val_sparsity += loss_dict['sparsity'].item()
                num_batches += 1

        return {
            'val_loss': val_loss / num_batches,
            'val_recon': val_recon / num_batches,
            'val_sparsity': val_sparsity / num_batches
        }

    def train(
        self,
        epochs: int,
        batch_size: int,
        val_split: float = 0.05,
        save_dir: Optional[str] = None
    ):
        """
        Train SAE for multiple epochs.

        Args:
            epochs: Number of epochs.
            batch_size: Batch size.
            val_split: Fraction of data for validation.
            save_dir: Directory to save checkpoints.
        """
        # Split data
        num_val = int(len(self.hidden_states) * val_split)
        num_train = len(self.hidden_states) - num_val

        train_states = self.hidden_states[:num_train]
        val_states = self.hidden_states[num_train:]

        # Create dataloaders
        train_dataset = torch.utils.data.TensorDataset(train_states)
        val_dataset = torch.utils.data.TensorDataset(val_states)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=(self.device == 'cuda'),
            persistent_workers=True,
            prefetch_factor=4
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=(self.device == 'cuda'),
            persistent_workers=True
        )

        print(f"\nTraining SAE for layer {self.layer_idx}")
        print(f"Train samples: {num_train:,}")
        print(f"Val samples: {num_val:,}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}\n")

        for epoch in range(epochs):
            self.epoch = epoch

            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.validate(val_loader)

            # Update history
            self.history['train_loss'].append(train_metrics['train_loss'])
            self.history['val_loss'].append(val_metrics['val_recon'])
            self.history['recon_loss'].append(train_metrics['recon_loss'])
            self.history['sparsity_loss'].append(train_metrics['sparsity_loss'])
            self.history['train_sparsity'].append(train_metrics['sparsity'])
            self.history['sparsity'].append(val_metrics['val_sparsity'])
            self.history['lr'].append(self.scheduler.get_last_lr()[0])

            # Print summary
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_metrics['train_loss']:.6f}")
            print(f"  Val Recon Loss: {val_metrics['val_recon']:.6f}")
            print(f"  Val Sparsity: {val_metrics['val_sparsity']:.4f}")
            print(f"  LR: {self.scheduler.get_last_lr()[0]:.6f}")

            # Save best model
            if val_metrics['val_recon'] < self.best_loss:
                self.best_loss = val_metrics['val_recon']
                if save_dir:
                    self.save_checkpoint(save_dir, is_best=True)
                print(f"  ✓ New best validation loss: {self.best_loss:.6f}")

            # Scheduler step
            self.scheduler.step()

        # Save final model
        if save_dir:
            self.save_checkpoint(save_dir, is_best=False)

        print(f"\nTraining complete!")
        print(f"Best validation recon loss: {self.best_loss:.6f}")

        return self.history

    def save_checkpoint(self, save_dir: str, is_best: bool = False):
        """Save model checkpoint."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'layer_idx': self.layer_idx,
            'epoch': self.epoch,
            'model_state_dict': self.sae.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'history': self.history,
            'config': self.config
        }

        if is_best:
            filename = save_dir / f'sae_layer_{self.layer_idx}_best.pt'
        else:
            filename = save_dir / f'sae_layer_{self.layer_idx}_final.pt'

        torch.save(checkpoint, filename)
        print(f"Saved checkpoint to {filename}")

        # Also save history as JSON
        history_file = save_dir / f'sae_layer_{self.layer_idx}_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)


def train_sae_for_layer(
    layer_idx: int,
    hidden_states_path: str,
    config_path: Optional[str] = None,
    save_dir: str = './models/checkpoints',
    device: str = 'cuda'
):
    """
    Train SAE for a specific layer.

    Args:
        layer_idx: Layer index to train.
        hidden_states_path: Path to extracted hidden states.
        config_path: Path to config file.
        save_dir: Directory to save checkpoints.
        device: Device to use.

    Returns:
        Training history.
    """
    # Load config
    config = load_config(config_path)

    # Create SAE
    sae = JumpReLUSAE(
        d_model=config.sae.d_model,
        d_hidden=config.sae.d_hidden,
        threshold=config.sae.threshold,
        lambda_sparse=config.sae.lambda_sparse
    )

    # Create trainer
    trainer = SAETrainer(
        sae=sae,
        layer_idx=layer_idx,
        hidden_states_path=hidden_states_path,
        config={
            'learning_rate': config.sae_training.learning_rate,
            'weight_decay': config.sae_training.weight_decay,
            'gradient_clip': config.sae_training.gradient_clip,
            'epochs': config.sae_training.epochs
        },
        device=device
    )

    # Train
    history = trainer.train(
        epochs=config.sae_training.epochs,
        batch_size=config.sae_training.batch_size,
        save_dir=save_dir
    )

    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SAE for a layer')
    parser.add_argument('--layer', type=int, required=True, help='Layer index (0-11)')
    parser.add_argument('--states', type=str, required=True, help='Path to hidden states')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--save-dir', type=str, default='./models/checkpoints', help='Save directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')

    args = parser.parse_args()

    history = train_sae_for_layer(
        layer_idx=args.layer,
        hidden_states_path=args.states,
        config_path=args.config,
        save_dir=args.save_dir,
        device=args.device
    )

    print("\nTraining history saved to checkpoints directory")
