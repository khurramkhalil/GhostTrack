"""
SAE Trainer.

Wraps the training loop for :class:`~ghosttrack.sae.model.JumpReLUSAE`.
Designed to work both on-cluster (loading large hidden-state tensors from
disk) and in unit tests (passing small synthetic tensors directly).

Typical usage::

    from ghosttrack.sae import JumpReLUSAE, SAETrainer
    from ghosttrack.utils import load_config

    config = load_config("config/model_configs/gpt2-medium.yaml")
    sae = JumpReLUSAE(d_model=config.sae.d_model, d_hidden=config.sae.d_hidden)
    trainer = SAETrainer(device="cuda")
    history = trainer.train(
        sae=sae,
        hidden_states=states,          # [N, d_model] tensor
        config=config.sae_training,
        layer_idx=0,
        save_dir="checkpoints/",
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.utils.data
from tqdm import tqdm

from .model import JumpReLUSAE


class SAETrainer:
    """
    Trains a :class:`JumpReLUSAE` on extracted hidden states.

    Args:
        device: ``"cuda"`` or ``"cpu"``.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------ #
    # Main entry point                                                     #
    # ------------------------------------------------------------------ #

    def train(
        self,
        sae: JumpReLUSAE,
        hidden_states: torch.Tensor,
        config: Any,
        layer_idx: int = 0,
        save_dir: Optional[str] = None,
    ) -> Dict[str, list]:
        """
        Train *sae* on *hidden_states*.

        Args:
            sae: The SAE to train (moved to ``self.device`` internally).
            hidden_states: ``[N, d_model]`` float32 tensor of token activations.
            config: :class:`~ghosttrack.utils.config.SAETrainingConfig` (or any
                object with attributes ``learning_rate``, ``weight_decay``,
                ``gradient_clip``, ``epochs``, ``batch_size``, ``val_split``).
            layer_idx: Layer index — used only for checkpoint filenames.
            save_dir: If provided, save best and final checkpoints here.

        Returns:
            Training history dict with lists for ``train_loss``, ``val_loss``,
            ``recon_loss``, ``sparsity_loss``, ``sparsity``, ``lr``.
        """
        sae = sae.to(self.device)

        # Hyper-parameters
        lr = float(config.learning_rate)
        weight_decay = float(getattr(config, "weight_decay", 0.0))
        grad_clip = float(getattr(config, "gradient_clip", 0.0))
        epochs = int(config.epochs)
        batch_size = int(config.batch_size)
        val_split = float(getattr(config, "val_split", 0.05))

        # Optimiser + scheduler
        optimizer = torch.optim.Adam(sae.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Train / val split — keep float16 slices (views, no copy).
        # Conversion float16→float32 happens on-device in the training loop
        # via x.to(device, dtype=float32), avoiding a full float32 copy in CPU RAM.
        # For Phi-2/Qwen each layer file is ~15 GB; calling .float() here would
        # create a 30 GB copy and OOM on 48 Gi nodes.
        n = len(hidden_states)
        n_val = max(1, int(n * val_split))
        n_train = n - n_val
        train_states = hidden_states[:n_train]   # view, preserves float16
        val_states = hidden_states[n_train:]     # view, preserves float16

        # DataLoaders — num_workers=0 avoids forking large tensors into worker
        # processes, which can trigger CoW page duplication on some kernels.
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_states),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(val_states),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        history: Dict[str, list] = {
            "train_loss": [], "val_loss": [], "recon_loss": [],
            "sparsity_loss": [], "sparsity": [], "lr": [],
        }
        best_val_loss = float("inf")

        for epoch in range(epochs):
            # --- train ---
            train_metrics = self._train_epoch(sae, train_loader, optimizer, grad_clip, epoch)
            # --- validate ---
            val_metrics = self._validate(sae, val_loader)
            # --- scheduler ---
            scheduler.step()

            history["train_loss"].append(train_metrics["train_loss"])
            history["val_loss"].append(val_metrics["val_loss"])
            history["recon_loss"].append(train_metrics["recon_loss"])
            history["sparsity_loss"].append(train_metrics["sparsity_loss"])
            history["sparsity"].append(val_metrics["val_sparsity"])
            history["lr"].append(scheduler.get_last_lr()[0])

            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                if save_dir:
                    self._save(sae, optimizer, scheduler, layer_idx, epoch,
                               best_val_loss, history, config, save_dir, best=True)

        if save_dir:
            self._save(sae, optimizer, scheduler, layer_idx, epoch,
                       best_val_loss, history, config, save_dir, best=False)

        return history

    # ------------------------------------------------------------------ #
    # Internal training helpers                                            #
    # ------------------------------------------------------------------ #

    def _train_epoch(
        self,
        sae: JumpReLUSAE,
        loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        grad_clip: float,
        epoch: int,
    ) -> Dict[str, float]:
        sae.train()
        totals = dict(train_loss=0.0, recon_loss=0.0, sparsity_loss=0.0, n=0)

        for step, (x,) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}", leave=False)):
            x = x.to(self.device, dtype=torch.float32)
            # SAE expects [batch, seq, d_model]; hidden states are [batch, d_model]
            x = x.unsqueeze(1)

            optimizer.zero_grad()
            loss_dict = sae.loss(x, return_components=True)
            loss_dict["total_loss"].backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(sae.parameters(), grad_clip)
            optimizer.step()

            if step % 100 == 0:
                sae.normalize_decoder_weights()

            totals["train_loss"] += loss_dict["total_loss"].item()
            totals["recon_loss"] += loss_dict["recon_loss"].item()
            totals["sparsity_loss"] += loss_dict["sparsity_loss"].item()
            totals["n"] += 1

        n = max(totals.pop("n"), 1)
        return {k: v / n for k, v in totals.items()}

    @torch.no_grad()
    def _validate(
        self,
        sae: JumpReLUSAE,
        loader: torch.utils.data.DataLoader,
    ) -> Dict[str, float]:
        sae.eval()
        total_loss = total_sparsity = 0.0
        n = 0

        for (x,) in loader:
            x = x.to(self.device, dtype=torch.float32).unsqueeze(1)
            loss_dict = sae.loss(x, return_components=True)
            total_loss += loss_dict["recon_loss"].item()
            total_sparsity += loss_dict["sparsity"].item()
            n += 1

        n = max(n, 1)
        return {"val_loss": total_loss / n, "val_sparsity": total_sparsity / n}

    # ------------------------------------------------------------------ #
    # Checkpointing                                                        #
    # ------------------------------------------------------------------ #

    def _save(
        self,
        sae: JumpReLUSAE,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        layer_idx: int,
        epoch: int,
        best_loss: float,
        history: Dict,
        config: Any,
        save_dir: str,
        best: bool,
    ) -> None:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        suffix = "best" if best else "final"
        ckpt_path = out / f"sae_layer_{layer_idx}_{suffix}.pt"

        torch.save(
            {
                "layer_idx": layer_idx,
                "epoch": epoch,
                "model_state_dict": sae.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_loss": best_loss,
                "history": history,
            },
            ckpt_path,
        )

        # Companion JSON history for easy inspection
        history_path = out / f"sae_layer_{layer_idx}_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
