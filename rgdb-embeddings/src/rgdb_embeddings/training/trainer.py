"""Training loops for embedding models."""

from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import TrainingConfig
from .callbacks import Callback, TrainingState
from .losses import RotatELoss, TripletLoss


class EmbeddingTrainer:
    """
    General purpose trainer for embedding models.

    Supports:
    - Fine-tuning sentence transformers
    - Contrastive learning
    - Custom loss functions
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        loss_fn: Optional[nn.Module] = None,
        callbacks: Optional[List[Callback]] = None,
    ):
        """
        Initialize the trainer.

        Args:
            model: The model to train
            config: Training configuration
            loss_fn: Loss function (default: TripletLoss)
            callbacks: List of callbacks
        """
        self.model = model
        self.config = config
        self.loss_fn = loss_fn or TripletLoss()
        self.callbacks = callbacks or []

        # Setup device with graceful fallback
        if config.device == "cuda" and not torch.cuda.is_available():
            import warnings
            warnings.warn(
                "CUDA requested but not available. Falling back to CPU. "
                "For GPU training, ensure CUDA is installed and a GPU is available.",
                RuntimeWarning
            )
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(config.device)

        self.model = self.model.to(self.device)

        # Setup optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler will be set up in train()
        self.scheduler = None

        # Training state
        self.state = TrainingState()

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> TrainingState:
        """
        Run the training loop.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader

        Returns:
            Training state with history
        """
        # Setup scheduler
        total_steps = len(train_loader) * self.config.epochs
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=self.config.warmup_steps,
        )
        main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - self.config.warmup_steps,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[self.config.warmup_steps],
        )

        # Notify callbacks
        for callback in self.callbacks:
            callback.on_train_start(self, self.state)

        # Training loop
        for epoch in range(self.config.epochs):
            self.state.epoch = epoch + 1

            # Notify callbacks
            for callback in self.callbacks:
                callback.on_epoch_start(self, self.state)

            # Train one epoch
            train_loss = self._train_epoch(train_loader)

            # Validate if loader provided
            metrics = {"loss": train_loss}
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

            # Notify callbacks
            for callback in self.callbacks:
                callback.on_epoch_end(self, self.state, metrics)

            # Check for early stopping
            should_stop = any(cb.should_stop() for cb in self.callbacks)
            if should_stop:
                break

        # Notify callbacks
        for callback in self.callbacks:
            callback.on_train_end(self, self.state)

        return self.state

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {self.state.epoch}")
        for batch in pbar:
            self.state.step += 1

            # Notify callbacks
            for callback in self.callbacks:
                callback.on_batch_start(self, self.state, batch)

            # Move batch to device
            batch = self._to_device(batch)

            # Forward pass
            loss = self._compute_loss(batch)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            # Track loss
            loss_val = loss.item()
            total_loss += loss_val
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})

            # Notify callbacks
            for callback in self.callbacks:
                callback.on_batch_end(self, self.state, batch, loss_val)

        return total_loss / num_batches

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss for a batch."""
        # Handle different batch formats
        if "anchor_emb" in batch:
            # Precomputed embeddings
            anchor = batch["anchor_emb"]
            positive = batch["positive_emb"]
            negative = batch["negative_emb"]
        elif "anchor_text" in batch:
            # Text inputs - encode on the fly
            anchor = self.model.encode(batch["anchor_text"])
            positive = self.model.encode(batch["positive_text"])
            negative = self.model.encode(batch["negative_text"])
        else:
            raise ValueError("Batch must contain embeddings or text")

        return self.loss_fn(anchor, positive, negative)

    def _to_device(self, batch: Union[Dict, torch.Tensor]) -> Union[Dict, torch.Tensor]:
        """Move batch to device."""
        if isinstance(batch, dict):
            return {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
        elif isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        return batch

    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = self._to_device(batch)
                loss = self._compute_loss(batch)
                total_loss += loss.item()
                num_batches += 1

        return {"loss": total_loss / num_batches}

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "state": self.state,
        }, path)

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.state = checkpoint.get("state", TrainingState())


class RotatETrainer:
    """
    Specialized trainer for RotatE-style models.

    Optimized for knowledge graph embeddings with relation rotations.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainingConfig] = None,
        callbacks: Optional[List[Callback]] = None,
    ):
        """
        Initialize the RotatE trainer.

        Args:
            model: RotatE model
            config: Training configuration
            callbacks: List of callbacks
        """
        self.model = model
        self.config = config or TrainingConfig(mode="rotate")
        self.callbacks = callbacks or []

        # Setup device
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Loss
        self.loss_fn = RotatELoss(margin=self.config.margin)

        # State
        self.state = TrainingState()

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> TrainingState:
        """Run the training loop."""
        # Notify callbacks
        for callback in self.callbacks:
            callback.on_train_start(self, self.state)

        for epoch in range(self.config.epochs):
            self.state.epoch = epoch + 1

            for callback in self.callbacks:
                callback.on_epoch_start(self, self.state)

            train_loss = self._train_epoch(train_loader)

            metrics = {"loss": train_loss}
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

            for callback in self.callbacks:
                callback.on_epoch_end(self, self.state, metrics)

            if any(cb.should_stop() for cb in self.callbacks):
                break

        for callback in self.callbacks:
            callback.on_train_end(self, self.state)

        return self.state

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {self.state.epoch}")
        for batch in pbar:
            self.state.step += 1

            # Move to device
            head_ids = batch["head_id"].to(self.device)
            tail_ids = batch["tail_id"].to(self.device)
            angle_bins = batch["angle_bin"].to(self.device)
            neg_tails = batch["negative_tails"].to(self.device)

            # Forward pass - compute scores
            pos_scores = self.model.score_triple(head_ids, tail_ids, angle_bins)

            # Negative scores
            batch_size, num_neg = neg_tails.shape
            head_expanded = head_ids.unsqueeze(1).expand(-1, num_neg).reshape(-1)
            neg_flat = neg_tails.reshape(-1)
            bins_expanded = angle_bins.unsqueeze(1).expand(-1, num_neg).reshape(-1)
            neg_scores = self.model.score_triple(head_expanded, neg_flat, bins_expanded)
            neg_scores = neg_scores.view(batch_size, num_neg)

            # Loss
            loss = self.loss_fn(pos_scores, neg_scores)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )

            self.optimizer.step()

            # Track
            loss_val = loss.item()
            total_loss += loss_val
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})

        return total_loss / num_batches

    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model with MRR and Hits@K."""
        self.model.eval()

        all_ranks = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                head_ids = batch["head_id"].to(self.device)
                tail_ids = batch["tail_id"].to(self.device)
                angle_bins = batch["angle_bin"].to(self.device)

                # Score all entities as potential tails
                batch_size = head_ids.size(0)
                num_entities = self.model.num_nodes

                # Get head embeddings rotated
                head_emb = self.model.node_embeddings(head_ids)
                rotated_head = self.model.rotate_batch(head_emb, angle_bins)

                # Get all entity embeddings
                all_emb = self.model.node_embeddings.weight

                # Compute distances to all entities
                # [batch, dim] @ [dim, num_entities] -> [batch, num_entities]
                re_head, im_head = rotated_head.chunk(2, dim=-1)
                re_all, im_all = all_emb.chunk(2, dim=-1)

                # L2 distance in complex space
                diff_re = re_head.unsqueeze(2) - re_all.t().unsqueeze(0)
                diff_im = im_head.unsqueeze(2) - im_all.t().unsqueeze(0)
                distances = torch.sqrt(diff_re ** 2 + diff_im ** 2 + 1e-8).sum(dim=1)

                # Rank the true tail
                for i in range(batch_size):
                    true_tail = tail_ids[i].item()
                    scores = distances[i]
                    rank = (scores < scores[true_tail]).sum().item() + 1
                    all_ranks.append(rank)

        ranks = np.array(all_ranks)
        mrr = np.mean(1.0 / ranks)
        hits_1 = np.mean(ranks <= 1)
        hits_3 = np.mean(ranks <= 3)
        hits_10 = np.mean(ranks <= 10)

        return {
            "mrr": mrr,
            "hits@1": hits_1,
            "hits@3": hits_3,
            "hits@10": hits_10,
        }

    def get_embeddings(self) -> np.ndarray:
        """Get all node embeddings as numpy array."""
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model.node_embeddings.weight.cpu().numpy()
        return embeddings.astype(np.float32)
