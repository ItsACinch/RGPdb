"""Training callbacks for monitoring and control."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


@dataclass
class TrainingState:
    """Current state of training."""

    epoch: int = 0
    step: int = 0
    best_metric: float = float("inf")
    best_epoch: int = 0
    metrics_history: List[Dict[str, float]] = field(default_factory=list)


class Callback:
    """Base class for training callbacks."""

    def on_train_start(self, trainer, state: TrainingState) -> None:
        """Called at the start of training."""
        pass

    def on_train_end(self, trainer, state: TrainingState) -> None:
        """Called at the end of training."""
        pass

    def on_epoch_start(self, trainer, state: TrainingState) -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(
        self, trainer, state: TrainingState, metrics: Dict[str, float]
    ) -> None:
        """Called at the end of each epoch."""
        pass

    def on_batch_start(self, trainer, state: TrainingState, batch: Any) -> None:
        """Called at the start of each batch."""
        pass

    def on_batch_end(
        self, trainer, state: TrainingState, batch: Any, loss: float
    ) -> None:
        """Called at the end of each batch."""
        pass

    def should_stop(self) -> bool:
        """Return True to stop training early."""
        return False


class LoggingCallback(Callback):
    """Callback for logging training progress."""

    def __init__(
        self,
        log_every: int = 100,
        use_rich: bool = True,
    ):
        """
        Initialize logging callback.

        Args:
            log_every: Log every N steps
            use_rich: Use rich for pretty printing
        """
        self.log_every = log_every
        self.use_rich = use_rich
        self.step_times: List[float] = []
        self.last_log_time = time.time()

        if use_rich:
            try:
                from rich.console import Console
                from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

                self.console = Console()
                self.progress = None
            except ImportError:
                self.use_rich = False
                self.console = None

    def on_train_start(self, trainer, state: TrainingState) -> None:
        """Initialize progress tracking."""
        self.train_start_time = time.time()
        if self.use_rich:
            self.console.print("[bold green]Starting training...[/bold green]")

    def on_train_end(self, trainer, state: TrainingState) -> None:
        """Print final summary."""
        duration = time.time() - self.train_start_time
        if self.use_rich:
            self.console.print(f"\n[bold green]Training complete![/bold green]")
            self.console.print(f"Total time: {duration:.1f}s")
            self.console.print(f"Best epoch: {state.best_epoch}")
            self.console.print(f"Best metric: {state.best_metric:.4f}")
        else:
            print(f"\nTraining complete! Total time: {duration:.1f}s")
            print(f"Best epoch: {state.best_epoch}, Best metric: {state.best_metric:.4f}")

    def on_epoch_start(self, trainer, state: TrainingState) -> None:
        """Log epoch start."""
        self.epoch_start_time = time.time()
        self.epoch_losses: List[float] = []

    def on_epoch_end(
        self, trainer, state: TrainingState, metrics: Dict[str, float]
    ) -> None:
        """Log epoch summary."""
        epoch_time = time.time() - self.epoch_start_time
        avg_loss = sum(self.epoch_losses) / len(self.epoch_losses) if self.epoch_losses else 0

        if self.use_rich:
            self.console.print(
                f"[bold]Epoch {state.epoch}[/bold] | "
                f"Loss: {avg_loss:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )
            if metrics:
                metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
                self.console.print(f"  Metrics: {metrics_str}")
        else:
            print(f"Epoch {state.epoch} | Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s")

    def on_batch_end(
        self, trainer, state: TrainingState, batch: Any, loss: float
    ) -> None:
        """Log batch progress."""
        self.epoch_losses.append(loss)

        if state.step % self.log_every == 0:
            current_time = time.time()
            elapsed = current_time - self.last_log_time
            steps_per_sec = self.log_every / elapsed if elapsed > 0 else 0
            self.last_log_time = current_time

            if self.use_rich:
                self.console.print(
                    f"  Step {state.step} | Loss: {loss:.4f} | "
                    f"Speed: {steps_per_sec:.1f} steps/s"
                )
            else:
                print(f"  Step {state.step} | Loss: {loss:.4f}")


class CheckpointCallback(Callback):
    """Callback for saving model checkpoints."""

    def __init__(
        self,
        checkpoint_dir: Path,
        save_every: int = 1,
        save_best: bool = True,
        metric_name: str = "loss",
        mode: str = "min",
    ):
        """
        Initialize checkpoint callback.

        Args:
            checkpoint_dir: Directory to save checkpoints
            save_every: Save every N epochs
            save_best: Save the best model
            metric_name: Metric to monitor for best model
            mode: "min" or "max" for metric
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every
        self.save_best = save_best
        self.metric_name = metric_name
        self.mode = mode

    def on_epoch_end(
        self, trainer, state: TrainingState, metrics: Dict[str, float]
    ) -> None:
        """Save checkpoint if needed."""
        # Save periodic checkpoint
        if state.epoch % self.save_every == 0:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{state.epoch}.pt"
            self._save_checkpoint(trainer, state, checkpoint_path)

        # Save best model
        if self.save_best:
            metric_value = metrics.get(self.metric_name, float("inf"))

            is_best = False
            if self.mode == "min" and metric_value < state.best_metric:
                is_best = True
            elif self.mode == "max" and metric_value > state.best_metric:
                is_best = True

            if is_best:
                state.best_metric = metric_value
                state.best_epoch = state.epoch
                best_path = self.checkpoint_dir / "best_model.pt"
                self._save_checkpoint(trainer, state, best_path)

    def _save_checkpoint(
        self, trainer, state: TrainingState, path: Path
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": state.epoch,
            "step": state.step,
            "best_metric": state.best_metric,
            "best_epoch": state.best_epoch,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
        }

        if hasattr(trainer, "scheduler") and trainer.scheduler is not None:
            checkpoint["scheduler_state_dict"] = trainer.scheduler.state_dict()

        torch.save(checkpoint, path)


class EarlyStoppingCallback(Callback):
    """Callback for early stopping based on metric."""

    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.0,
        metric_name: str = "loss",
        mode: str = "min",
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as improvement
            metric_name: Metric to monitor
            mode: "min" or "max"
        """
        self.patience = patience
        self.min_delta = min_delta
        self.metric_name = metric_name
        self.mode = mode
        self.counter = 0
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self._should_stop = False

    def on_epoch_end(
        self, trainer, state: TrainingState, metrics: Dict[str, float]
    ) -> None:
        """Check if training should stop."""
        metric_value = metrics.get(self.metric_name, float("inf"))

        if self.mode == "min":
            improved = metric_value < self.best_value - self.min_delta
        else:
            improved = metric_value > self.best_value + self.min_delta

        if improved:
            self.best_value = metric_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self._should_stop = True
                print(f"Early stopping triggered after {state.epoch} epochs")

    def should_stop(self) -> bool:
        """Return whether to stop training."""
        return self._should_stop


class MetricsHistoryCallback(Callback):
    """Callback for tracking and saving metrics history."""

    def __init__(self, output_path: Optional[Path] = None):
        """
        Initialize metrics history.

        Args:
            output_path: Path to save metrics JSON (optional)
        """
        self.output_path = Path(output_path) if output_path else None
        self.history: List[Dict[str, Any]] = []

    def on_epoch_end(
        self, trainer, state: TrainingState, metrics: Dict[str, float]
    ) -> None:
        """Record metrics for this epoch."""
        record = {
            "epoch": state.epoch,
            "step": state.step,
            **metrics,
        }
        self.history.append(record)
        state.metrics_history = self.history

        if self.output_path:
            with open(self.output_path, "w") as f:
                json.dump(self.history, f, indent=2)

    def on_train_end(self, trainer, state: TrainingState) -> None:
        """Save final metrics."""
        if self.output_path:
            with open(self.output_path, "w") as f:
                json.dump(self.history, f, indent=2)
