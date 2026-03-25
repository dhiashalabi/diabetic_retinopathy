"""
Training engine with AMP, early stopping on QWK, cosine LR with warmup,
gradient clipping, and two-phase schedule.

Designed for maximum accuracy on the DR grading task:
- Phase 1: head-only with high LR
- Phase 2: fine-tune with low LR, warmup, and cosine decay
- Checkpointing on Quadratic Weighted Kappa (clinical metric)
- Label smoothing + class-weighted loss
- Mixed-precision training for speed
"""

import os
import json
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm

from . import config
from .utils import get_logger

logger = get_logger(__name__)


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy with label smoothing *and* class weights.

    Combines two strategies for accuracy on imbalanced ordinal data:
    - Label smoothing prevents overconfident predictions
    - Class weights up-weight rare severe/proliferative cases
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES,
                 smoothing: float = config.LABEL_SMOOTHING,
                 weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.weight is not None:
            weight = self.weight.to(pred.device)
        else:
            weight = None
        return nn.functional.cross_entropy(
            pred, target,
            weight=weight,
            label_smoothing=self.smoothing,
        )


class CosineWarmupScheduler:
    """
    Linear warmup followed by cosine annealing.

    Warmup prevents destabilising the pretrained weights during the
    first few epochs of fine-tuning.
    """

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 min_lr: float = 1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            # Linear warmup
            factor = (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            factor = 0.5 * (1.0 + np.cos(np.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = max(base_lr * factor, self.min_lr)

    def get_last_lr(self) -> list[float]:
        return [pg["lr"] for pg in self.optimizer.param_groups]


class EarlyStopping:
    """Stop training when the monitored metric stops improving."""

    def __init__(self, patience: int = config.EARLY_STOPPING_PATIENCE,
                 mode: str = "max", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def state_dict(self) -> dict:
        return {
            "counter": self.counter,
            "best_score": self.best_score,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, d: dict) -> None:
        self.counter = d["counter"]
        self.best_score = d["best_score"]
        self.should_stop = d["should_stop"]


class Trainer:
    """
    Two-phase training engine for the DR classifier.

    Optimised for high accuracy with:
    - Mixed-precision training (AMP)
    - Gradient clipping to stabilise fine-tuning
    - Label-smoothed, class-weighted loss
    - Checkpointing on QWK (the clinical gold-standard metric)
    - Cosine LR with warmup for Phase 2
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        class_weights: Optional[torch.Tensor] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.class_weights = class_weights
        self.scaler = GradScaler(device.type, enabled=device.type == "cuda")
        self.history: dict[str, list] = {
            "train_loss": [], "train_acc": [], "train_kappa": [],
            "val_loss": [], "val_acc": [], "val_kappa": [], "lr": [],
        }
        self.best_kappa = -1.0

    def _build_criterion(self) -> nn.Module:
        return LabelSmoothingCrossEntropy(
            weight=self.class_weights,
            smoothing=config.LABEL_SMOOTHING,
        )

    def train_one_epoch(
        self, loader: DataLoader, optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> dict:
        self.model.train()

        # Keep BN frozen in backbone during fine-tune
        from .model import DRModel
        if isinstance(self.model, DRModel):
            for module in self.model.backbone.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    module.eval()

        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(loader, desc="  Train", leave=False)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=self.device.type == "cuda"):
                outputs = self.model(images)
                loss = criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), config.GRADIENT_CLIP_NORM,
            )
            self.scaler.step(optimizer)
            self.scaler.update()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100 * correct / total:.1f}%",
            )

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        epoch_kappa = cohen_kappa_score(
            all_labels, all_preds, weights="quadratic",
        )

        return {"loss": epoch_loss, "acc": epoch_acc, "kappa": epoch_kappa}

    @torch.no_grad()
    def validate(self, loader: DataLoader, criterion: nn.Module) -> dict:
        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(loader, desc="  Valid", leave=False)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast("cuda", enabled=self.device.type == "cuda"):
                outputs = self.model(images)
                loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        epoch_kappa = cohen_kappa_score(
            all_labels, all_preds, weights="quadratic",
        )

        return {"loss": epoch_loss, "acc": epoch_acc, "kappa": epoch_kappa}

    def _save_checkpoint(self, epoch: int, phase: str, metrics: dict,
                         optimizer=None, scaler=None):
        path = os.path.join(
            config.CHECKPOINTS_DIR, f"best_model_{phase}.pth",
        )
        ckpt = {
            "epoch": epoch,
            "phase": phase,
            "model_state_dict": self.model.state_dict(),
            "metrics": metrics,
        }
        if optimizer is not None:
            ckpt["optimizer_state_dict"] = optimizer.state_dict()
        if scaler is not None:
            ckpt["scaler_state_dict"] = scaler.state_dict()
        torch.save(ckpt, path)
        logger.info(f"Checkpoint saved: {path} (QWK={metrics['kappa']:.4f})")

    def _save_resume_state(
        self, *, phase: str, next_epoch: int, epochs: int,
        lr: float, weight_decay: float, warmup_epochs: int,
        optimizer: torch.optim.Optimizer, early_stop: EarlyStopping,
        phase_done: bool,
    ) -> None:
        """Save full training state after each epoch for crash recovery."""
        payload = {
            "model_state_dict": self.model.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": {k: list(v) for k, v in self.history.items()},
            "best_kappa": self.best_kappa,
            "phase": phase,
            "next_epoch": next_epoch,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "warmup_epochs": warmup_epochs,
            "early_stop": early_stop.state_dict(),
            "phase_done": phase_done,
        }
        path = config.RESUME_STATE_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(payload, path)

    @staticmethod
    def has_resume_state() -> bool:
        return os.path.isfile(config.RESUME_STATE_PATH)

    def _load_resume_state(self) -> dict:
        path = config.RESUME_STATE_PATH
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        self.scaler.load_state_dict(state["scaler_state_dict"])
        self.history = state["history"]
        self.best_kappa = state["best_kappa"]
        logger.info(
            f"Restored resume state: phase={state['phase']}, "
            f"next_epoch={state['next_epoch']}, best_kappa={state['best_kappa']:.4f}"
        )
        return state

    def _clear_resume_state(self) -> None:
        path = config.RESUME_STATE_PATH
        if os.path.isfile(path):
            os.remove(path)
            logger.info("Training complete — resume state cleared.")

    def resume_from_checkpoint(
        self, path: str, optimizer=None, scaler=None,
    ) -> int:
        """
        Restore model, optimizer, and scaler from a checkpoint.

        Returns the epoch number to resume from (epoch + 1).
        """
        if not os.path.exists(path):
            logger.warning(f"No checkpoint found at {path}")
            return 0
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scaler is not None and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.best_kappa = ckpt["metrics"].get("kappa", -1.0)
        resume_epoch = ckpt["epoch"] + 1
        logger.info(
            f"Resumed from {path} (epoch {resume_epoch}, "
            f"QWK={ckpt['metrics']['kappa']:.4f})"
        )
        return resume_epoch

    def fit_phase(
        self,
        phase: str,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        epochs: int,
        lr: float,
        weight_decay: float,
        warmup_epochs: int = 0,
        *,
        _start_epoch: int = 0,
        _optimizer: Optional[torch.optim.Optimizer] = None,
        _early_stop: Optional[EarlyStopping] = None,
    ) -> bool:
        """
        Run one training phase (head-only or fine-tuning).

        Returns True if the phase completed normally, False if early-stopped.

        Internal resume parameters (prefixed with _) are passed by fit() when
        restoring from a crash — callers should never set them directly.
        """
        criterion = self._build_criterion()

        if _optimizer is not None:
            optimizer = _optimizer
        else:
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=lr, weight_decay=weight_decay,
            )

        scheduler = CosineWarmupScheduler(
            optimizer, warmup_epochs=warmup_epochs, total_epochs=epochs,
        )
        early_stop = _early_stop or EarlyStopping(
            patience=config.EARLY_STOPPING_PATIENCE, mode="max",
        )

        remaining = epochs - _start_epoch
        logger.info(f"\n{'='*60}")
        if _start_epoch > 0:
            logger.info(
                f"Phase: {phase.upper()} | Resuming from epoch {_start_epoch+1}/{epochs} "
                f"({remaining} remaining) | LR: {lr}"
            )
        else:
            logger.info(f"Phase: {phase.upper()} | Epochs: {epochs} | LR: {lr}")
        logger.info(f"{'='*60}")

        early_stopped = False

        for epoch in range(_start_epoch, epochs):
            t0 = time.time()

            scheduler.step(epoch)
            current_lr = optimizer.param_groups[0]["lr"]

            train_metrics = self.train_one_epoch(
                train_loader, optimizer, criterion,
            )
            val_metrics = self.validate(valid_loader, criterion)

            # Append to global history
            for split, metrics in [("train", train_metrics), ("val", val_metrics)]:
                self.history[f"{split}_loss"].append(metrics["loss"])
                self.history[f"{split}_acc"].append(metrics["acc"])
                self.history[f"{split}_kappa"].append(metrics["kappa"])
            self.history["lr"].append(current_lr)

            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch+1:02d}/{epochs} ({elapsed:.0f}s) | "
                f"LR: {current_lr:.2e} | "
                f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['acc']:.4f} "
                f"QWK: {train_metrics['kappa']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['acc']:.4f} "
                f"QWK: {val_metrics['kappa']:.4f}"
            )

            if val_metrics["kappa"] > self.best_kappa:
                self.best_kappa = val_metrics["kappa"]
                self._save_checkpoint(
                    epoch, phase, val_metrics,
                    optimizer=optimizer, scaler=self.scaler,
                )

            if early_stop(val_metrics["kappa"]):
                logger.info(
                    f"Early stopping at epoch {epoch+1} "
                    f"(best QWK: {early_stop.best_score:.4f})"
                )
                early_stopped = True
                self._save_resume_state(
                    phase=phase, next_epoch=epoch + 1, epochs=epochs,
                    lr=lr, weight_decay=weight_decay,
                    warmup_epochs=warmup_epochs, optimizer=optimizer,
                    early_stop=early_stop, phase_done=True,
                )
                break

            # Persist resume state after every successful epoch
            self._save_resume_state(
                phase=phase, next_epoch=epoch + 1, epochs=epochs,
                lr=lr, weight_decay=weight_decay,
                warmup_epochs=warmup_epochs, optimizer=optimizer,
                early_stop=early_stop,
                phase_done=(epoch + 1 == epochs),
            )

        return not early_stopped

    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        *,
        resume: bool = True,
    ) -> dict:
        """
        Full two-phase training with automatic crash recovery.

        When ``resume=True`` (default), the method checks for a saved resume
        state on disk.  If one exists it restores model weights, optimizer,
        scaler, early-stopping state, and training history, then picks up
        exactly where training left off — even mid-phase.

        Set ``resume=False`` to force a fresh start (the resume file is
        ignored and deleted).

        Phase 1 — head-only (high LR, no warmup)
        Phase 2 — fine-tune from block5 (low LR, warmup)
        """
        from .model import freeze_backbone, unfreeze_backbone, count_parameters

        skip_phase1 = False
        p1_start_epoch = 0
        p2_start_epoch = 0
        p1_optimizer = None
        p2_optimizer = None
        p1_early_stop = None
        p2_early_stop = None

        if resume and self.has_resume_state():
            state = self._load_resume_state()
            phase = state["phase"]

            if phase == "phase1" and not state["phase_done"]:
                p1_start_epoch = state["next_epoch"]
                freeze_backbone(self.model)
                p1_optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=state["lr"], weight_decay=state["weight_decay"],
                )
                p1_optimizer.load_state_dict(state["optimizer_state_dict"])
                p1_early_stop = EarlyStopping(
                    patience=config.EARLY_STOPPING_PATIENCE, mode="max",
                )
                p1_early_stop.load_state_dict(state["early_stop"])

            elif phase == "phase1" and state["phase_done"]:
                skip_phase1 = True

            elif phase == "phase2" and not state["phase_done"]:
                skip_phase1 = True
                p2_start_epoch = state["next_epoch"]
                unfreeze_backbone(self.model, from_block=config.PHASE2_UNFREEZE_FROM)
                p2_optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=state["lr"], weight_decay=state["weight_decay"],
                )
                p2_optimizer.load_state_dict(state["optimizer_state_dict"])
                p2_early_stop = EarlyStopping(
                    patience=config.EARLY_STOPPING_PATIENCE, mode="max",
                )
                p2_early_stop.load_state_dict(state["early_stop"])

            elif phase == "phase2" and state["phase_done"]:
                logger.info("Resume state indicates training already completed.")
                self._clear_resume_state()
                return self.history

        elif not resume:
            self._clear_resume_state()

        # ---- Phase 1: Head Only ----
        if not skip_phase1:
            freeze_backbone(self.model)
            params = count_parameters(self.model)
            logger.info(
                f"Phase 1 params: {params['trainable']:,} trainable / "
                f"{params['total']:,} total ({params['trainable_pct']:.1f}%)"
            )

            self.fit_phase(
                phase="phase1",
                train_loader=train_loader,
                valid_loader=valid_loader,
                epochs=config.PHASE1_EPOCHS,
                lr=config.PHASE1_LR,
                weight_decay=config.PHASE1_WEIGHT_DECAY,
                warmup_epochs=0,
                _start_epoch=p1_start_epoch,
                _optimizer=p1_optimizer,
                _early_stop=p1_early_stop,
            )
        else:
            logger.info("Phase 1 already completed — skipping.")

        # ---- Phase 2: Fine-Tune ----
        if p2_optimizer is None:
            unfreeze_backbone(self.model, from_block=config.PHASE2_UNFREEZE_FROM)

        params = count_parameters(self.model)
        logger.info(
            f"Phase 2 params: {params['trainable']:,} trainable / "
            f"{params['total']:,} total ({params['trainable_pct']:.1f}%)"
        )

        self.fit_phase(
            phase="phase2",
            train_loader=train_loader,
            valid_loader=valid_loader,
            epochs=config.PHASE2_EPOCHS,
            lr=config.PHASE2_LR,
            weight_decay=config.PHASE2_WEIGHT_DECAY,
            warmup_epochs=config.PHASE2_WARMUP_EPOCHS,
            _start_epoch=p2_start_epoch,
            _optimizer=p2_optimizer,
            _early_stop=p2_early_stop,
        )

        # All done — save final history and clean up resume file
        history_path = os.path.join(config.LOGS_DIR, "training_history.json")
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        with open(history_path, "w") as f:
            json.dump(
                {k: [float(v) for v in vs] for k, vs in self.history.items()},
                f, indent=2,
            )
        logger.info(f"Training history saved to {history_path}")

        self._clear_resume_state()
        return self.history

    def load_best_checkpoint(self, phase: str = "phase2"):
        """Load the best checkpoint for evaluation / inference."""
        path = os.path.join(
            config.CHECKPOINTS_DIR, f"best_model_{phase}.pth",
        )
        if not os.path.exists(path):
            logger.warning(f"No checkpoint found at {path}")
            return
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        logger.info(
            f"Loaded checkpoint: {path} "
            f"(epoch {ckpt['epoch']+1}, QWK={ckpt['metrics']['kappa']:.4f})"
        )
