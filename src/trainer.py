"""
Three-phase training engine with AMP, ordinal-aware loss, Mixup,
early stopping on QWK, cosine LR with warmup, and gradient clipping.

Phase 1 — head-only with light augmentation (high LR, no warmup)
Phase 2 — partial unfreeze with full augmentation + Mixup (low LR, warmup)
Phase 3 — full unfreeze with full augmentation + Mixup (very low LR, warmup)

Checkpointing on Quadratic Weighted Kappa (the clinical gold-standard metric).
"""

import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config
from .augmentations import mixup_data
from .utils import get_logger

logger = get_logger(__name__)


# -----------------------------------------------------------------------
# Loss functions
# -----------------------------------------------------------------------


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy with optional label smoothing and class weights.
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        smoothing: float = config.LABEL_SMOOTHING,
        weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(pred.device) if self.weight is not None else None
        return F.cross_entropy(
            pred,
            target,
            weight=weight,
            label_smoothing=self.smoothing,
        )


class OrdinalCrossEntropy(nn.Module):
    """
    Cross-entropy with ordinal-aware soft targets.

    Instead of a hard one-hot target, constructs a soft distribution where
    the true class gets most mass but neighbouring classes receive some mass
    weighted by 1 / (1 + distance^2).  This naturally penalises far-off
    predictions more than near misses, aligning the loss with the QWK metric.

    Parameters
    ----------
    num_classes : int
        Number of ordinal classes.
    smoothing : float
        Total probability mass spread to non-true classes (similar to label
        smoothing but distributed according to ordinal distance).
    weight : torch.Tensor or None
        Per-class weights for rebalancing (only used when
        config.USE_CLASS_WEIGHTED_LOSS is True).
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        smoothing: float = config.LABEL_SMOOTHING,
        weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.weight = weight

        soft_targets = self._build_soft_targets(num_classes, smoothing)
        self.register_buffer("soft_targets", soft_targets)

    @staticmethod
    def _build_soft_targets(num_classes: int, smoothing: float) -> torch.Tensor:
        """
        Build an (num_classes, num_classes) matrix of ordinal soft targets.

        Row k is the target distribution when the true class is k.
        """
        targets = torch.zeros(num_classes, num_classes)
        for k in range(num_classes):
            raw_weights = torch.zeros(num_classes)
            for j in range(num_classes):
                if j == k:
                    continue
                dist = abs(k - j)
                raw_weights[j] = 1.0 / (1.0 + dist * dist)

            if raw_weights.sum() > 0:
                raw_weights = raw_weights / raw_weights.sum() * smoothing

            targets[k] = raw_weights
            targets[k, k] = 1.0 - smoothing

        return targets

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(pred, dim=1)
        soft = self.soft_targets[target]  # (B, C)

        per_sample_loss = -(soft * log_probs).sum(dim=1)  # (B,)

        if self.weight is not None:
            w = self.weight.to(pred.device)
            sample_w = w[target]
            per_sample_loss = per_sample_loss * sample_w

        return per_sample_loss.mean()


# -----------------------------------------------------------------------
# Learning rate scheduler
# -----------------------------------------------------------------------


class CosineWarmupScheduler:
    """Linear warmup followed by cosine annealing."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, min_lr: float = 1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            factor = (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            pg["lr"] = max(base_lr * factor, self.min_lr)

    def get_last_lr(self) -> list[float]:
        return [pg["lr"] for pg in self.optimizer.param_groups]


# -----------------------------------------------------------------------
# Early stopping
# -----------------------------------------------------------------------


class EarlyStopping:
    """Stop training when the monitored metric stops improving."""

    def __init__(
        self,
        patience: int = config.EARLY_STOPPING_PATIENCE,
        mode: str = "max",
        min_delta: float = 1e-4,
    ):
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


# -----------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------


class Trainer:
    """
    Three-phase training engine for the DR classifier.

    Features:
    - Mixed-precision training (AMP)
    - Gradient clipping for stable fine-tuning
    - Ordinal-aware or label-smoothed cross-entropy loss
    - Optional Mixup augmentation (Phase 2/3)
    - Checkpointing on QWK (the clinical gold-standard metric)
    - Cosine LR with warmup
    - Phase-aware data loaders (light augmentation for Phase 1)
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        class_weights: torch.Tensor | None = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.class_weights = class_weights if config.USE_CLASS_WEIGHTED_LOSS else None
        self.scaler = GradScaler(device.type, enabled=device.type == "cuda")
        self.history: dict[str, list] = {
            "train_loss": [],
            "train_acc": [],
            "train_kappa": [],
            "val_loss": [],
            "val_acc": [],
            "val_kappa": [],
            "lr": [],
        }
        self.best_kappa = -1.0

    def _build_criterion(self) -> nn.Module:
        if config.USE_ORDINAL_LOSS:
            return OrdinalCrossEntropy(
                weight=self.class_weights,
                smoothing=config.LABEL_SMOOTHING,
            )
        return LabelSmoothingCrossEntropy(
            weight=self.class_weights,
            smoothing=config.LABEL_SMOOTHING,
        )

    def _freeze_backbone_bn(self):
        """Keep BatchNorm layers in eval mode during fine-tuning."""
        from .model import DRModel

        if isinstance(self.model, DRModel):
            for module in self.model.backbone.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    module.eval()

    def train_one_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        use_mixup: bool = False,
    ) -> dict:
        self.model.train()
        self._freeze_backbone_bn()

        running_loss = 0.0
        correct = 0
        total = 0
        all_preds: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        pbar = tqdm(loader, desc="  Train", leave=False)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if use_mixup and config.USE_MIXUP:
                images, labels_a, labels_b, lam = mixup_data(images, labels)
            else:
                labels_a = labels_b = labels
                lam = 1.0

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=self.device.type == "cuda"):
                outputs = self.model(images)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                config.GRADIENT_CLIP_NORM,
            )
            self.scaler.step(optimizer)
            self.scaler.update()

            batch_size = labels_a.size(0)
            running_loss += loss.item() * batch_size
            preds = outputs.argmax(dim=1)
            correct += (preds == labels_a).sum().item()
            total += batch_size
            all_preds.append(preds.detach())
            all_labels.append(labels_a.detach())

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100 * correct / total:.1f}%",
            )

        if total == 0:
            return {"loss": 0.0, "acc": 0.0, "kappa": 0.0}

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        y_pred = torch.cat(all_preds).cpu().numpy()
        y_true = torch.cat(all_labels).cpu().numpy()
        epoch_kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")

        return {"loss": epoch_loss, "acc": epoch_acc, "kappa": epoch_kappa}

    @torch.no_grad()
    def validate(self, loader: DataLoader, criterion: nn.Module) -> dict:
        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0
        all_preds: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

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
            all_preds.append(preds.detach())
            all_labels.append(labels.detach())

        if total == 0:
            return {"loss": 0.0, "acc": 0.0, "kappa": 0.0}

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        y_pred = torch.cat(all_preds).cpu().numpy()
        y_true = torch.cat(all_labels).cpu().numpy()
        epoch_kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")

        return {"loss": epoch_loss, "acc": epoch_acc, "kappa": epoch_kappa}

    def _save_checkpoint(self, epoch: int, phase: str, metrics: dict, optimizer=None, scaler=None):
        path = os.path.join(
            config.CHECKPOINTS_DIR,
            f"best_model_{phase}.pth",
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

    def fit_phase(
        self,
        phase: str,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        epochs: int,
        lr: float,
        weight_decay: float,
        warmup_epochs: int = 0,
        use_mixup: bool = False,
    ) -> bool:
        """
        Run one training phase.

        Returns True if the phase completed normally, False if early-stopped.
        """
        criterion = self._build_criterion()

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=epochs,
        )
        early_stop = EarlyStopping(
            patience=config.EARLY_STOPPING_PATIENCE,
            mode="max",
        )

        logger.info(f"\n{'=' * 60}")
        logger.info(
            f"Phase: {phase.upper()} | Epochs: {epochs} | LR: {lr} | Mixup: {use_mixup}"
        )
        logger.info(f"{'=' * 60}")

        early_stopped = False

        for epoch in range(epochs):
            t0 = time.time()

            scheduler.step(epoch)
            current_lr = optimizer.param_groups[0]["lr"]

            train_metrics = self.train_one_epoch(
                train_loader,
                optimizer,
                criterion,
                use_mixup=use_mixup,
            )
            val_metrics = self.validate(valid_loader, criterion)

            for split, metrics in [("train", train_metrics), ("val", val_metrics)]:
                self.history[f"{split}_loss"].append(metrics["loss"])
                self.history[f"{split}_acc"].append(metrics["acc"])
                self.history[f"{split}_kappa"].append(metrics["kappa"])
            self.history["lr"].append(current_lr)

            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch + 1:02d}/{epochs} ({elapsed:.0f}s) | "
                f"LR: {current_lr:.2e} | "
                f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['acc']:.4f} "
                f"QWK: {train_metrics['kappa']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['acc']:.4f} "
                f"QWK: {val_metrics['kappa']:.4f}"
            )

            if val_metrics["kappa"] > self.best_kappa:
                self.best_kappa = val_metrics["kappa"]
                self._save_checkpoint(
                    epoch,
                    phase,
                    val_metrics,
                    optimizer=optimizer,
                    scaler=self.scaler,
                )

            if early_stop(val_metrics["kappa"]):
                logger.info(
                    f"Early stopping at epoch {epoch + 1} (best QWK: {early_stop.best_score:.4f})"
                )
                early_stopped = True
                break

        return not early_stopped

    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        train_loader_light: DataLoader | None = None,
    ) -> dict:
        """
        Full three-phase training pipeline.

        Parameters
        ----------
        train_loader : DataLoader
            Training loader with full (heavy) augmentation for Phase 2/3.
        valid_loader : DataLoader
            Validation loader (deterministic transforms).
        train_loader_light : DataLoader or None
            Training loader with light augmentation for Phase 1 (head-only).
            If None, ``train_loader`` is used for all phases.
        """
        from .model import count_parameters, freeze_backbone, unfreeze_backbone

        phase1_loader = train_loader_light if train_loader_light is not None else train_loader

        # ---- Phase 1: Head Only (light augmentation, no Mixup) ----
        freeze_backbone(self.model)
        params = count_parameters(self.model)
        phase1_trainable = params["trainable"]
        logger.info(
            f"Phase 1 params: {params['trainable']:,} trainable / "
            f"{params['total']:,} total ({params['trainable_pct']:.1f}%)"
        )

        self.fit_phase(
            phase="phase1",
            train_loader=phase1_loader,
            valid_loader=valid_loader,
            epochs=config.PHASE1_EPOCHS,
            lr=config.PHASE1_LR,
            weight_decay=config.PHASE1_WEIGHT_DECAY,
            warmup_epochs=0,
            use_mixup=False,
        )

        # ---- Phase 2: Partial Unfreeze (full augmentation + Mixup) ----
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        unfreeze_backbone(self.model, from_block=config.PHASE2_UNFREEZE_FROM)

        params = count_parameters(self.model)
        if params["trainable"] <= phase1_trainable:
            raise RuntimeError(
                "Phase 2 fine-tuning did not increase trainable parameters. "
                "Check PHASE2_UNFREEZE_FROM and backbone parameter names."
            )
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
            use_mixup=True,
        )

        # ---- Phase 3: Full Unfreeze (full augmentation + Mixup) ----
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        unfreeze_backbone(self.model, from_block=None)

        params = count_parameters(self.model)
        logger.info(
            f"Phase 3 params: {params['trainable']:,} trainable / "
            f"{params['total']:,} total ({params['trainable_pct']:.1f}%)"
        )

        self.fit_phase(
            phase="phase3",
            train_loader=train_loader,
            valid_loader=valid_loader,
            epochs=config.PHASE3_EPOCHS,
            lr=config.PHASE3_LR,
            weight_decay=config.PHASE3_WEIGHT_DECAY,
            warmup_epochs=config.PHASE3_WARMUP_EPOCHS,
            use_mixup=True,
        )

        # Save final history
        history_path = os.path.join(config.LOGS_DIR, "training_history.json")

        def _sanitize(v):
            f = float(v)
            return 0.0 if (f != f) else f

        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        with open(history_path, "w") as f:
            json.dump(
                {k: [_sanitize(v) for v in vs] for k, vs in self.history.items()},
                f,
                indent=2,
            )
        logger.info(f"Training history saved to {history_path}")

        return self.history

    def load_best_checkpoint(self, phase: str = "phase3"):
        """Load the best checkpoint for evaluation / inference."""
        path = os.path.join(
            config.CHECKPOINTS_DIR,
            f"best_model_{phase}.pth",
        )
        if not os.path.exists(path):
            logger.warning(f"No checkpoint found at {path}")
            return
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        logger.info(
            f"Loaded checkpoint: {path} "
            f"(epoch {ckpt['epoch'] + 1}, QWK={ckpt['metrics']['kappa']:.4f})"
        )
