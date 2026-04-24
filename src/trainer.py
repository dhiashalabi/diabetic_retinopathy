"""
Three-phase training engine with AMP, ordinal-aware loss, Mixup,
early stopping on QWK, cosine LR with warmup, and gradient clipping.

Phase 1 — head-only with light augmentation (high LR, no warmup)
Phase 2 — partial unfreeze with full augmentation + Mixup (low LR, warmup)
Phase 3 — full unfreeze with full augmentation (Mixup optional via config)

Checkpointing on Quadratic Weighted Kappa (the clinical gold-standard metric).
"""

import contextlib
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config
from .augmentations import consistency_augment, mixup_data, mixup_data_class_conditional
from .losses import CombinedLoss, FocalLoss, LabelSmoothingCrossEntropy, OrdinalCrossEntropy
from .utils import get_logger

logger = get_logger(__name__)


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
    - Combined Ordinal + Focal loss
    - Optional Mixup augmentation (Phase 2/3)
    - Optional EMA (Exponential Moving Average)
    - Checkpointing on QWK (the clinical gold-standard metric)
    - Cosine LR with warmup
    - Discriminative learning rates (head vs backbone)
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        class_weights: torch.Tensor | None = None,
        ema=None,
    ):
        self.model = model.to(device)
        self.device = device
        self.ema = ema

        # Class weights for loss function
        if config.USE_CLASS_WEIGHTED_LOSS and class_weights is not None:
            self.class_weights = class_weights
        else:
            self.class_weights = None

        self._amp_dtype, self._use_grad_scaler = config.resolve_amp(self.device)
        self.scaler = GradScaler(
            device.type,
            enabled=device.type == "cuda" and self._use_grad_scaler,
        )
        if self.device.type == "cuda" and self._amp_dtype is not None:
            logger.info(
                "AMP dtype=%s grad_scaler=%s",
                self._amp_dtype,
                self._use_grad_scaler,
            )
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

    def _autocast(self, enabled: bool):
        """CUDA autocast with config AMP dtype; null context when disabled or CPU."""
        if not enabled or self.device.type != "cuda" or self._amp_dtype is None:
            return contextlib.nullcontext()
        return autocast("cuda", dtype=self._amp_dtype, enabled=True)

    def _build_criterion(self, reduction: str = "mean") -> nn.Module:
        """Build the loss function based on config flags."""
        if config.USE_ORDINAL_LOSS and config.USE_FOCAL_LOSS:
            return CombinedLoss(
                class_weights=self.class_weights,
                smoothing=config.LABEL_SMOOTHING,
                focal_gamma=config.FOCAL_GAMMA,
                reduction=reduction,
            )
        elif config.USE_ORDINAL_LOSS:
            return OrdinalCrossEntropy(
                weight=self.class_weights,
                smoothing=config.LABEL_SMOOTHING,
                reduction=reduction,
            )
        elif config.USE_FOCAL_LOSS:
            return FocalLoss(
                alpha=self.class_weights,
                label_smoothing=config.LABEL_SMOOTHING,
                gamma=config.FOCAL_GAMMA,
                reduction=reduction,
            )
        return LabelSmoothingCrossEntropy(
            weight=self.class_weights,
            smoothing=config.LABEL_SMOOTHING,
            reduction=reduction,
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
        criterion_unreduced: nn.Module,
        use_mixup: bool = False,
        use_amp: bool = True,
    ) -> dict:
        self.model.train()
        self._freeze_backbone_bn()

        amp_on = bool(use_amp and self.device.type == "cuda")

        running_loss = 0.0
        correct = 0
        total = 0
        all_preds: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        pbar = tqdm(loader, desc="  Train", leave=False)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # --- Consistency regularization (without doubling batch) ---
            apply_consistency = config.USE_CONSISTENCY_REG and not use_mixup
            consistency_loss = torch.tensor(0.0, device=self.device)

            # --- Mixup ---
            if use_mixup and config.USE_MIXUP:
                if config.USE_CLASS_CONDITIONAL_MIXUP:
                    images, labels_a, labels_b, lam = mixup_data_class_conditional(images, labels)
                else:
                    images, labels_a, labels_b, lam = mixup_data(images, labels)
                use_mixup_this_batch = True
            else:
                labels_a = labels_b = labels
                lam = 1.0
                use_mixup_this_batch = False

            optimizer.zero_grad(set_to_none=True)

            with self._autocast(amp_on):
                outputs = self.model(images)

                # --- Loss computation (clean, no branching) ---
                if use_mixup_this_batch:
                    # Mixup: weighted sum of losses with per-sample reduction
                    loss_a = criterion_unreduced(outputs, labels_a)
                    loss_b = criterion_unreduced(outputs, labels_b)
                    if isinstance(lam, torch.Tensor):
                        loss = (lam * loss_a + (1 - lam) * loss_b).mean()
                    else:
                        loss = (lam * loss_a + (1 - lam) * loss_b).mean()
                else:
                    loss = criterion(outputs, labels_a)

                # --- Consistency regularization (memory-efficient) ---
                if apply_consistency:
                    with torch.no_grad():
                        prob_clean = F.softmax(outputs.detach(), dim=1)
                    aug_images = consistency_augment(images)
                    out_aug = self.model(aug_images)
                    log_prob_aug = F.log_softmax(out_aug, dim=1)
                    consistency_loss = F.kl_div(log_prob_aug, prob_clean, reduction="batchmean")
                    loss = loss + config.CONSISTENCY_WEIGHT * consistency_loss

            if not torch.isfinite(loss).all():
                logger.warning(
                    "Non-finite training loss; skipping batch"
                    + (" (GradScaler updated)." if self._use_grad_scaler and amp_on else ".")
                )
                if self._use_grad_scaler and amp_on:
                    self.scaler.update()
                continue

            if self._use_grad_scaler and amp_on:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    config.GRADIENT_CLIP_NORM,
                )
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    config.GRADIENT_CLIP_NORM,
                )
                optimizer.step()

            # --- EMA update ---
            if self.ema is not None:
                self.ema.update(self.model)

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
    def validate(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        use_amp: bool = True,
    ) -> dict:
        # Use EMA model for validation if available
        eval_model = self.ema.shadow if self.ema is not None else self.model
        eval_model.eval()

        amp_on = bool(use_amp and self.device.type == "cuda")

        running_loss = 0.0
        correct = 0
        total = 0
        all_preds: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        pbar = tqdm(loader, desc="  Valid", leave=False)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with self._autocast(amp_on):
                outputs = eval_model(images)
                loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            if not torch.isfinite(loss).all():
                logger.warning("Non-finite validation loss; skipping batch in metrics.")
                continue
            running_loss += loss.item() * batch_size
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size
            all_preds.append(preds.detach())
            all_labels.append(labels.detach())

        if total == 0:
            return {"loss": float("nan"), "acc": 0.0, "kappa": 0.0}

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
        if scaler is not None and self._use_grad_scaler:
            ckpt["scaler_state_dict"] = scaler.state_dict()
        if self.ema is not None:
            ckpt["ema_state_dict"] = self.ema.state_dict()
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
        use_amp: bool = True,
    ) -> bool:
        """
        Run one training phase.

        Returns True if the phase completed normally, False if early-stopped.
        """
        self.best_kappa = -1.0

        # Build two versions of the criterion: reduced (scalar) and unreduced (per-sample)
        criterion = self._build_criterion(reduction="mean").to(self.device)
        criterion_unreduced = self._build_criterion(reduction="none").to(self.device)

        # Discriminative learning rates
        if config.USE_DISCRIMINATIVE_LR and phase != "phase1" and phase != "s1_phase1":
            head_params = list(self.model.head.parameters()) + list(
                self.model.global_pool.parameters()
            )
            backbone_params = [p for p in self.model.backbone.parameters() if p.requires_grad]

            optimizer = torch.optim.AdamW(
                [
                    {"params": backbone_params, "lr": lr},
                    {"params": head_params, "lr": lr * config.HEAD_LR_MULTIPLIER},
                ],
                weight_decay=weight_decay,
            )
        else:
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
            f"Phase: {phase.upper()} | Epochs: {epochs} | LR: {lr} | "
            f"Mixup: {use_mixup} | AMP: {use_amp and self.device.type == 'cuda'} "
            f"(dtype={self._amp_dtype}, scaler={self._use_grad_scaler})"
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
                criterion_unreduced,
                use_mixup=use_mixup,
                use_amp=use_amp,
            )
            val_metrics = self.validate(valid_loader, criterion, use_amp=use_amp)

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
                    scaler=self.scaler if self._use_grad_scaler else None,
                )

            if early_stop(val_metrics["kappa"]):
                logger.info(
                    f"Early stopping at epoch {epoch + 1} (best QWK: {early_stop.best_score:.4f})"
                )
                early_stopped = True
                break

        return not early_stopped

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
        if self.ema is not None and "ema_state_dict" in ckpt:
            self.ema.load_state_dict(ckpt["ema_state_dict"])
        logger.info(
            f"Loaded checkpoint: {path} "
            f"(epoch {ckpt['epoch'] + 1}, QWK={ckpt['metrics']['kappa']:.4f})"
        )

    def save_history(self, path: str | None = None):
        """Save training history to JSON."""
        if path is None:
            path = os.path.join(config.LOGS_DIR, "training_history.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training history saved to {path}")
