"""
Loss functions for Diabetic Retinopathy classification.

All losses support a ``reduction`` parameter:
  - "mean" (default): returns scalar — standard for training
  - "none": returns per-sample (B,) tensor — needed for Mixup

Provides:
  - FocalLoss: down-weights easy examples, focuses on hard borderline cases
  - OrdinalCrossEntropy: ordinal-aware soft targets aligned with QWK metric
  - LabelSmoothingCrossEntropy: standard CE with label smoothing
  - CombinedLoss: weighted sum of OrdinalCE + FocalLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    Down-weights well-classified examples (e.g. the overwhelming No_DR class)
    and focuses learning on hard borderline cases (No_DR↔Mild confusion).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Parameters
    ----------
    gamma : float
        Focusing parameter.  gamma=0 recovers standard CE.
        gamma=2 is the standard choice (heavily down-weights easy examples).
    alpha : Tensor or None
        Per-class weights.  If None, all classes weighted equally.
    label_smoothing : float
        Standard label smoothing applied before focal weighting.
    reduction : str
        "mean" returns scalar, "none" returns per-sample tensor.
    """

    def __init__(
        self,
        gamma: float = config.FOCAL_GAMMA,
        alpha: torch.Tensor | None = None,
        label_smoothing: float = config.LABEL_SMOOTHING,
        num_classes: int = config.NUM_CLASSES,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            pred,
            target,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )

        pt = torch.exp(-ce_loss)
        focal_weight = (1.0 - pt) ** self.gamma
        loss = focal_weight * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.to(pred.device)[target]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        return loss


class OrdinalCrossEntropy(nn.Module):
    """
    Cross-entropy with ordinal-aware soft targets.

    Instead of a hard one-hot target, constructs a soft distribution where
    the true class gets most mass but neighbouring classes receive some mass
    weighted by 1 / (1 + distance²).  This naturally penalises far-off
    predictions more than near misses, aligning the loss with the QWK metric.

    Parameters
    ----------
    num_classes : int
        Number of ordinal classes.
    smoothing : float
        Total probability mass spread to non-true classes (distributed
        according to ordinal distance).
    weight : torch.Tensor or None
        Per-class weights for rebalancing.
    reduction : str
        "mean" returns scalar, "none" returns per-sample tensor.
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        smoothing: float = config.LABEL_SMOOTHING,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

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
        soft = self.soft_targets.to(pred.device)[target]  # (B, C)

        per_sample_loss = -(soft * log_probs).sum(dim=1)  # (B,)

        if self.weight is not None:
            w = self.weight.to(pred.device)
            sample_w = w[target]
            per_sample_loss = per_sample_loss * sample_w

        if self.reduction == "mean":
            return per_sample_loss.mean()
        return per_sample_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy with optional label smoothing and class weights.
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        smoothing: float = config.LABEL_SMOOTHING,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.weight = weight
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(pred.device) if self.weight is not None else None
        return F.cross_entropy(
            pred,
            target,
            weight=weight,
            label_smoothing=self.smoothing,
            reduction=self.reduction,
        )


class CombinedLoss(nn.Module):
    """
    Weighted combination of OrdinalCE and FocalLoss.

    The ordinal component aligns gradients with the QWK metric
    (near-misses penalised less than far-off misses).
    The focal component down-weights easy examples and focuses
    learning on hard borderline cases.

    Parameters
    ----------
    ordinal_weight : float
        Weight for the ordinal CE loss term.
    focal_weight : float
        Weight for the focal loss term.
    reduction : str
        "mean" returns scalar, "none" returns per-sample tensor.
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        smoothing: float = config.LABEL_SMOOTHING,
        focal_gamma: float = config.FOCAL_GAMMA,
        class_weights: torch.Tensor | None = None,
        ordinal_weight: float = 0.5,
        focal_weight: float = 0.5,
        reduction: str = "mean",
    ):
        super().__init__()
        self.ordinal_weight = ordinal_weight
        self.focal_weight = focal_weight
        self.reduction = reduction

        # Sub-losses always return per-sample tensors internally
        self.ordinal_loss = OrdinalCrossEntropy(
            num_classes=num_classes,
            smoothing=smoothing,
            weight=class_weights,
            reduction="none",
        )
        self.focal_loss = FocalLoss(
            gamma=focal_gamma,
            alpha=class_weights,
            label_smoothing=smoothing,
            num_classes=num_classes,
            reduction="none",
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_ord = self.ordinal_loss(pred, target)
        loss_focal = self.focal_loss(pred, target)
        combined = self.ordinal_weight * loss_ord + self.focal_weight * loss_focal

        if self.reduction == "mean":
            return combined.mean()
        return combined
