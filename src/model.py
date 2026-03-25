"""
EfficientNet-B4 model builder with freeze / unfreeze utilities.

Architecture choices targeting high accuracy:
- timm pretrained weights (ImageNet-1k, top-1 ~83%)
- Custom head: pooling -> BN -> dropout -> FC -> BN -> dropout -> FC
- Two-phase training: head-only then backbone fine-tune
- BatchNorm freezing during fine-tuning to preserve ImageNet statistics
"""

import logging

import torch
import torch.nn as nn
import timm

from . import config

logger = logging.getLogger(__name__)


class DRModel(nn.Module):
    """
    EfficientNet-B4 with a custom two-layer classification head.

    The deeper head (pool -> BN -> drop -> FC -> ReLU -> BN -> drop -> FC)
    gives the model more capacity to learn DR-specific representations
    before the final softmax, improving accuracy on fine-grained grades.
    """

    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        num_classes: int = config.NUM_CLASSES,
        dropout_rate: float = config.DROPOUT_RATE,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0,
        )
        in_features = self.backbone.num_features  # 1792 for efficientnet_b4

        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, num_classes),
        )

        self._init_head()

    def _init_head(self):
        """Kaiming initialisation for linear layers in the head."""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature vector before the classification head (for Grad-CAM)."""
        return self.backbone(x)


# -----------------------------------------------------------------------
# Freeze / unfreeze utilities
# -----------------------------------------------------------------------

def freeze_backbone(model: DRModel) -> None:
    """Freeze the entire backbone — only the head is trainable."""
    for param in model.backbone.parameters():
        param.requires_grad = False


def unfreeze_backbone(model: DRModel, from_block: str = config.PHASE2_UNFREEZE_FROM) -> None:
    """
    Unfreeze backbone from ``from_block`` onwards (e.g. "block5").

    Layers before the specified block remain frozen.  All BatchNorm
    layers in the backbone are kept frozen regardless — this is
    critical for transfer-learning stability when batch sizes are
    small relative to the pretrained regime.
    """
    found = False
    for name, param in model.backbone.named_parameters():
        if from_block in name:
            found = True
        param.requires_grad = found

    if not found:
        logger.warning(
            f"Block '{from_block}' not found in backbone parameters. "
            f"All backbone layers remain frozen."
        )

    freeze_batchnorm(model)


def freeze_batchnorm(model: DRModel) -> None:
    """
    Freeze all BatchNorm layers in the backbone.

    When BN layers become trainable with small batches, they update
    running statistics with noisy estimates, destroying the pretrained
    ImageNet statistics and causing training instability.
    """
    for module in model.backbone.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False


def count_parameters(model: nn.Module) -> dict:
    """Count total, trainable, and frozen parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_pct": 100.0 * trainable / max(total, 1),
    }


def build_model(pretrained: bool = True) -> DRModel:
    """Convenience: build a fresh DRModel with default config."""
    return DRModel(pretrained=pretrained)
