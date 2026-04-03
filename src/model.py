"""
EfficientNet-B3 model builder with freeze / unfreeze utilities.

Architecture choices targeting high accuracy on DR grading:
- timm pretrained EfficientNet-B3 weights (ImageNet-1k, top-1 ~83.3%)
- Custom two-layer classification head with BatchNorm and Dropout
- Three-phase training: head-only, partial unfreeze, full unfreeze
- BatchNorm freezing during fine-tuning to preserve ImageNet statistics
"""

import logging

import timm
import torch
import torch.nn as nn

from . import config

logger = logging.getLogger(__name__)


class DRModel(nn.Module):
    """
    EfficientNet-B3 with a custom two-layer classification head.

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
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )
        in_features = self.backbone.num_features  # 1536 for efficientnet_b3

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


def unfreeze_backbone(model: DRModel, from_block: str | None = None) -> None:
    """
    Unfreeze backbone from ``from_block`` onwards.

    If from_block is None, unfreezes all backbone parameters (full unfreeze).
    Supports multiple formats:
    - None — full unfreeze of all backbone layers
    - "block5" — traditional EfficientNet block name (blocks.5 in timm)
    - "features[-2:]" — unfreeze last 2 feature blocks (e.g., for 8-block model, unfreeze blocks 6+7)
    - Direct layer name from named_parameters()

    All BatchNorm layers in the backbone are kept frozen regardless — this is
    critical for transfer-learning stability when batch sizes are small
    relative to the pretrained regime.
    """
    if from_block is None:
        # Full unfreeze: unfreeze all backbone parameters
        for param in model.backbone.parameters():
            param.requires_grad = True
        logger.info("Full backbone unfreeze: all backbone parameters now trainable.")
        freeze_batchnorm(model)
        return

    found = False
    candidate_prefixes: list[str] = []

    # Parse "features[-2:]" syntax to unfreeze last N blocks
    if isinstance(from_block, str) and "features[" in from_block and "]" in from_block:
        try:
            # Extract slice notation (e.g., "-2:" from "features[-2:]")
            import re

            match = re.search(r"features\[(.*?)\]", from_block)
            if match:
                slice_str = match.group(1)
                # Get all feature blocks and apply slice
                all_params = list(model.backbone.named_parameters())
                # Find feature block indices
                feature_indices = set()
                for name, _ in all_params:
                    feat_match = re.search(r"features\.(\d+)", name)
                    if feat_match:
                        feature_indices.add(int(feat_match.group(1)))

                if feature_indices:
                    # Convert to sorted list and apply slice
                    sorted_indices = sorted(feature_indices)
                    parts = slice_str.split(",")
                    args = [int(p.strip()) if p.strip() else None for p in parts]
                    slice_obj = slice(*args)
                    unfrozen_indices = sorted_indices[slice_obj]

                    # Build candidate prefixes for unfrozen blocks
                    for idx in unfrozen_indices:
                        candidate_prefixes.append(f"features.{idx}")

                    logger.info(f"Parsed '{from_block}' → unfrozen blocks: {unfrozen_indices}")
        except Exception as e:
            logger.warning(
                f"Failed to parse '{from_block}': {e}. Falling back to literal string match."
            )
            candidate_prefixes.append(from_block)
    else:
        # Traditional block name parsing (e.g., "block5" → "blocks.5")
        normalized = str(from_block).strip().lower().replace("_", "")
        if normalized.startswith("block") and normalized[5:].isdigit():
            idx = int(normalized[5:])
            candidate_prefixes.append(f"blocks.{idx}")
        candidate_prefixes.extend(
            [
                from_block,
                from_block.replace("_", "."),
            ]
        )

    for name, param in model.backbone.named_parameters():
        if not found and any(
            name.startswith(prefix) or prefix in name for prefix in candidate_prefixes
        ):
            found = True
        param.requires_grad = found

    if not found:
        logger.warning(
            f"Block '{from_block}' not found in backbone parameters. "
            f"Tried prefixes: {candidate_prefixes}. All backbone layers remain frozen."
        )
    else:
        logger.info(
            f"Backbone unfreezing started from '{from_block}' "
            f"(unfrozen prefixes: {candidate_prefixes})."
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
