"""
EfficientNet-B3 model builder with freeze / unfreeze utilities.

Architecture choices targeting high accuracy on DR grading:
- timm pretrained EfficientNet-B3 weights (ImageNet-1k, top-1 ~83.3%)
- Custom two-layer classification head with BatchNorm and Dropout
- Three-phase training: head-only, partial unfreeze, full unfreeze
- BatchNorm freezing during fine-tuning to preserve ImageNet statistics
"""

import logging
import re

import timm
import torch
import torch.nn as nn

from . import config

logger = logging.getLogger(__name__)


def _parse_backbone_block_slice(slice_str: str) -> slice:
    """
    Parse slice notation inside ``blocks[...]`` or ``features[...]``.

    Supports comma-separated (``-2, None``) and colon forms (``-2:``, ``:3``, ``3:5``).
    """
    s = slice_str.strip()
    if not s:
        return slice(None)
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        args = [int(p) if p else None for p in parts]
        return slice(*args)
    if ":" in s:
        start_s, _, stop_s = s.partition(":")
        start = int(start_s) if start_s.strip() else None
        stop = int(stop_s) if stop_s.strip() else None
        return slice(start, stop)
    return slice(int(s))


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
    - "blocks[-2:]" — unfreeze last N block indices (timm EfficientNet uses ``blocks.N``)
    - "features[-2:]" — same for ResNet-style backbones that use ``features.N``
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

    # Parse "blocks[-2:]" or "features[-2:]" to unfreeze last N backbone stages
    block_slice_match = (
        re.search(r"(blocks|features)\[(.*?)\]", from_block)
        if isinstance(from_block, str)
        else None
    )
    if block_slice_match:
        try:
            block_prefix = block_slice_match.group(1)
            slice_inner = block_slice_match.group(2)
            block_indices: set[int] = set()
            pat = re.compile(rf"{re.escape(block_prefix)}\.(\d+)")
            for name, _ in model.backbone.named_parameters():
                bm = pat.search(name)
                if bm:
                    block_indices.add(int(bm.group(1)))

            if block_indices:
                sorted_indices = sorted(block_indices)
                slice_obj = _parse_backbone_block_slice(slice_inner)
                unfrozen = sorted_indices[slice_obj]
                unfrozen_list = [unfrozen] if isinstance(unfrozen, int) else list(unfrozen)
                for idx in unfrozen_list:
                    candidate_prefixes.append(f"{block_prefix}.{idx}")
                logger.info(
                    f"Parsed '{from_block}' → unfrozen {block_prefix} indices: {unfrozen_list}"
                )
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
