"""
EfficientNetV2-S model builder with GeM pooling, improved head,
and freeze / unfreeze utilities.

Architecture choices targeting high accuracy on DR grading:
- timm pretrained EfficientNetV2-S weights (ImageNet-21K → 1K fine-tuned)
- Generalised Mean (GeM) pooling — adapts between avg/max, better for fine-grained tasks
- Custom two-layer classification head with BatchNorm, SiLU, and Dropout
- Stochastic depth (drop-path) for regularisation
- Three-phase training: head-only, partial unfreeze, full unfreeze
- BatchNorm freezing during fine-tuning to preserve pretrained statistics
- Exponential Moving Average (EMA) for stable fine-tuning
"""

import copy
import logging
import re

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config

logger = logging.getLogger(__name__)


class GeM(nn.Module):
    """
    Generalised Mean Pooling.

    Learns a pooling parameter p that interpolates between average pooling
    (p=1) and max pooling (p→∞).  Proven more effective than standard
    average pooling for fine-grained classification tasks.

    Reference: Radenović et al., "Fine-tuning CNN Image Retrieval with
    No Human Annotation", TPAMI 2019.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(min=self.eps).pow(self.p).mean(dim=[-2, -1]).pow(1.0 / self.p)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p.item():.2f})"


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
    EfficientNetV2-S with GeM pooling and a custom two-layer classification head.

    The deeper head (GeM → BN → drop → FC → SiLU → BN → drop → FC)
    gives the model more capacity to learn DR-specific representations
    before the final softmax, improving accuracy on fine-grained grades.
    """

    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        num_classes: int = config.NUM_CLASSES,
        dropout_rate: float = config.DROPOUT_RATE,
        pretrained: bool = True,
        drop_path_rate: float = config.STOCHASTIC_DEPTH_RATE,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",  # disable default pooling — we use GeM
            drop_path_rate=drop_path_rate,
        )
        in_features = self.backbone.num_features

        # Generalised Mean Pooling replaces default average pooling
        self.global_pool = GeM(p=3.0)

        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 768),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(768),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(768, num_classes),
        )

        self._init_head()

    def _init_head(self):
        """Proper initialisation for the classification head."""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                # Use trunc_normal_ which works well with any activation
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        pooled = self.global_pool(features)
        return self.head(pooled)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature vector before the classification head (for Grad-CAM)."""
        features = self.backbone(x)
        return self.global_pool(features)


# -----------------------------------------------------------------------
# Exponential Moving Average
# -----------------------------------------------------------------------


class ModelEMA:
    """
    Exponential Moving Average of model weights.

    Maintains a shadow copy of the model whose weights are an exponentially
    weighted average of the training weights.  The EMA model typically
    generalises better than the raw trained model.

    Usage:
        ema = ModelEMA(model, decay=0.9999)
        # During training:
        ema.update(model)
        # During evaluation:
        ema.apply(model)  # swap weights for inference
    """

    def __init__(self, model: nn.Module, decay: float = config.EMA_DECAY):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for param in self.shadow.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update EMA weights with current model weights."""
        for ema_p, model_p in zip(self.shadow.parameters(), model.parameters(), strict=True):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)

    def apply(self, model: nn.Module):
        """Copy EMA weights into the model (for evaluation/export)."""
        model.load_state_dict(self.shadow.state_dict())

    def state_dict(self) -> dict:
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict: dict):
        self.shadow.load_state_dict(state_dict)


# -----------------------------------------------------------------------
# Freeze / unfreeze utilities
# -----------------------------------------------------------------------


def freeze_backbone(model: DRModel) -> None:
    """Freeze the entire backbone — only the head and GeM pool are trainable."""
    for param in model.backbone.parameters():
        param.requires_grad = False


def unfreeze_backbone(model: DRModel, from_block: str | None = None) -> None:
    """
    Unfreeze backbone from ``from_block`` onwards.

    If from_block is None, unfreezes all backbone parameters (full unfreeze).
    Supports multiple formats:
    - None — full unfreeze of all backbone layers
    - "block5" — traditional EfficientNet block name (blocks.5 in timm)
    - "blocks[-1:]" — unfreeze last N block indices (timm uses ``blocks.N``)
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

    # Parse "blocks[-1:]" or "features[-2:]" to unfreeze last N backbone stages
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

            # Also try common EfficientNetV2 internal naming patterns
            alt_prefixes = [block_prefix]
            if block_prefix == "blocks":
                alt_prefixes.extend(["stages", "features"])

            for prefix in alt_prefixes:
                idx_pat = re.compile(rf"{re.escape(prefix)}\.(\d+)")
                for name, _ in model.backbone.named_parameters():
                    bm = idx_pat.search(name)
                    if bm:
                        block_indices.add(int(bm.group(1)))

            if block_indices:
                sorted_indices = sorted(block_indices)
                slice_obj = _parse_backbone_block_slice(slice_inner)
                unfrozen = sorted_indices[slice_obj]
                unfrozen_list = [unfrozen] if isinstance(unfrozen, int) else list(unfrozen)
                for prefix in alt_prefixes:
                    for idx in unfrozen_list:
                        candidate_prefixes.append(f"{prefix}.{idx}")
                logger.info(f"Parsed '{from_block}' → unfrozen indices: {unfrozen_list}")
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
    statistics and causing training instability.
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


def build_model(
    model_name: str = config.MODEL_NAME,
    num_classes: int = config.NUM_CLASSES,
    pretrained: bool = True,
    drop_path_rate: float = config.STOCHASTIC_DEPTH_RATE,
    dropout_rate: float = config.DROPOUT_RATE,
) -> DRModel:
    """Convenience: build a DRModel with configurable parameters."""
    return DRModel(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        drop_path_rate=drop_path_rate,
        dropout_rate=dropout_rate,
    )
