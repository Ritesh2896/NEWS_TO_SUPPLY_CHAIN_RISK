"""Centralized Configuration for Graph Neural Networks in BDS-35 (Phase 8 & 9).

Supports model selection between:
  - model = "graphsage" (Hamilton et al., 2017)
  - model = "gat" (Veličković et al., 2018)

Maintains shared hyperparameters, training schedules, and academic disclaimers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class GNNModelType(str, Enum):
    """Supported GNN propagation architectures."""
    GRAPHSAGE = "graphsage"
    GAT = "gat"


# Academic disclaimer mandated by methodology
ATTENTION_CAUSALITY_DISCLAIMER: str = (
    "Attention weights reflect relative feature aggregation importance in localized "
    "graph neighborhoods. Attention weights do NOT prove or establish causal relationships."
)


@dataclass
class CentralizedGNNConfig:
    """Centralized configuration for GraphSAGE and GAT architectures and training."""
    model_type: str = GNNModelType.GRAPHSAGE.value
    in_dim: int = 16
    hidden_dim: int = 32
    out_dim: int = 1
    # GAT-specific parameters
    num_heads: int = 2
    gat_concat: bool = True
    return_attention_weights: bool = True
    # Shared training parameters
    epochs: int = 50
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    dropout: float = 0.1
    smoothness_weight: float = 0.20
    seed: int = 42
    device: str = "cpu"
    # Checkpointing paths
    checkpoint_path: str = "models/gnn_model.pt"
    log_path: str = "logs/gnn_training.json"
    paradigm: str = "SEMI_SUPERVISED_RISK_PROPAGATION_DEMO"
    attention_disclaimer: str = ATTENTION_CAUSALITY_DISCLAIMER

    def __post_init__(self) -> None:
        self.model_type = self.model_type.lower()
        if self.model_type not in (GNNModelType.GRAPHSAGE.value, GNNModelType.GAT.value):
            raise ValueError(f"Unsupported model_type: '{self.model_type}'. Choose 'graphsage' or 'gat'.")

        # Auto-specialize default checkpoint and log paths if not customized
        if self.checkpoint_path == "models/gnn_model.pt":
            self.checkpoint_path = f"models/{self.model_type}_phase{8 if self.model_type == 'graphsage' else 9}.pt"
        if self.log_path == "logs/gnn_training.json":
            self.log_path = f"logs/{self.model_type}_training.json"


DEFAULT_GNN_CONFIG = CentralizedGNNConfig()
