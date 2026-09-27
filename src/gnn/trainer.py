"""Reproducible Trainer for GraphSAGE Risk Propagation (Phase 8).

Trains GraphSAGE under a clearly documented demonstration/semi-supervised regime:
  - Uses Phase 7 deterministic heuristic risk as initial seed signals on labeled nodes.
  - Uses Graph Dirichlet / Laplacian smoothness loss across supply-chain edges to enforce
    consistent risk propagation across connected dependencies:
      Loss = Loss_seed + alpha * Loss_smoothness
  - Strictly adheres to academic integrity:
      * Never reports fabricated accuracy metrics.
      * Reports genuine optimization metrics: Epoch loss, seed MSE, smoothness delta.
      * Ensures strict reproducibility with seeds.
      * Saves training logs and checkpoints.
"""
from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import asdict, dataclass
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.gnn.dataset import GraphDatasetBundle
from src.gnn.graphsage import GraphSAGEPropagationModel

logger = logging.getLogger("graphsage_trainer")


@dataclass
class GNNTrainingConfig:
    """Hyperparameters and configuration for GraphSAGE training."""
    epochs: int = 50
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    hidden_dim: int = 32
    dropout: float = 0.1
    smoothness_weight: float = 0.20
    seed: int = 42
    device: str = "cpu"
    checkpoint_path: str = "models/graphsage_phase8.pt"
    log_path: str = "logs/graphsage_training.json"
    paradigm: str = "SEMI_SUPERVISED_RISK_PROPAGATION_DEMO"


def set_reproducibility_seed(seed: int = 42) -> None:
    """Set seeds across Python, NumPy, and PyTorch for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_graph_smoothness_loss(
    predictions: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute Dirichlet energy / graph Laplacian smoothness loss across edges.

    Encourages connected nodes in the supply chain graph to have smooth, topologically
    consistent risk propagation.
    """
    if edge_index.numel() == 0 or edge_index.size(1) == 0:
        return torch.tensor(0.0, device=predictions.device)

    src, dst = edge_index[0], edge_index[1]
    diff = predictions[src] - predictions[dst]
    sq_diff = diff.pow(2)

    if edge_weight is not None and edge_weight.numel() == sq_diff.size(0):
        weighted_sq_diff = edge_weight.unsqueeze(-1) * sq_diff
        return weighted_sq_diff.mean()

    return sq_diff.mean()


class GraphSAGETrainer:
    """Trains GraphSAGE with documented semi-supervised risk propagation."""

    def __init__(self, config: GNNTrainingConfig | None = None) -> None:
        self.config = config or GNNTrainingConfig()
        set_reproducibility_seed(self.config.seed)
        self.device = torch.device(self.config.device if torch.cuda.is_available() and self.config.device != "cpu" else "cpu")

    def train(self, dataset: GraphDatasetBundle) -> dict[str, Any]:
        """Execute reproducible training loop and persist logs and checkpoints.

        Args:
            dataset: GraphDatasetBundle containing PyG Data and node mappings.

        Returns:
            Dictionary containing training summary, loss history, and checkpoint path.
        """
        data = dataset.data.to(self.device)
        in_dim = data.x.size(1)

        # Initialize model
        model = GraphSAGEPropagationModel(
            in_dim=in_dim,
            hidden_dim=self.config.hidden_dim,
            out_dim=1,
            dropout=self.config.dropout,
        ).to(self.device)

        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        mse_criterion = nn.MSELoss()

        # Identify nodes with non-zero seed targets for semi-supervised supervision
        seed_mask = (data.y > 0.0).squeeze(-1)
        has_seeds = seed_mask.sum().item() > 0

        loss_history: list[dict[str, float]] = []

        model.train()
        for epoch in range(1, self.config.epochs + 1):
            optimizer.zero_grad()

            risk_pred, _ = model(data.x, data.edge_index, data.edge_weight)

            # 1. Semi-supervised seed loss (MSE against deterministic baseline signals)
            if has_seeds:
                seed_loss = mse_criterion(risk_pred[seed_mask], data.y[seed_mask])
            else:
                # Self-supervised fallback: smooth auto-propagation
                seed_loss = mse_criterion(risk_pred, data.x[:, 12:13])

            # 2. Graph Laplacian smoothness loss
            smooth_loss = compute_graph_smoothness_loss(risk_pred, data.edge_index, data.edge_weight)

            # Combined objective
            total_loss = seed_loss + self.config.smoothness_weight * smooth_loss

            total_loss.backward()
            optimizer.step()

            loss_history.append({
                "epoch": epoch,
                "total_loss": round(float(total_loss.item()), 6),
                "seed_loss": round(float(seed_loss.item()), 6),
                "smoothness_loss": round(float(smooth_loss.item()), 6),
                "mean_risk_pred": round(float(risk_pred.mean().item()), 4),
            })

        # Save checkpoint
        metadata = {
            "config": asdict(self.config),
            "final_loss": loss_history[-1]["total_loss"],
            "epochs_trained": self.config.epochs,
            "experimental_setup": dataset.experimental_setup,
            "feature_names": dataset.feature_names,
        }
        model.save_checkpoint(self.config.checkpoint_path, metadata=metadata)

        # Save training logs
        os.makedirs(os.path.dirname(os.path.abspath(self.config.log_path)), exist_ok=True)
        with open(self.config.log_path, "w", encoding="utf-8") as f:
            json.dump({
                "config": asdict(self.config),
                "training_log": loss_history,
                "experimental_setup": dataset.experimental_setup,
                "academic_disclaimer": (
                    "No synthetic classification accuracy is reported. The metrics reflect "
                    "loss convergence during semi-supervised graph risk propagation training."
                ),
            }, f, indent=2)

        return {
            "status": "COMPLETED",
            "checkpoint_path": self.config.checkpoint_path,
            "log_path": self.config.log_path,
            "final_loss": loss_history[-1]["total_loss"],
            "epochs": self.config.epochs,
            "experimental_setup": dataset.experimental_setup,
        }
