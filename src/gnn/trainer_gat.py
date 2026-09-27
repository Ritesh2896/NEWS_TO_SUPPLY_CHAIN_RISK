"""Reproducible Trainer for Graph Attention Networks (GAT) in BDS-35 (Phase 9).

Trains GAT using centralized configuration with semi-supervised seed supervision
and graph Laplacian smoothness regularization:
  - Uses identical feature schema and dataset bundle as GraphSAGE.
  - Generates convergence logs without reporting fabricated accuracy percentages.
  - Enforces reproducibility across random seeds.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim

from src.gnn.config import CentralizedGNNConfig, GNNModelType, ATTENTION_CAUSALITY_DISCLAIMER
from src.gnn.dataset import GraphDatasetBundle
from src.gnn.gat import GATPropagationModel
from src.gnn.trainer import set_reproducibility_seed, compute_graph_smoothness_loss

logger = logging.getLogger("gat_trainer")


class GATTrainer:
    """Trains Graph Attention Network (GAT) with centralized configuration."""

    def __init__(self, config: CentralizedGNNConfig | None = None) -> None:
        self.config = config or CentralizedGNNConfig(model_type=GNNModelType.GAT.value)
        set_reproducibility_seed(self.config.seed)
        self.device = torch.device(self.config.device if torch.cuda.is_available() and self.config.device != "cpu" else "cpu")

    def train(self, dataset: GraphDatasetBundle) -> dict[str, Any]:
        """Execute reproducible training loop and persist checkpoints and logs.

        Args:
            dataset: GraphDatasetBundle containing PyG Data and node mappings.

        Returns:
            Dictionary containing training summary, loss history, and checkpoint path.
        """
        data = dataset.data.to(self.device)
        in_dim = data.x.size(1)

        # Initialize GAT model
        model = GATPropagationModel(
            in_dim=in_dim,
            hidden_dim=self.config.hidden_dim,
            out_dim=1,
            heads=self.config.num_heads,
            dropout=self.config.dropout,
        ).to(self.device)

        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        mse_criterion = nn.MSELoss()

        # Seed mask for semi-supervised supervision
        seed_mask = (data.y > 0.0).squeeze(-1)
        has_seeds = seed_mask.sum().item() > 0

        loss_history: list[dict[str, float]] = []

        model.train()
        for epoch in range(1, self.config.epochs + 1):
            optimizer.zero_grad()

            risk_pred, _, _ = model(data.x, data.edge_index, return_attention_weights=False)

            # 1. Semi-supervised seed loss (MSE against deterministic risk baseline)
            if has_seeds:
                seed_loss = mse_criterion(risk_pred[seed_mask], data.y[seed_mask])
            else:
                seed_loss = mse_criterion(risk_pred, data.x[:, 12:13])

            # 2. Graph Laplacian smoothness loss across edges
            smooth_loss = compute_graph_smoothness_loss(risk_pred, data.edge_index, data.edge_weight)

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
            "disclaimer": ATTENTION_CAUSALITY_DISCLAIMER,
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
                    "loss convergence during semi-supervised GAT risk propagation training. "
                    + ATTENTION_CAUSALITY_DISCLAIMER
                ),
            }, f, indent=2)

        return {
            "status": "COMPLETED",
            "model_type": "GAT",
            "checkpoint_path": self.config.checkpoint_path,
            "log_path": self.config.log_path,
            "final_loss": loss_history[-1]["total_loss"],
            "epochs": self.config.epochs,
            "experimental_setup": dataset.experimental_setup,
        }
