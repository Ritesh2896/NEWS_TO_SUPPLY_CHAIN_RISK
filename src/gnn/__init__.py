"""Graph Neural Network (GNN) modules for supply-chain risk propagation in BDS-35.

Supports both:
  - GraphSAGE (Hamilton et al., 2017)
  - Graph Attention Network (GAT) (Veličković et al., 2018)
"""

from src.gnn.config import (
    CentralizedGNNConfig,
    GNNModelType,
    DEFAULT_GNN_CONFIG,
    ATTENTION_CAUSALITY_DISCLAIMER,
)
from src.gnn.graphsage import GraphSAGEPropagationModel
from src.gnn.gat import GATPropagationModel
from src.gnn.dataset import (
    CANONICAL_NODE_TYPES,
    EXPERIMENTAL_SETUP_DISCLAIMER,
    GraphDatasetBundle,
    build_supply_chain_pyg_dataset,
)
from src.gnn.trainer import (
    GNNTrainingConfig,
    GraphSAGETrainer,
    set_reproducibility_seed,
    compute_graph_smoothness_loss,
)
from src.gnn.trainer_gat import GATTrainer
from src.gnn.inference import (
    predict_risk_propagation,
    score_graph,
)
from src.gnn.inference_gat import predict_gat_risk_propagation

__all__ = [
    "CentralizedGNNConfig",
    "GNNModelType",
    "DEFAULT_GNN_CONFIG",
    "ATTENTION_CAUSALITY_DISCLAIMER",
    "GraphSAGEPropagationModel",
    "GATPropagationModel",
    "CANONICAL_NODE_TYPES",
    "EXPERIMENTAL_SETUP_DISCLAIMER",
    "GraphDatasetBundle",
    "build_supply_chain_pyg_dataset",
    "GNNTrainingConfig",
    "GraphSAGETrainer",
    "GATTrainer",
    "set_reproducibility_seed",
    "compute_graph_smoothness_loss",
    "predict_risk_propagation",
    "predict_gat_risk_propagation",
    "score_graph",
]
