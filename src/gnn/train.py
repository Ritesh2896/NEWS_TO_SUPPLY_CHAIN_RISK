"""Train GraphSAGE and GAT models on supply-chain risk labels.

Evaluates performance on a reproducible train/test split using ROC-AUC, PR-AUC, F1,
Precision, and Recall. Persists model weights (.pt) and evaluation metadata (.json).
"""
from __future__ import annotations
import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.gnn.feature_engineering import build_node_features
from src.gnn.gnn_model import GraphSAGE, GAT, save_model

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def train_model(
    model_type: str = "graphsage",
    epochs: int = 100,
    lr: float = 0.01,
    labels_path: str = "data/processed/risk_labels.csv",
    out_path: str = "models/graphsage_model.pt",
) -> dict[str, Any]:
    """Train GraphSAGE or GAT on the dataset and return genuine evaluation metrics."""
    # 1. Load data tables
    sup_path = os.path.join(ROOT, "data/master/suppliers.csv")
    sp_path = os.path.join(ROOT, "data/master/supplier_products.csv")
    sl_path = os.path.join(ROOT, "data/master/supplier_locations.csv")
    edges_path = os.path.join(ROOT, "data/graph/edges.csv")

    suppliers = pd.read_csv(sup_path) if os.path.exists(sup_path) else pd.DataFrame()
    sp_df = pd.read_csv(sp_path) if os.path.exists(sp_path) else pd.DataFrame()
    sl_df = pd.read_csv(sl_path) if os.path.exists(sl_path) else pd.DataFrame()
    edges_df = pd.read_csv(edges_path) if os.path.exists(edges_path) else pd.DataFrame()

    # 2. Extract features and edges
    feats, edges = build_node_features(
        suppliers=suppliers,
        relationships=edges_df,
        supplier_products=sp_df,
        supplier_locations=sl_df,
    )

    if feats.empty:
        raise RuntimeError("No supplier features could be extracted.")

    # 3. Load labels
    full_labels_path = os.path.join(ROOT, labels_path) if not os.path.isabs(labels_path) else labels_path
    if not os.path.exists(full_labels_path):
        raise FileNotFoundError(f"Labels file not found: {full_labels_path}")

    lab_df = pd.read_csv(full_labels_path)
    id_col = "entity_id" if "entity_id" in lab_df.columns else "supplier_id"

    # Merge features with labels
    m = feats.merge(lab_df[[id_col, "deterministic_risk"]], left_on="supplier_id", right_on=id_col, how="inner")
    if m.empty:
        raise RuntimeError("No matching supplier IDs between features and labels.")

    # Derive binary label: High/Critical risk (>= 0.60) vs Low/Medium (< 0.60)
    # This reflects the operational alert threshold in supply-chain risk
    y_raw = m["deterministic_risk"].astype(float).to_numpy()
    y_binary = (y_raw >= 0.60).astype(np.float32)

    feat_cols = [c for c in feats.columns if c != "supplier_id" and c != id_col]
    X_mat = m[feat_cols].to_numpy(np.float32)
    n = len(m)

    # 4. Construct adjacency tensor
    adj_mat = np.zeros((n, n), dtype=np.float32)
    for u, v in edges:
        if u < n and v < n:
            adj_mat[u, v] = 1.0
            adj_mat[v, u] = 1.0
    adj_tensor = torch.tensor(adj_mat, dtype=torch.float32)

    # 5. Train / Test Split
    indices = np.arange(n)
    train_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=y_binary)

    x_tensor = torch.tensor(X_mat, dtype=torch.float32)
    y_tensor = torch.tensor(y_binary, dtype=torch.float32)

    # 6. Instantiate Model
    in_dim = len(feat_cols)
    if model_type.lower() == "gat":
        model = GAT(in_dim=in_dim, hidden=16, n_heads=2, dropout=0.1)
    else:
        model = GraphSAGE(in_dim=in_dim, hidden=32, dropout=0.1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    # 7. Training Loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(x_tensor, adj_tensor)
        loss = loss_fn(logits[train_idx], y_tensor[train_idx])
        loss.backward()
        optimizer.step()

    # 8. Evaluation on Test Set
    model.eval()
    with torch.no_grad():
        test_logits = model(x_tensor, adj_tensor)[test_idx]
        test_probs = torch.sigmoid(test_logits).numpy()
        test_preds = (test_probs >= 0.50).astype(int)
        y_test_np = y_binary[test_idx]

    roc_auc = float(roc_auc_score(y_test_np, test_probs)) if len(np.unique(y_test_np)) > 1 else 0.5
    pr_auc = float(average_precision_score(y_test_np, test_probs)) if len(np.unique(y_test_np)) > 1 else 0.5
    acc = float(accuracy_score(y_test_np, test_preds))
    prec = float(precision_score(y_test_np, test_preds, zero_division=0))
    rec = float(recall_score(y_test_np, test_preds, zero_division=0))
    f1 = float(f1_score(y_test_np, test_preds, zero_division=0))

    metrics = {
        "model_type": model_type.upper(),
        "total_nodes": n,
        "train_samples": len(train_idx),
        "test_samples": len(test_idx),
        "epochs": epochs,
        "learning_rate": lr,
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "data_status": "SYNTHETIC_DEMO_BENCHMARK",
        "evaluation_standard": "Stratified 80/20 Holdout",
    }

    # 9. Save Artifacts
    full_out = os.path.join(ROOT, out_path) if not os.path.isabs(out_path) else out_path
    save_model(model, full_out, metrics)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train GraphSAGE or GAT for BDS-35")
    parser.add_argument("--labels", default="data/processed/risk_labels.csv")
    parser.add_argument("--model", choices=["graphsage", "gat", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    models_to_train = ["graphsage", "gat"] if args.model == "both" else [args.model]
    for m in models_to_train:
        out = f"models/{m}_model.pt"
        print(f"\n--- Training {m.upper()} ({args.epochs} epochs) ---")
        metrics = train_model(model_type=m, epochs=args.epochs, lr=args.lr, labels_path=args.labels, out_path=out)
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
