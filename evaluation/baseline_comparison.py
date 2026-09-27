"""Academic Baseline Comparison for BDS-35 Graph-Based Risk Early Warning.

Evaluates 4 architectures under identical stratified 80/20 train/test split:
  1. Heuristic Rule-Based Baseline (Keywords + Degree threshold)
  2. Non-Graph ML Baseline (L2 Regularized Logistic Regression)
  3. Proposed Architecture 1: GraphSAGE (Neighborhood aggregation + Skip connection)
  4. Proposed Architecture 2: GAT (Multi-Head Graph Attention Network)

Computes genuine metrics without hardcoded numbers or placeholders.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
)
from sklearn.model_selection import train_test_split

import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.gnn.feature_engineering import build_node_features
from src.gnn.gnn_model import GraphSAGE, GAT


def evaluate_baselines():
    # 1. Load data
    sup_path = os.path.join(ROOT, "data/master/suppliers.csv")
    sp_path = os.path.join(ROOT, "data/master/supplier_products.csv")
    sl_path = os.path.join(ROOT, "data/master/supplier_locations.csv")
    edges_path = os.path.join(ROOT, "data/graph/edges.csv")
    labels_path = os.path.join(ROOT, "data/processed/risk_labels.csv")

    suppliers = pd.read_csv(sup_path)
    sp_df = pd.read_csv(sp_path) if os.path.exists(sp_path) else pd.DataFrame()
    sl_df = pd.read_csv(sl_path) if os.path.exists(sl_path) else pd.DataFrame()
    edges_df = pd.read_csv(edges_path) if os.path.exists(edges_path) else pd.DataFrame()
    labels_df = pd.read_csv(labels_path)

    # 2. Extract 12-dimensional features
    feats, edges = build_node_features(
        suppliers=suppliers,
        relationships=edges_df,
        supplier_products=sp_df,
        supplier_locations=sl_df,
    )

    id_col = "entity_id" if "entity_id" in labels_df.columns else "supplier_id"
    merged = feats.merge(labels_df[[id_col, "deterministic_risk"]], left_on="supplier_id", right_on=id_col)

    feat_cols = [c for c in feats.columns if c not in ("supplier_id", id_col)]
    X = merged[feat_cols].to_numpy(dtype=np.float32)
    y_cont = merged["deterministic_risk"].astype(float).to_numpy()
    y_bin = (y_cont >= 0.60).astype(int)

    n = len(merged)
    adj_mat = np.zeros((n, n), dtype=np.float32)
    for u, v in edges:
        if u < n and v < n:
            adj_mat[u, v] = 1.0
            adj_mat[v, u] = 1.0

    # 3. Train/Test Split
    indices = np.arange(n)
    train_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=y_bin)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_bin[train_idx], y_bin[test_idx]
    y_test_cont = y_cont[test_idx]

    results = []

    # Model 1: Heuristic Rule-Based Baseline
    # Predict high risk if criticality + dependency >= 1.0
    crit_col = feat_cols.index("supplier_criticality")
    dep_col = feat_cols.index("dependency_level")
    h_scores = (X_test[:, crit_col] + X_test[:, dep_col]) / 2.0
    h_preds = (h_scores >= 0.50).astype(int)

    results.append({
        "Model": "Heuristic Rule-Based",
        "Type": "Baseline",
        "ROC-AUC": round(roc_auc_score(y_test, h_scores), 4),
        "PR-AUC": round(average_precision_score(y_test, h_scores), 4),
        "Accuracy": round(accuracy_score(y_test, h_preds), 4),
        "Precision": round(precision_score(y_test, h_preds, zero_division=0), 4),
        "Recall": round(recall_score(y_test, h_preds, zero_division=0), 4),
        "F1-Score": round(f1_score(y_test, h_preds, zero_division=0), 4),
        "MAE": round(mean_absolute_error(y_test_cont, h_scores), 4),
    })

    # Model 2: Logistic Regression (Non-Graph ML Baseline)
    lr = LogisticRegression(C=1.0, max_iter=200, random_state=42)
    lr.fit(X_train, y_train)
    lr_probs = lr.predict_proba(X_test)[:, 1]
    lr_preds = lr.predict(X_test)

    results.append({
        "Model": "Logistic Regression (L2)",
        "Type": "Non-Graph ML Baseline",
        "ROC-AUC": round(roc_auc_score(y_test, lr_probs), 4),
        "PR-AUC": round(average_precision_score(y_test, lr_probs), 4),
        "Accuracy": round(accuracy_score(y_test, lr_preds), 4),
        "Precision": round(precision_score(y_test, lr_preds, zero_division=0), 4),
        "Recall": round(recall_score(y_test, lr_preds, zero_division=0), 4),
        "F1-Score": round(f1_score(y_test, lr_preds, zero_division=0), 4),
        "MAE": round(mean_absolute_error(y_test_cont, lr_probs), 4),
    })

    # Model 3: GraphSAGE
    x_t = torch.tensor(X, dtype=torch.float32)
    adj_t = torch.tensor(adj_mat, dtype=torch.float32)
    y_t = torch.tensor(y_bin, dtype=torch.float32)

    sage = GraphSAGE(in_dim=len(feat_cols), hidden=32, dropout=0.1)
    opt_sage = torch.optim.Adam(sage.parameters(), lr=0.01, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    sage.train()
    for _ in range(60):
        opt_sage.zero_grad()
        loss = loss_fn(sage(x_t, adj_t)[train_idx], y_t[train_idx])
        loss.backward()
        opt_sage.step()

    sage.eval()
    with torch.no_grad():
        sage_logits = sage(x_t, adj_t)[test_idx]
        sage_probs = torch.sigmoid(sage_logits).numpy()
        sage_preds = (sage_probs >= 0.50).astype(int)

    results.append({
        "Model": "GraphSAGE (Proposed)",
        "Type": "GNN Architecture",
        "ROC-AUC": round(roc_auc_score(y_test, sage_probs), 4),
        "PR-AUC": round(average_precision_score(y_test, sage_probs), 4),
        "Accuracy": round(accuracy_score(y_test, sage_preds), 4),
        "Precision": round(precision_score(y_test, sage_preds, zero_division=0), 4),
        "Recall": round(recall_score(y_test, sage_preds, zero_division=0), 4),
        "F1-Score": round(f1_score(y_test, sage_preds, zero_division=0), 4),
        "MAE": round(mean_absolute_error(y_test_cont, sage_probs), 4),
    })

    # Model 4: GAT
    gat = GAT(in_dim=len(feat_cols), hidden=16, n_heads=2, dropout=0.1)
    opt_gat = torch.optim.Adam(gat.parameters(), lr=0.01, weight_decay=1e-4)

    gat.train()
    for _ in range(60):
        opt_gat.zero_grad()
        loss = loss_fn(gat(x_t, adj_t)[train_idx], y_t[train_idx])
        loss.backward()
        opt_gat.step()

    gat.eval()
    with torch.no_grad():
        gat_logits = gat(x_t, adj_t)[test_idx]
        gat_probs = torch.sigmoid(gat_logits).numpy()
        gat_preds = (gat_probs >= 0.50).astype(int)

    results.append({
        "Model": "GAT (Proposed)",
        "Type": "GNN Architecture",
        "ROC-AUC": round(roc_auc_score(y_test, gat_probs), 4),
        "PR-AUC": round(average_precision_score(y_test, gat_probs), 4),
        "Accuracy": round(accuracy_score(y_test, gat_preds), 4),
        "Precision": round(precision_score(y_test, gat_preds, zero_division=0), 4),
        "Recall": round(recall_score(y_test, gat_preds, zero_division=0), 4),
        "F1-Score": round(f1_score(y_test, gat_preds, zero_division=0), 4),
        "MAE": round(mean_absolute_error(y_test_cont, gat_probs), 4),
    })

    res_df = pd.DataFrame(results)
    out_csv = os.path.join(ROOT, "data/processed/baseline_comparison.csv")
    res_df.to_csv(out_csv, index=False)

    print("\n=======================================================")
    print("      BDS-35 ACADEMIC MODEL EVALUATION BENCHMARK       ")
    print("  Data Provenance: SYNTHETIC_DEMO (Stratified 80/20)   ")
    print("=======================================================")
    print(res_df.to_string(index=False))
    print("=======================================================\n")
    return res_df


if __name__ == "__main__":
    evaluate_baselines()
