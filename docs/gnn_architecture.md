# BDS-35 GNN Architecture

This build adapts the public `oladri-renuka/supply-chain-risk-gnn` design ideas to the BDS-35 problem. The reference repository describes NewsAPI ingestion, a heterogeneous supply-chain graph, GraphSAGE/GAT, XGBoost/logistic baselines, temporal evaluation, and GNNExplainer. BDS-35 keeps the graph-learning idea but changes the production objective to event extraction + geospatial supplier/product exposure + explainable alerts.

## Production flow
NewsAPI -> relevance -> event/entity extraction -> supplier/product/location linking -> graph -> deterministic risk -> graph propagation -> combined alert -> FastAPI/Streamlit.

## GNN implementation
`src/gnn/gnn_model.py` contains CPU-friendly GraphSAGE/GAT-style PyTorch models without a PyG dependency. `src/gnn/train.py` can train on a labelled `data/labels.csv`. Until weights are trained, `src/gnn/inference.py` uses a transparent topology-aware propagation fallback so the demo remains runnable.

## 12 feature slots
The reference project's 12-dimensional feature idea is retained, but BDS-35 fills unavailable financial fields with 0 until real financial/master data is supplied. News risk, media volume, supplier concentration, geographic volatility and PageRank are populated by the BDS-35 pipeline.
