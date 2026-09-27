# Real-time architecture

1. NewsAPI `/v2/everything` retrieves recent articles using topic + date window.
2. Cross-article URL/title deduplication removes repeats.
3. Local relevance filter removes unrelated articles.
4. Local NLP extracts organizations, locations and disruption event types.
5. Entity linking maps extracted organizations to supplier master data.
6. Product/location relationships create the supply-chain graph.
7. Geospatial distance contributes exposure.
8. Deterministic risk engine creates an explainable event risk score.
9. GraphSAGE/GAT-style propagation adds topology-aware risk when graph features are available; a transparent fallback keeps the app runnable before model training.
10. FastAPI exposes the pipeline and Streamlit provides analyst UI.

No OpenAI or Gemini key is required by this build.
