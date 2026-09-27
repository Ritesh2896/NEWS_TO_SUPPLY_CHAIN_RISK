# BDS-35 Reproducibility Guide

## 1. Environment & Prerequisites

To ensure complete experimental and operational reproducibility, the BDS-35 platform is engineered to run on standard commodity hardware (CPU or CUDA-enabled GPU) without requiring proprietary cloud AI API subscriptions.

### System Specifications
- **Operating System**: Windows 10/11, Ubuntu 22.04 LTS, or macOS Sonoma
- **Python Version**: Python 3.11 or Python 3.13 (Verified on Python 3.13.5 64-bit)
- **Disk Space**: ~2 GB (including PyTorch and PyTorch Geometric dependencies)

---

## 2. Step-by-Step Installation (Windows PowerShell)

```powershell
# 1. Clone repository & navigate to workspace
cd c:\BDS35_News_to_Risk

# 2. Create isolated virtual environment
python -m venv .venv

# 3. Activate virtual environment
.venv\Scripts\Activate.ps1

# 4. Upgrade core build tools
python -m pip install --upgrade pip setuptools wheel

# 5. Install pinned dependencies
pip install -r requirements.txt
```

---

## 3. Environment Configuration

Copy `.env.example` to create your local `.env`:
```powershell
Copy-Item .env.example .env
```

Contents of `.env`:
```ini
# Add a valid free or developer key from https://newsapi.org/register
NEWSAPI_API_KEY=your_newsapi_key_here

# Pipeline runtime defaults
LIVE_NEWS_TOPIC=supply chain disruptions ports logistics manufacturing trade
LIVE_NEWS_PROVIDER=newsapi
LIVE_NEWS_HOURS=48
LIVE_NEWS_LIMIT=8
DATABASE_URL=sqlite:///./bds35.db
```

*Note: If no API key is provided, the ingestion engine operates in static replay mode using benchmark articles, ensuring zero pipeline crashes.*

---

## 4. Deterministic Seeds & Randomness Control

The platform strictly seeds all random number generators across Python, NumPy, and PyTorch:

```python
import os
import random
import numpy as np
import torch

SEED = 42

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

All model weight initializations, node shuffling, and data split functions utilize `SEED = 42`.

---

## 5. Model Training Reproduction

To re-train the Graph Neural Network models from scratch:

### 5.1 Train GraphSAGE (Phase 8)
```powershell
python -m src.gnn.trainer --model graphsage --epochs 100 --lr 0.01 --seed 42
```
- Output Checkpoint: `models/graphsage_phase8.pt`

### 5.2 Train Graph Attention Network (Phase 9)
```powershell
python -m src.gnn.trainer_gat --epochs 100 --lr 0.005 --heads 2 --seed 42
```
- Output Checkpoint: `models/gat_phase9.pt`

---

## 6. Pipeline Execution Reproduction

### 6.1 Direct Command-Line Smoke Test
```powershell
python -c "from src.pipeline.live_pipeline import execute_live_pipeline; res = execute_live_pipeline(topic='logistics strike factory port', hours=48, limit=5); print('STATUS:', res['status'], 'ALERTS:', res['alerts_count'])"
```

### 6.2 Running the FastAPI Backend
```powershell
python -m uvicorn src.api.main:app --reload --port 8000
```
- Test Health: Open `http://127.0.0.1:8000/health` in your browser.
- Interactive Docs: Open `http://127.0.0.1:8000/docs`.

### 6.3 Running the Streamlit Dashboard
```powershell
streamlit run dashboard/app.py
```
- Dashboard UI: Automatically opens at `http://localhost:8501`.

---

## 7. Full Test Suite Reproduction

To verify all 128 tests and compilation integrity:

```powershell
# Run all automated tests
pytest -q

# Run python syntax and byte-compilation audit
python -m compileall .
```
Both commands must complete with 0 failures and exit code 0.
