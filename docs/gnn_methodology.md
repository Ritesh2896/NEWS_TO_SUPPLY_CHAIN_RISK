# BDS-35 Graph Neural Network (GNN) Methodology

## 1. Overview & Theoretical Motivation

Supply chains are inherently networked systems. Disruption shocks do not impact entities in isolation; a localized factory shutdown at an obscure tier-3 raw material supplier can propagate cascades of component shortages across intermediate tier-2 sub-assemblers, ultimately halting assembly lines at tier-1 contractors.

Traditional linear or tree-based risk models treat suppliers as independent tabular rows, failing to model multi-hop transitive dependencies. BDS-35 utilizes **Graph Neural Networks (GNNs)** built on **PyTorch Geometric** to learn inductive representations over multi-relational supply chain topologies.

We deploy two complementary neural architectures:
1. **GraphSAGE (SAmple and aggreGatE)**: For scalable, inductive structural feature propagation.
2. **Graph Attention Networks (GAT)**: For multi-head attention-weighted neighborhood aggregation and localized edge interpretability.

---

## 2. Input Representation & Feature Engineering

### 2.1 16-Dimensional Node Feature Schema ($\mathbf{x}_v \in \mathbb{R}^{16}$)
Each node in the homogeneous supplier projection graph carries a normalized 16-dimensional feature vector:

| Feature Index | Feature Name | Range | Description |
| :---: | :--- | :---: | :--- |
| $0$ | `criticality_score` | $[0.0, 1.0]$ | Inherent operational criticality of the supplier |
| $1$ | `financial_stability` | $[0.0, 1.0]$ | Financial resilience score (inverted risk) |
| $2$ | `tier_normalized` | $[0.0, 1.0]$ | Tier depth normalized ($1/1, 1/2, 1/3$) |
| $3$ | `max_product_criticality` | $[0.0, 1.0]$ | Maximum criticality across all sourced products |
| $4$ | `avg_sourcing_share` | $[0.0, 1.0]$ | Mean enterprise sourcing allocation |
| $5$ | `single_source_flag` | $\{0.0, 1.0\}$ | Indicator if supplier is a sole source |
| $6$ | `event_severity_max` | $[0.0, 1.0]$ | Peak severity among directly linked events |
| $7$ | `geo_exposure_score` | $[0.1, 1.0]$ | Peak Haversine exposure score across facilities |
| $8$ | `deterministic_risk` | $[0.0, 1.0]$ | Baseline heuristic risk score |
| $9$ | `in_degree_norm` | $[0.0, 1.0]$ | Normalized number of upstream dependencies |
| $10$ | `out_degree_norm` | $[0.0, 1.0]$ | Normalized number of downstream dependents |
| $11$ | `pagerank_score` | $[0.0, 1.0]$ | Structural topological centrality |
| $12$ | `clustering_coeff` | $[0.0, 1.0]$ | Local neighborhood density |
| $13$ | `facility_count_norm` | $[0.0, 1.0]$ | Normalized count of operating facilities |
| $14$ | `product_count_norm` | $[0.0, 1.0]$ | Normalized breadth of component catalog |
| $15$ | `recent_news_count` | $[0.0, 1.0]$ | Normalized volume of articles mentioning entity |

---

## 3. Architecture 1: GraphSAGE Propagation

GraphSAGE learns inductive node representations by aggregating feature information from localized neighborhoods.

### 3.1 Layer Formulation
For node $v \in \mathcal{V}$ at layer $k \in \{1, 2\}$:
$$\mathbf{h}_{\mathcal{N}(v)}^{(k)} = \text{AGGREGATE}_k \left( \left\{ \mathbf{h}_u^{(k-1)}, \forall u \in \mathcal{N}(v) \right\} \right)$$
$$\mathbf{h}_v^{(k)} = \sigma \left( \mathbf{W}^{(k)} \cdot \left[ \mathbf{h}_v^{(k-1)} \,\|\, \mathbf{h}_{\mathcal{N}(v)}^{(k)} \right] \right)$$

Where:
- $\text{AGGREGATE}$ uses mean aggregation: $\frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} \mathbf{h}_u^{(k-1)}$.
- $\sigma(\cdot)$ is the LeakyReLU activation function.
- Dropout ($p = 0.20$) is applied between layers for regularization.
- Output Layer: A linear projection $\mathbf{w}_{out}^\top \mathbf{h}_v^{(2)}$ followed by a Sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$ outputs the continuous scalar propagation risk $\hat{y}_v \in [0.0, 1.0]$.

### 3.2 Dirichlet Smoothness Regularization
Because real-world labeled ground-truth supplier default labels are scarce, the GraphSAGE model is trained via self-supervised Dirichlet energy minimization over supplier dependency edges $\mathcal{E}_{dep}$:
$$\mathcal{L}_{\text{Dirichlet}} = \frac{1}{|\mathcal{E}_{dep}|} \sum_{(u, v) \in \mathcal{E}_{dep}} w_{uv} \left\| \hat{y}_u - \hat{y}_v \right\|^2 + \lambda \sum_{v \in \mathcal{V}} \left\| \hat{y}_v - y_v^{\text{det}} \right\|^2$$
This ensures that risk signals diffuse smoothly across strongly dependent supplier pairs while anchoring to explainable deterministic baselines.

---

## 4. Architecture 2: Graph Attention Network (GAT)

While GraphSAGE weights all neighbors equally (or by static edge weights), Graph Attention Networks compute dynamic attention coefficients, allowing the model to assign higher weights to catastrophic upstream bottlenecks.

### 4.1 Multi-Head Attention Formulation
For edge $(i, j)$ at layer $k$:
$$\alpha_{ij} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^\top \left[ \mathbf{W}\mathbf{h}_i \,\|\, \mathbf{W}\mathbf{h}_j \right] \right)\right)}{\sum_{k \in \mathcal{N}(i)} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^\top \left[ \mathbf{W}\mathbf{h}_i \,\|\, \mathbf{W}\mathbf{h}_k \right] \right)\right)}$$

With multi-head attention ($K = 2$ heads):
$$\mathbf{h}_i^{(1)} = \Vert_{k=1}^K \sigma\left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^k \mathbf{W}^k \mathbf{h}_j \right)$$

### 4.2 Neighborhood Attention Extraction & Interpretability
During inference, the attention coefficients $\alpha_{ij}$ are preserved and exported. In the Streamlit dashboard and API, supply chain analysts can inspect which upstream suppliers contributed most heavily to a target supplier's elevated risk score.

---

## 5. Model Checkpoints & Determinism

- Models are trained with fixed random seeds (`torch.manual_seed(42)`, `np.random.seed(42)`).
- Model artifacts are saved to:
  - `models/graphsage_phase8.pt`
  - `models/gat_phase9.pt`
- Both models execute in $< 50\text{ ms}$ on standard CPU environments, satisfying low-latency early warning constraints.

---

## 6. Academic Disclaimers & Integrity Constraints

> **CRITICAL NON-CAUSALITY NOTICE**:
> Graph Attention Network (GAT) attention weights reflect localized feature aggregation importance within the defined topological neighborhood. **Attention weights do NOT prove, demonstrate, or establish causal relationships.** An elevated attention weight indicates mathematical feature correlation, not verified physical disruption causation.
>
> **NO ACCURACY METRIC FABRICATION**:
> In accordance with scientific integrity guidelines, BDS-35 does not claim or fabricate synthetic classification accuracy, ROC-AUC, or F1 scores against synthetic demo datasets. Model readiness is verified strictly through Dirichlet convergence, deterministic output bounds, and gradient stability.
