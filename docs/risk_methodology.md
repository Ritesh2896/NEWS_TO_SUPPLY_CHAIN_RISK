# BDS-35 Risk Scoring & Alert Methodology

## 1. Overview of Multi-Tier Risk Evaluation

Supply chain risk cannot be captured through a single opaque machine learning model or a static geographic distance check. The BDS-35 platform implements a transparent, multi-layered risk evaluation framework combining:
1. **Geospatial Exposure Decay**: Non-linear proximity assessment between event epicenters and supplier operating facilities.
2. **Deterministic Heuristic Risk Engine**: An explainable, multi-factor rule-based index normalized to $[0.0, 1.0]$.
3. **Graph Neural Network Propagation**: Topological shock diffusion across multi-tier supplier dependency networks.
4. **Tri-Model Synthesis**: Configurable weighted blending into operational risk bands with human-readable explanations.

---

## 2. Geospatial Exposure Engine

Geospatial exposure measures the physical proximity between a disruption epicenter and the operating facilities of a supplier.

### 2.1 Haversine Distance Formula
Given the coordinates of an event epicenter $(\phi_1, \lambda_1)$ and a supplier facility $(\phi_2, \lambda_2)$ in radians:
$$a = \sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)$$
$$c = 2 \cdot \text{atan2}\left(\sqrt{a}, \sqrt{1-a}\right)$$
$$d = R \cdot c \quad (\text{where } R = 6371.0 \text{ km})$$

### 2.2 Exposure Zones & Decay Scores
Physical proximity alone does not guarantee disruption, but operational vulnerability decreases with distance. BDS-35 converts distance into an exposure score $E_{geo} \in [0.1, 1.0]$ across 4 calibrated zones:

| Zone | Distance Range ($d$) | Exposure Score ($E_{geo}$) | Operational Description |
| :--- | :--- | :---: | :--- |
| **Zone 1: Direct Impact** | $0 \le d \le 25\text{ km}$ | **1.00** | Immediate blast radius (terminal shutdown, localized flood, city curfew) |
| **Zone 2: Near Impact** | $25 < d \le 100\text{ km}$ | **0.70** | Commuting disruptions, feeder logistics delays, secondary power outages |
| **Zone 3: Regional Exposure** | $100 < d \le 250\text{ km}$ | **0.40** | Arterial freight congestion, regional customs backlogs |
| **Zone 4: Distant Exposure** | $d > 250\text{ km}$ | **0.10** | Minimal direct physical impact; macroscopic market effects only |

When a supplier operates multiple facilities, the platform evaluates all sites and assigns the maximum exposure observed across operating facilities:
$$E_{geo}(s, e) = \max_{f \in \text{Facilities}(s)} E_{geo}(\text{Location}(f), \text{Location}(e))$$

---

## 3. Deterministic Heuristic Risk Engine

The deterministic risk score provides an audited, transparent baseline index computed directly from structural attributes and event indicators.

### 3.1 Candidate Factors ($f_i \in [0.0, 1.0]$)
1. **Event Severity ($S_e$)**: Disruption intensity estimated by the NLP module based on headline phrasing, lexical intensifiers, and duration indicators.
2. **Geographic Exposure ($E_{geo}$)**: Haversine exposure score of the supplier's closest active facility to the disruption epicenter.
3. **Supplier Criticality ($C_s$)**: Enterprise-assigned operational tier score from `suppliers.csv`.
4. **Product Criticality ($C_p$)**: Highest criticality score among products sourced from the affected supplier from `products.csv`.
5. **Dependency Strength ($D_{dep}$)**: Enterprise sourcing share or bill-of-materials volume fraction allocated to this supplier.
6. **Single-Source Dependency ($P_{ss}$)**: Binary penalty flag ($1.0$ if the supplier is a sole source for any critical component, $0.0$ otherwise).

### 3.2 Weighted Formulation & Constraints
$$\text{Deterministic Risk} = \sum_{i=1}^6 w_i \cdot f_i$$

Subject to the strict normalization constraint:
$$\sum_{i=1}^6 w_i = 1.00 \quad \text{where } w_i \ge 0$$

**Default Calibrated Weights**:
- $w_1 (\text{Event Severity}) = 0.25$
- $w_2 (\text{Geospatial Exposure}) = 0.20$
- $w_3 (\text{Supplier Criticality}) = 0.15$
- $w_4 (\text{Product Criticality}) = 0.15$
- $w_5 (\text{Dependency Strength}) = 0.15$
- $w_6 (\text{Single Source Penalty}) = 0.10$

---

## 4. Tri-Model Combined Risk Formulation

To balance explainable heuristic rules with learned topological shock diffusion, BDS-35 blends three independent estimators:
$$\text{Combined Risk} = \alpha \cdot \text{Deterministic Risk} + \beta \cdot \text{GraphSAGE Risk} + \gamma \cdot \text{GAT Risk}$$

**Default Ensemble Configuration**:
- $\alpha = 0.50$ (Deterministic baseline)
- $\beta = 0.25$ (GraphSAGE topological diffusion)
- $\gamma = 0.25$ (GAT attention propagation)
- Constraints: $\alpha + \beta + \gamma = 1.00$, and $\text{Combined Risk} \in [0.0, 1.0]$.

---

## 5. Operational Risk Bands

Continuous combined risk scores are discretized into operational action tiers:

| Risk Band | Score Threshold | Operational Protocol |
| :--- | :---: | :--- |
| **LOW** | $[0.00, 0.40)$ | Routine automated monitoring. No procurement escalation required. |
| **MEDIUM** | $[0.40, 0.60)$ | Warning flagged. Automated inventory buffer checks triggered. |
| **HIGH** | $[0.60, 0.75)$ | Escalated alert. Alternate sourcing identification and safety stock audit. |
| **CRITICAL** | $[0.75, 1.00]$ | Immediate emergency protocol. Executive escalation and logistics rerouting. |

---

## 6. Canonical Explanation Generation

Every alert generated by BDS-35 includes an automated, human-interpretable rationale explaining which specific factors drove the risk score:

```
Supplier: SUP00014 (Shenzhen Semiconductor Components)
Risk Score: 0.785 (CRITICAL)
Contributing Factors:
1. Event Severity is elevated (0.80) due to reported industrial fire.
2. Direct Geographic Exposure detected (Distance: 12.4 km in Zone 1).
3. Supplier Criticality is high (0.85) for Tier-1 electronic components.
4. Single-source dependency identified for Critical Component PRD00042.
5. Topological shock propagated via GraphSAGE/GAT from upstream supplier SUP00088.
```

---

## 7. Mandatory Academic Disclaimers

> **PROJECT THRESHOLD DISCLAIMER**:
> Operational risk bands (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`) and contributing weights are project-defined heuristic calibration thresholds engineered for proactive early warning and decision triage. They do NOT represent externally verified ground-truth probabilities of default or bankruptcy.
