"""Generate professional UML and architectural diagrams for BDS-35 Project Report.

Generates:
1. gantt_chart.png
2. er_diagram.png
3. dfd_diagram.png
4. class_diagram.png
5. sequence_diagram.png
6. state_chart_diagram.png
7. use_case_diagram.png
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

OUT_DIR = os.path.abspath("reports/diagrams")
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

# -------------------------------------------------------------
# 1. GANTT CHART
# -------------------------------------------------------------
def make_gantt():
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    tasks = [
        "Project Inception & Architecture",
        "Master Dataset Curation & Integrity",
        "NewsAPI Ingestion & Deduplication",
        "Local NLP Event & Entity Extraction",
        "Deterministic 4-Stage Entity Linker",
        "Heterogeneous Graph Construction",
        "Haversine Geospatial Exposure Engine",
        "Deterministic Risk Scoring Engine",
        "GraphSAGE Topological Risk Model",
        "GAT Multi-Head Attention Model",
        "Tri-Model Alert Generation & Blend",
        "FastAPI REST Backend Serving",
        "Streamlit 12-Page Dashboard UI",
        "Comprehensive Automated Testing"
    ]
    starts = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
    durations = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(tasks)))
    
    y_pos = np.arange(len(tasks))
    ax.barh(y_pos, durations, left=starts, align='center', color=colors, edgecolor='#1E3A8A', height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tasks, fontsize=9, fontweight='medium')
    ax.invert_yaxis()
    ax.set_xlabel('Project Timeline (Weeks / Phases)', fontsize=10, fontweight='bold', labelpad=10)
    ax.set_title('BDS-35 Project Development Timeline (Gantt Chart)', fontsize=12, fontweight='bold', pad=15)
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "gantt_chart.png"), bbox_inches='tight')
    plt.close(fig)
    print("Generated gantt_chart.png")

# -------------------------------------------------------------
# 2. E-R DIAGRAM
# -------------------------------------------------------------
def make_er():
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    ax.axis('off')
    
    def draw_entity(x, y, w, h, title, fields, color='#E0F2FE', border='#0284C7'):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03", 
                                      linewidth=1.5, edgecolor=border, facecolor=color)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.04, title, ha='center', va='top', fontsize=9.5, fontweight='bold', color='#0F172A')
        line_y = y + h - 0.07
        ax.plot([x, x + w], [line_y, line_y], color=border, lw=1)
        field_text = "\n".join(fields)
        ax.text(x + 0.02, line_y - 0.02, field_text, ha='left', va='top', fontsize=7.5, family='monospace', color='#334155')

    def draw_rel(x, y, w, h, text):
        diamond = patches.Polygon([[x + w/2, y], [x + w, y + h/2], [x + w/2, y + h], [x, y + h/2]], 
                                  closed=True, edgecolor='#D97706', facecolor='#FEF3C7', lw=1.2)
        ax.add_patch(diamond)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=7.5, fontweight='bold', color='#92400E')

    # Entities
    draw_entity(0.05, 0.65, 0.22, 0.28, "SUPPLIERS", [
        "PK supplier_id: string",
        "   name: string",
        "   country: string",
        "   criticality: float",
        "   tier: integer",
        "   provenance: enum"
    ])
    
    draw_entity(0.72, 0.65, 0.23, 0.28, "PRODUCTS", [
        "PK product_id: string",
        "   name: string",
        "   category: string",
        "   criticality: float",
        "   lead_time_days: int",
        "   provenance: enum"
    ])

    draw_entity(0.05, 0.08, 0.22, 0.28, "FACILITIES", [
        "PK facility_id: string",
        "FK supplier_id: string",
        "FK location_id: string",
        "   facility_name: str",
        "   facility_type: enum",
        "   provenance: enum"
    ])

    draw_entity(0.72, 0.08, 0.23, 0.28, "LOCATIONS", [
        "PK location_id: string",
        "   name: string",
        "   country: string",
        "   latitude: float",
        "   longitude: float",
        "   location_type: str"
    ])

    draw_entity(0.38, 0.38, 0.24, 0.28, "RISK_ALERTS", [
        "PK alert_id: string",
        "FK supplier_id: string",
        "FK event_id: string",
        "   deterministic_risk",
        "   graphsage_risk",
        "   gat_risk, combined",
        "   risk_band, explanation"
    ])

    # Relationships
    draw_rel(0.40, 0.74, 0.20, 0.10, "SUPPLIES\n(Share, Single)")
    draw_rel(0.07, 0.44, 0.18, 0.10, "OPERATES\n(1 to N)")
    draw_rel(0.74, 0.44, 0.19, 0.10, "LOCATED_AT\n(N to 1)")
    draw_rel(0.40, 0.16, 0.20, 0.10, "EXPOSURE\n(Haversine)")

    # Connectors
    ax.annotate("", xy=(0.40, 0.79), xytext=(0.27, 0.79), arrowprops=dict(arrowstyle="-", color='#475569', lw=1.2))
    ax.annotate("", xy=(0.72, 0.79), xytext=(0.60, 0.79), arrowprops=dict(arrowstyle="-", color='#475569', lw=1.2))
    
    ax.annotate("", xy=(0.16, 0.65), xytext=(0.16, 0.54), arrowprops=dict(arrowstyle="-", color='#475569', lw=1.2))
    ax.annotate("", xy=(0.16, 0.44), xytext=(0.16, 0.36), arrowprops=dict(arrowstyle="-", color='#475569', lw=1.2))

    ax.annotate("", xy=(0.83, 0.65), xytext=(0.83, 0.54), arrowprops=dict(arrowstyle="-", color='#475569', lw=1.2))
    ax.annotate("", xy=(0.83, 0.44), xytext=(0.83, 0.36), arrowprops=dict(arrowstyle="-", color='#475569', lw=1.2))

    ax.annotate("", xy=(0.27, 0.22), xytext=(0.40, 0.21), arrowprops=dict(arrowstyle="-", color='#475569', lw=1.2))
    ax.annotate("", xy=(0.72, 0.22), xytext=(0.60, 0.21), arrowprops=dict(arrowstyle="-", color='#475569', lw=1.2))

    ax.set_title("BDS-35 Entity-Relationship (E-R) Diagram", fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "er_diagram.png"), bbox_inches='tight')
    plt.close(fig)
    print("Generated er_diagram.png")

# -------------------------------------------------------------
# 3. DFD (DATA FLOW DIAGRAM)
# -------------------------------------------------------------
def make_dfd():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.axis('off')

    def draw_proc(x, y, w, h, pid, name):
        circle = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.06",
                                        facecolor="#EFF6FF", edgecolor="#2563EB", lw=1.5)
        ax.add_patch(circle)
        ax.text(x + w/2, y + h - 0.035, pid, ha='center', va='top', fontsize=8, fontweight='bold', color='#1D4ED8')
        ax.text(x + w/2, y + h/2 - 0.01, name, ha='center', va='center', fontsize=8, fontweight='bold', color='#1E293B')

    def draw_store(x, y, w, h, sid, name):
        ax.plot([x, x + w], [y + h, y + h], color='#475569', lw=1.8)
        ax.plot([x, x + w], [y, y], color='#475569', lw=1.8)
        ax.fill_between([x, x + w], [y, y], [y + h, y + h], color='#F1F5F9', alpha=0.9)
        ax.text(x + 0.015, y + h/2, f"[{sid}] {name}", ha='left', va='center', fontsize=7.5, fontweight='bold', color='#334155')

    def draw_ext(x, y, w, h, name):
        rect = patches.Rectangle((x, y), w, h, facecolor="#F8FAFC", edgecolor="#0F172A", lw=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, name, ha='center', va='center', fontsize=8.5, fontweight='bold', color='#0F172A')

    # External entities
    draw_ext(0.02, 0.72, 0.16, 0.14, "External NewsAPI\nService")
    draw_ext(0.82, 0.72, 0.16, 0.14, "Procurement Risk\nAnalyst")
    draw_ext(0.82, 0.12, 0.16, 0.14, "FastAPI / Streamlit\nDashboard")

    # Processes
    draw_proc(0.24, 0.72, 0.15, 0.14, "P1.0", "Ingest &\nDeduplicate")
    draw_proc(0.44, 0.72, 0.15, 0.14, "P2.0", "Local NLP &\nEntity Linking")
    draw_proc(0.64, 0.72, 0.15, 0.14, "P3.0", "Hetero Graph\nBuilder")

    draw_proc(0.64, 0.40, 0.15, 0.14, "P4.0", "Geospatial &\nDet Risk Engine")
    draw_proc(0.44, 0.40, 0.15, 0.14, "P5.0", "GNN Shock\nPropagation")
    draw_proc(0.24, 0.40, 0.15, 0.14, "P6.0", "Tri-Model Alert\nSynthesizer")

    # Data stores
    draw_store(0.04, 0.43, 0.16, 0.08, "D1", "Master Data CSVs")
    draw_store(0.40, 0.15, 0.20, 0.08, "D2", "Graph Repositories")
    draw_store(0.64, 0.15, 0.16, 0.08, "D3", "Alerts & Events DB")

    # Connecting arrows
    def arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#2563EB", lw=1.2, mutation_scale=10))
        if label:
            mx, my = (x1 + x2)/2, (y1 + y2)/2
            ax.text(mx, my + 0.015, label, ha='center', va='bottom', fontsize=7, color='#1E40AF', fontweight='medium')

    arrow(0.18, 0.79, 0.24, 0.79, "Raw Articles")
    arrow(0.39, 0.79, 0.44, 0.79, "Clean Articles")
    arrow(0.59, 0.79, 0.64, 0.79, "Events & Links")
    arrow(0.71, 0.72, 0.71, 0.54, "Graph Network")
    arrow(0.64, 0.47, 0.59, 0.47, "Det Risk & Geo")
    arrow(0.44, 0.47, 0.39, 0.47, "GNN Embeddings")
    arrow(0.31, 0.40, 0.31, 0.22, "Alerts Record")
    arrow(0.31, 0.22, 0.64, 0.20)
    arrow(0.31, 0.47, 0.82, 0.19, "Live Alerts Stream")
    arrow(0.82, 0.79, 0.79, 0.79, "Query / Triggers")

    ax.set_title("BDS-35 Data Flow Diagram (DFD Level 1)", fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "dfd_diagram.png"), bbox_inches='tight')
    plt.close(fig)
    print("Generated dfd_diagram.png")

# -------------------------------------------------------------
# 4. CLASS DIAGRAM
# -------------------------------------------------------------
def make_class_diagram():
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    ax.axis('off')

    def draw_class(x, y, w, h, name, attrs, methods):
        rect = patches.Rectangle((x, y), w, h, facecolor="#F8FAFC", edgecolor="#1E293B", lw=1.2)
        ax.add_patch(rect)
        # Title
        ax.fill_between([x, x + w], [y + h - 0.05, y + h - 0.05], [y + h, y + h], color="#E2E8F0")
        ax.text(x + w/2, y + h - 0.028, name, ha='center', va='center', fontsize=8.5, fontweight='bold', color='#0F172A')
        # Attrs line
        h_attr = len(attrs) * 0.024 + 0.02
        line1_y = y + h - 0.05
        ax.plot([x, x + w], [line1_y, line1_y], color='#1E293B', lw=1)
        ax.text(x + 0.015, line1_y - 0.015, "\n".join(attrs), ha='left', va='top', fontsize=7, family='monospace', color='#334155')
        # Methods line
        line2_y = line1_y - h_attr
        ax.plot([x, x + w], [line2_y, line2_y], color='#1E293B', lw=1)
        ax.text(x + 0.015, line2_y - 0.015, "\n".join(methods), ha='left', va='top', fontsize=7, family='monospace', color='#0F766E')

    # Classes
    draw_class(0.04, 0.58, 0.28, 0.35, "NewsIngestionEngine", [
        "- api_key: str",
        "- topic: str",
        "- window_hours: int"
    ], [
        "+ ingest_live_news()",
        "+ deduplicate_articles()",
        "+ calculate_relevance()"
    ])

    draw_class(0.36, 0.58, 0.28, 0.35, "LocalNLPEngine", [
        "- spacy_nlp: Language",
        "- event_keywords: dict",
        "- thresholds: float"
    ], [
        "+ extract_event()",
        "+ extract_entities()",
        "+ calculate_severity()"
    ])

    draw_class(0.68, 0.58, 0.28, 0.35, "EntityLinker", [
        "- master_suppliers: DataFrame",
        "- master_locations: DataFrame",
        "- fuzzy_cutoff: float = 0.85"
    ], [
        "+ link_exact()",
        "+ link_normalized()",
        "+ link_fuzzy()",
        "+ resolve_entity()"
    ])

    draw_class(0.04, 0.10, 0.28, 0.38, "HeterogeneousGraphBuilder", [
        "- nx_graph: MultiDiGraph",
        "- pyg_data: HeteroData",
        "- pruned_edges_count: int"
    ], [
        "+ build_from_dataframes()",
        "+ validate_and_prune()",
        "+ export_statistics()",
        "+ to_pyg_heterodata()"
    ])

    draw_class(0.36, 0.10, 0.28, 0.38, "GNNPropagationEngine", [
        "- sage_model: GraphSAGE",
        "- gat_model: GAT",
        "- weights_dir: Path"
    ], [
        "+ load_checkpoints()",
        "+ forward_sage(x, edge_index)",
        "+ forward_gat(x, edge_index)",
        "+ score_supply_chain()"
    ])

    draw_class(0.68, 0.10, 0.28, 0.38, "TriModelRiskSynthesizer", [
        "- alpha: float = 0.50",
        "- beta: float = 0.25",
        "- gamma: float = 0.25"
    ], [
        "+ compute_deterministic_risk()",
        "+ compute_haversine_exposure()",
        "+ blend_risk_scores()",
        "+ generate_explanation()"
    ])

    # Connecting relations
    ax.annotate("", xy=(0.36, 0.75), xytext=(0.32, 0.75), arrowprops=dict(arrowstyle="->", color='#475569', lw=1.2))
    ax.annotate("", xy=(0.68, 0.75), xytext=(0.64, 0.75), arrowprops=dict(arrowstyle="->", color='#475569', lw=1.2))
    ax.annotate("", xy=(0.18, 0.48), xytext=(0.18, 0.58), arrowprops=dict(arrowstyle="->", color='#475569', lw=1.2))
    ax.annotate("", xy=(0.36, 0.28), xytext=(0.32, 0.28), arrowprops=dict(arrowstyle="->", color='#475569', lw=1.2))
    ax.annotate("", xy=(0.68, 0.28), xytext=(0.64, 0.28), arrowprops=dict(arrowstyle="->", color='#475569', lw=1.2))

    ax.set_title("BDS-35 Object-Oriented System Class Diagram", fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "class_diagram.png"), bbox_inches='tight')
    plt.close(fig)
    print("Generated class_diagram.png")

# -------------------------------------------------------------
# 5. SEQUENCE DIAGRAM
# -------------------------------------------------------------
def make_sequence_diagram():
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    ax.axis('off')

    actors = ["User/API", "Pipeline Controller", "News & NLP", "Graph Engine", "GNN Models", "Alert Engine"]
    xs = np.linspace(0.08, 0.92, len(actors))
    y_top = 0.88
    y_bot = 0.06

    for x, act in zip(xs, actors):
        # Header box
        rect = patches.FancyBboxPatch((x - 0.07, y_top), 0.14, 0.07, boxstyle="round,pad=0.01",
                                      facecolor="#E0E7FF", edgecolor="#4338CA", lw=1.2)
        ax.add_patch(rect)
        ax.text(x, y_top + 0.035, act, ha='center', va='center', fontsize=8, fontweight='bold', color='#312E81')
        # Lifeline
        ax.plot([x, x], [y_top, y_bot], color='#94A3B8', linestyle='--', lw=1)

    steps = [
        (0, 1, 0.82, "POST /pipeline/run-live", True),
        (1, 2, 0.74, "1. Fetch & Extract Events/Entities", True),
        (2, 2, 0.67, "Local spaCy & 4-Stage Linking", False),
        (2, 1, 0.60, "Return Linked Entities & Severity", True),
        (1, 3, 0.52, "2. Update Heterogeneous Graph", True),
        (3, 1, 0.44, "Return Node/Edge Statistics", True),
        (1, 4, 0.36, "3. Run GraphSAGE & GAT Inference", True),
        (4, 1, 0.28, "Return Structural Propagation Scores", True),
        (1, 5, 0.20, "4. Blend Combined Risk & Explanations", True),
        (5, 1, 0.14, "Return Synthesized Early Warning Alerts", True),
        (1, 0, 0.08, "HTTP 200 OK (JSON Alert Summary)", True)
    ]

    for a1, a2, y, msg, is_arrow in steps:
        x1, x2 = xs[a1], xs[a2]
        if is_arrow:
            ax.annotate("", xy=(x2, y), xytext=(x1, y),
                        arrowprops=dict(arrowstyle="->", color="#2563EB", lw=1.2, mutation_scale=10))
            ax.text((x1 + x2)/2, y + 0.015, msg, ha='center', va='bottom', fontsize=7.5, color='#0F172A', fontweight='medium')
        else:
            # Self-call loop
            ax.plot([x1, x1 + 0.05, x1 + 0.05, x1], [y + 0.02, y + 0.02, y - 0.02, y - 0.02], color="#D97706", lw=1.2)
            ax.annotate("", xy=(x1, y - 0.02), xytext=(x1 + 0.01, y - 0.02), arrowprops=dict(arrowstyle="->", color="#D97706"))
            ax.text(x1 + 0.06, y, msg, ha='left', va='center', fontsize=7, color='#92400E', style='italic')

    ax.set_title("BDS-35 Live Execution Sequence Diagram", fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "sequence_diagram.png"), bbox_inches='tight')
    plt.close(fig)
    print("Generated sequence_diagram.png")

# -------------------------------------------------------------
# 6. STATE CHART DIAGRAM
# -------------------------------------------------------------
def make_state_chart():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.axis('off')

    def draw_state(x, y, w, h, name, desc):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                      facecolor="#FEF3C7", edgecolor="#D97706", lw=1.3)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + 0.015, name, ha='center', va='center', fontsize=8.5, fontweight='bold', color='#78350F')
        ax.text(x + w/2, y + h/2 - 0.02, desc, ha='center', va='center', fontsize=7, color='#92400E', style='italic')

    # Start and End
    circle_start = patches.Circle((0.08, 0.5), 0.025, facecolor="#0F172A", edgecolor="#0F172A")
    ax.add_patch(circle_start)

    circle_end_out = patches.Circle((0.92, 0.5), 0.025, facecolor="none", edgecolor="#0F172A", lw=1.5)
    circle_end_in = patches.Circle((0.92, 0.5), 0.018, facecolor="#0F172A", edgecolor="#0F172A")
    ax.add_patch(circle_end_out)
    ax.add_patch(circle_end_in)

    draw_state(0.15, 0.42, 0.13, 0.16, "INGESTED", "Raw Article JSON")
    draw_state(0.33, 0.42, 0.13, 0.16, "VALIDATED", "Dedup & Relevance OK")
    draw_state(0.51, 0.42, 0.13, 0.16, "EXTRACTED", "NLP Events & Links")
    draw_state(0.69, 0.42, 0.13, 0.16, "SCORED", "GNN & Det Blended")

    # Transitions
    def transition(x1, y1, x2, y2, label):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#2563EB", lw=1.3, mutation_scale=10))
        ax.text((x1 + x2)/2, y1 + 0.02, label, ha='center', va='bottom', fontsize=7, color='#1E40AF', fontweight='medium')

    transition(0.105, 0.5, 0.15, 0.5, "Live Fetch")
    transition(0.28, 0.5, 0.33, 0.5, "Relevance >= 0.20")
    transition(0.46, 0.5, 0.51, 0.5, "Confidence >= 0.85")
    transition(0.64, 0.5, 0.69, 0.5, "Graph Propagated")
    transition(0.82, 0.5, 0.895, 0.5, "Alert Synthesized")

    # Drop state for duplicates/irrelevant
    rect_drop = patches.FancyBboxPatch((0.33, 0.12), 0.13, 0.14, boxstyle="round,pad=0.02",
                                       facecolor="#FEE2E2", edgecolor="#DC2626", lw=1.2)
    ax.add_patch(rect_drop)
    ax.text(0.395, 0.19, "DROPPED / ARCHIVED\n(Duplicate/Irrelevant)", ha='center', va='center', fontsize=7, fontweight='bold', color='#991B1B')
    
    ax.annotate("", xy=(0.395, 0.26), xytext=(0.215, 0.42),
                arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.2, connectionstyle="arc3,rad=-0.2"))
    ax.text(0.25, 0.31, "SimHash / Score Fail", fontsize=6.5, color='#B91C1C')

    ax.set_title("BDS-35 Disruption Lifecycle State Chart Diagram", fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "state_chart_diagram.png"), bbox_inches='tight')
    plt.close(fig)
    print("Generated state_chart_diagram.png")

# -------------------------------------------------------------
# 7. USE CASE DIAGRAM
# -------------------------------------------------------------
def make_use_case():
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    ax.axis('off')

    # System boundary box
    sys_box = patches.Rectangle((0.26, 0.05), 0.48, 0.88, facecolor="#F8FAFC", edgecolor="#475569", lw=1.5, linestyle="--")
    ax.add_patch(sys_box)
    ax.text(0.50, 0.90, "BDS-35 Early Warning Platform", ha='center', va='center', fontsize=10, fontweight='bold', color='#0F172A')

    # Actors
    def draw_actor(x, y, name):
        # Head
        c = patches.Circle((x, y + 0.05), 0.025, facecolor="#F1F5F9", edgecolor="#0F172A", lw=1.5)
        ax.add_patch(c)
        # Body
        ax.plot([x, x], [y + 0.025, y - 0.03], color='#0F172A', lw=1.5)
        # Arms
        ax.plot([x - 0.03, x + 0.03], [y + 0.005, y + 0.005], color='#0F172A', lw=1.5)
        # Legs
        ax.plot([x, x - 0.025], [y - 0.03, y - 0.07], color='#0F172A', lw=1.5)
        ax.plot([x, x + 0.025], [y - 0.03, y - 0.07], color='#0F172A', lw=1.5)
        # Label
        ax.text(x, y - 0.10, name, ha='center', va='top', fontsize=8, fontweight='bold', color='#0F172A')

    draw_actor(0.12, 0.65, "Procurement\nRisk Analyst")
    draw_actor(0.12, 0.25, "System Administrator\n/ Data Engineer")
    draw_actor(0.88, 0.50, "NewsAPI Gateway\n/ External Data")

    # Use cases
    use_cases = [
        (0.50, 0.80, "Trigger Live Pipeline Run"),
        (0.50, 0.68, "Triage & Filter Risk Alerts"),
        (0.50, 0.56, "Inspect Ego-Network Graphs"),
        (0.50, 0.44, "Log Analyst Feedback (Human-in-Loop)"),
        (0.50, 0.32, "Audit Model Weights & Drift"),
        (0.50, 0.20, "Manage Supplier Master Catalog"),
        (0.50, 0.08, "Poll Automated Live Feeds")
    ]

    for cx, cy, uc_name in use_cases:
        ellipse = patches.Ellipse((cx, cy), 0.38, 0.08, facecolor="#EFF6FF", edgecolor="#2563EB", lw=1.2)
        ax.add_patch(ellipse)
        ax.text(cx, cy, uc_name, ha='center', va='center', fontsize=7.5, fontweight='bold', color='#1E40AF')

    # Connecting lines
    analyst_lines = [0.80, 0.68, 0.56, 0.44]
    for uy in analyst_lines:
        ax.plot([0.16, 0.31], [0.65, uy], color='#475569', lw=1)

    admin_lines = [0.32, 0.20]
    for uy in admin_lines:
        ax.plot([0.16, 0.31], [0.25, uy], color='#475569', lw=1)

    ax.plot([0.84, 0.69], [0.50, 0.80], color='#475569', lw=1)
    ax.plot([0.84, 0.69], [0.50, 0.08], color='#475569', lw=1)

    ax.set_title("BDS-35 Comprehensive Use Case Diagram", fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "use_case_diagram.png"), bbox_inches='tight')
    plt.close(fig)
    print("Generated use_case_diagram.png")

if __name__ == "__main__":
    make_gantt()
    make_er()
    make_dfd()
    make_class_diagram()
    make_sequence_diagram()
    make_state_chart()
    make_use_case()
    print("All 7 architectural diagrams successfully generated!")
