"""Graph schema definitions for the BDS-35 heterogeneous supply-chain graph."""
from __future__ import annotations
from enum import Enum
from typing import NamedTuple


class NodeType(str, Enum):
    """Heterogeneous node types in the BDS-35 supply-chain graph."""

    NEWS = "NEWS"
    EVENT = "EVENT"
    SUPPLIER = "SUPPLIER"
    PRODUCT = "PRODUCT"
    LOCATION = "LOCATION"
    FACILITY = "FACILITY"


class EdgeType(str, Enum):
    """Canonical edge types connecting heterogeneous nodes."""

    NEWS_HAS_EVENT = "NEWS_HAS_EVENT"
    EVENT_AT_LOCATION = "EVENT_AT_LOCATION"
    LOCATION_AFFECTS_FACILITY = "LOCATION_AFFECTS_FACILITY"
    FACILITY_BELONGS_TO_SUPPLIER = "FACILITY_BELONGS_TO_SUPPLIER"
    SUPPLIER_PROVIDES_PRODUCT = "SUPPLIER_PROVIDES_PRODUCT"
    SUPPLIER_DEPENDS_ON_SUPPLIER = "SUPPLIER_DEPENDS_ON_SUPPLIER"
    PRODUCT_DEPENDS_ON_PRODUCT = "PRODUCT_DEPENDS_ON_PRODUCT"


class EdgeConstraint(NamedTuple):
    source_type: NodeType
    edge_type: EdgeType
    target_type: NodeType


# Canonical edge constraints: (source_node_type, edge_type, target_node_type)
CANONICAL_EDGE_CONSTRAINTS: dict[EdgeType, tuple[NodeType, NodeType]] = {
    EdgeType.NEWS_HAS_EVENT: (NodeType.NEWS, NodeType.EVENT),
    EdgeType.EVENT_AT_LOCATION: (NodeType.EVENT, NodeType.LOCATION),
    EdgeType.LOCATION_AFFECTS_FACILITY: (NodeType.LOCATION, NodeType.FACILITY),
    EdgeType.FACILITY_BELONGS_TO_SUPPLIER: (NodeType.FACILITY, NodeType.SUPPLIER),
    EdgeType.SUPPLIER_PROVIDES_PRODUCT: (NodeType.SUPPLIER, NodeType.PRODUCT),
    EdgeType.SUPPLIER_DEPENDS_ON_SUPPLIER: (NodeType.SUPPLIER, NodeType.SUPPLIER),
    EdgeType.PRODUCT_DEPENDS_ON_PRODUCT: (NodeType.PRODUCT, NodeType.PRODUCT),
}

NODE_COLUMNS = ["node_id", "node_type", "label", "source", "data_status"]
EDGE_COLUMNS = ["source", "target", "edge_type", "weight", "data_status"]
