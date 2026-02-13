"""Interactive knowledge graph visualization using PyVis.

## RAG Theory: Knowledge Graph Visualization

A knowledge graph captures entities (nodes) and relationships (edges)
extracted from documents. Visualizing this graph reveals community
structure, hub entities, and domain clusters that text alone cannot
convey. Interactive visualization lets users explore these structures
by zooming, dragging, and hovering for entity metadata.

## Library Usage

- **PyVis** (wraps vis.js): Produces interactive HTML files with
  zoom, drag, hover. `cdn_resources="remote"` loads vis.js from
  jsDelivr CDN, keeping HTML files small and compatible with
  htmlpreview.github.io (which blocks large inline scripts).
- **NetworkX**: Provides the graph data structure and algorithms
  (degree centrality, subgraph extraction) that PyVis renders.
- **Neo4j driver**: Fetches entities and relationships via Cypher
  queries from the knowledge graph database.

## Data Flow

1. Cypher queries fetch entities (with pagerank, degree, community_id)
   and relationships (with type, weight, description) from Neo4j.
2. NetworkX DiGraph is built from the raw dicts (nodes keyed by
   normalized_name, edges from source to target).
3. Optional filtering selects top-N nodes by degree or pagerank,
   preserving all inter-edges between selected nodes.
4. PyVis renders the NetworkX graph with color-by-type, size-by-degree,
   and hover tooltips showing entity metadata.
5. HTML legend is injected via string post-processing (PyVis has no
   native legend support).
"""

import argparse
from pathlib import Path

import networkx as nx
from neo4j import Driver
from pyvis.network import Network

from src.config import PROJECT_ROOT
from src.graph.neo4j_client import get_driver, verify_connection
from src.shared.files import setup_logging

logger = setup_logging(__name__)

# Entity type color map — 9 distinct colors matching graphrag_types.yaml
ENTITY_TYPE_COLORS = {
    "PERSON": "#E63946",
    "BRAIN_STRUCTURE": "#457B9D",
    "BRAIN_FUNCTION": "#1D3557",
    "CHEMICAL": "#2A9D8F",
    "DISORDER": "#E9C46A",
    "MENTAL_STATE": "#F4A261",
    "BEHAVIOR": "#264653",
    "THEORY": "#A8DADC",
    "PRECEPT": "#6A4C93",
}

DEFAULT_COLOR = "#999999"


def fetch_graph_from_neo4j(
    driver: Driver,
) -> tuple[list[dict], list[dict]]:
    """Fetch all entities and relationships from Neo4j.

    Runs two Cypher queries: one for entity nodes (with graph metrics)
    and one for relationship edges (with weight and description).

    Args:
        driver: Neo4j driver instance.

    Returns:
        Tuple of (entities, relationships) where each is a list of dicts.

    Raises:
        Exception: If Neo4j queries fail.
    """
    entity_query = """
    MATCH (e:Entity)
    RETURN
        e.name AS name,
        e.normalized_name AS normalized_name,
        e.entity_type AS entity_type,
        e.description AS description,
        e.community_id AS community_id,
        coalesce(e.degree, 0) AS degree,
        coalesce(e.pagerank, 0.0) AS pagerank
    """
    entity_result = driver.execute_query(entity_query)
    entities = [dict(r) for r in entity_result.records]
    logger.info(f"Fetched {len(entities)} entities from Neo4j")

    rel_query = """
    MATCH (s:Entity)-[r:RELATED_TO]->(t:Entity)
    RETURN
        s.normalized_name AS source,
        t.normalized_name AS target,
        r.type AS type,
        r.description AS description,
        coalesce(r.weight, 1.0) AS weight
    """
    rel_result = driver.execute_query(rel_query)
    relationships = [dict(r) for r in rel_result.records]
    logger.info(f"Fetched {len(relationships)} relationships from Neo4j")

    return entities, relationships


def build_networkx_graph(
    entities: list[dict],
    relationships: list[dict],
) -> nx.DiGraph:
    """Build a NetworkX directed graph from entity and relationship dicts.

    Nodes are keyed by normalized_name with all entity properties as
    node attributes. Edges carry type, description, and weight.

    Args:
        entities: List of entity dicts from fetch_graph_from_neo4j.
        relationships: List of relationship dicts from fetch_graph_from_neo4j.

    Returns:
        NetworkX DiGraph with entity nodes and relationship edges.
    """
    G = nx.DiGraph()

    for entity in entities:
        G.add_node(
            entity["normalized_name"],
            name=entity.get("name", entity["normalized_name"]),
            entity_type=entity.get("entity_type", "UNKNOWN"),
            description=entity.get("description", ""),
            community_id=entity.get("community_id"),
            degree=entity.get("degree", 0),
            pagerank=entity.get("pagerank", 0.0),
        )

    for rel in relationships:
        source = rel["source"]
        target = rel["target"]
        # Only add edge if both nodes exist
        if source in G and target in G:
            G.add_edge(
                source,
                target,
                type=rel.get("type", "RELATED_TO"),
                description=rel.get("description", ""),
                weight=rel.get("weight", 1.0),
            )

    logger.info(
        f"Built NetworkX graph: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges"
    )
    return G


def filter_top_nodes(
    G: nx.DiGraph,
    top_n: int = 15,
    metric: str = "degree",
) -> nx.DiGraph:
    """Filter graph to top-N nodes by a given metric.

    Sorts nodes by the specified metric (degree or pagerank) in
    descending order, takes the top N, and returns the induced
    subgraph preserving all edges between selected nodes.

    Args:
        G: Full NetworkX DiGraph.
        top_n: Number of top nodes to keep.
        metric: Node attribute to sort by ("degree" or "pagerank").

    Returns:
        Induced subgraph with top-N nodes and their inter-edges.
    """
    sorted_nodes = sorted(
        G.nodes(),
        key=lambda n: G.nodes[n].get(metric, 0),
        reverse=True,
    )
    top_nodes = sorted_nodes[:top_n]
    subgraph = G.subgraph(top_nodes).copy()

    logger.info(
        f"Filtered to top {top_n} nodes by {metric}: "
        f"{subgraph.number_of_nodes()} nodes, "
        f"{subgraph.number_of_edges()} edges"
    )
    return subgraph


def _build_legend_html() -> str:
    """Build an HTML legend div for entity type colors.

    Returns:
        HTML string with a fixed-position CSS legend.
    """
    items = []
    for entity_type, color in ENTITY_TYPE_COLORS.items():
        label = entity_type.replace("_", " ").title()
        items.append(
            f'<div style="display:flex;align-items:center;margin:3px 0;">'
            f'<span style="display:inline-block;width:14px;height:14px;'
            f"background:{color};border-radius:50%;margin-right:8px;"
            f'flex-shrink:0;"></span>'
            f'<span style="font-size:12px;color:#333;">{label}</span>'
            f"</div>"
        )

    return (
        '<div id="graph-legend" style="'
        "position:fixed;top:10px;right:10px;background:rgba(255,255,255,0.95);"
        "border:1px solid #ddd;border-radius:8px;padding:12px 16px;"
        'box-shadow:0 2px 8px rgba(0,0,0,0.1);z-index:1000;font-family:sans-serif;">'
        '<div style="font-weight:bold;font-size:13px;margin-bottom:8px;'
        'color:#222;border-bottom:1px solid #eee;padding-bottom:6px;">Entity Types</div>'
        + "\n".join(items)
        + "</div>"
    )


def build_pyvis_network(
    G: nx.DiGraph,
    title: str,
    height: str = "900px",
) -> Network:
    """Build a PyVis interactive network from a NetworkX graph.

    Configures node colors by entity type, node sizes by degree
    (min-max scaled to 10-50px), hover tooltips with entity metadata,
    edge widths from weight, and Barnes-Hut physics for layout.

    Args:
        G: NetworkX DiGraph to visualize.
        title: Title displayed on the HTML page.
        height: CSS height for the visualization canvas.

    Returns:
        Configured PyVis Network ready for HTML export.
    """
    net = Network(
        height=height,
        width="100%",
        directed=True,
        cdn_resources="remote",
        notebook=False,
    )

    # Compute min-max scaling for node sizes
    degrees = [G.nodes[n].get("degree", 0) for n in G.nodes()]
    min_deg = min(degrees) if degrees else 0
    max_deg = max(degrees) if degrees else 1
    deg_range = max_deg - min_deg if max_deg != min_deg else 1

    hide_edge_labels = G.number_of_nodes() > 500

    # Add nodes
    for node_id in G.nodes():
        attrs = G.nodes[node_id]
        entity_type = attrs.get("entity_type", "UNKNOWN")
        color = ENTITY_TYPE_COLORS.get(entity_type, DEFAULT_COLOR)

        # Scale degree to node size (30-90px)
        degree = attrs.get("degree", 0)
        size = 30 + 60 * ((degree - min_deg) / deg_range)

        # Build hover tooltip
        name = attrs.get("name", node_id)
        description = attrs.get("description", "")
        if len(description) > 200:
            description = description[:200] + "..."

        tooltip = description

        net.add_node(
            node_id,
            label=name,
            title=tooltip,
            color=color,
            size=size,
        )

    # Add edges
    for source, target, data in G.edges(data=True):
        weight = data.get("weight", 1.0)
        edge_type = data.get("type", "RELATED_TO")
        description = data.get("description", "")
        if len(description) > 150:
            description = description[:150] + "..."

        edge_tooltip = f"{edge_type}\n{description}"

        edge_kwargs = {
            "width": max(0.5, min(weight, 5.0)),
            "title": edge_tooltip,
            "color": "#888888",
            "arrows": "to",
        }
        if not hide_edge_labels:
            edge_kwargs["label"] = edge_type

        net.add_edge(source, target, **edge_kwargs)

    # Configure physics
    # Strategy: strong repulsion + wide springs for initial layout,
    # then disable physics after stabilization so dragged nodes stay put.
    net.set_options("""
    {
        "physics": {
            "solver": "barnesHut",
            "barnesHut": {
                "gravitationalConstant": -100000,
                "centralGravity": 0.0,
                "springLength": 600,
                "springConstant": 0.003,
                "damping": 0.5,
                "avoidOverlap": 1.0
            },
            "stabilization": {
                "iterations": 400,
                "fit": true
            }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "navigationButtons": true
        },
        "edges": {
            "smooth": {
                "type": "continuous"
            },
            "font": {
                "background": "white",
                "size": 11
            }
        }
    }
    """)

    return net


def export_graph(
    output_dir: Path = None,
    top_n: int = 15,
    metric: str = "degree",
    include_full: bool = False,
) -> dict[str, Path]:
    """Export interactive knowledge graph visualizations to HTML files.

    Main entry point. Connects to Neo4j, fetches the graph data,
    builds a filtered visualization (and optionally a full one),
    and writes HTML files that load vis.js from CDN.

    Args:
        output_dir: Directory for HTML output. Defaults to
            PROJECT_ROOT / "docs" / "visualizations".
        top_n: Number of top nodes for the filtered graph.
        metric: Node attribute for filtering ("degree" or "pagerank").
        include_full: If True, also generate the full graph HTML
            (can be very large for 60K+ node graphs).

    Returns:
        Dict mapping file names to their output Paths.

    Raises:
        Exception: If Neo4j connection or queries fail.
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "docs" / "visualizations"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Connect to Neo4j
    driver = get_driver()
    verify_connection(driver)

    try:
        # Fetch graph data
        entities, relationships = fetch_graph_from_neo4j(driver)

        if not entities:
            logger.warning("No entities found in Neo4j. Is the graph populated?")
            return {}

        # Build full graph
        G = build_networkx_graph(entities, relationships)

        # Generate legend HTML
        legend_html = _build_legend_html()

        outputs = {}

        # Full graph (opt-in, large graphs produce huge HTML files)
        if include_full:
            logger.info("Generating full graph visualization...")
            net_full = build_pyvis_network(G, title="RAGLab Knowledge Graph (Full)")
            full_path = output_dir / "graph_full.html"
            net_full.save_graph(str(full_path))
            _inject_legend(full_path, legend_html)
            _inject_freeze_on_stabilize(full_path)
            outputs["graph_full.html"] = full_path
            logger.info(f"Saved full graph: {full_path}")

        # Filtered graph
        logger.info(
            f"Generating filtered graph (top {top_n} by {metric})..."
        )
        G_filtered = filter_top_nodes(G, top_n=top_n, metric=metric)
        net_filtered = build_pyvis_network(
            G_filtered,
            title=f"RAGLab Knowledge Graph (Top {top_n} by {metric})",
        )
        filtered_path = output_dir / "graph_filtered.html"
        net_filtered.save_graph(str(filtered_path))
        _inject_legend(filtered_path, legend_html)
        _inject_freeze_on_stabilize(filtered_path)
        outputs["graph_filtered.html"] = filtered_path
        logger.info(f"Saved filtered graph: {filtered_path}")

        logger.info(
            f"Export complete: {len(outputs)} files in {output_dir}"
        )
        return outputs

    finally:
        driver.close()


def _inject_legend(html_path: Path, legend_html: str) -> None:
    """Inject a legend div into a PyVis-generated HTML file.

    PyVis has no native legend support, so this post-processes the
    HTML to insert a fixed-position legend before the closing body tag.

    Args:
        html_path: Path to the HTML file to modify.
        legend_html: HTML string for the legend div.
    """
    content = html_path.read_text(encoding="utf-8")
    content = content.replace("</body>", f"{legend_html}\n</body>")
    html_path.write_text(content, encoding="utf-8")


def _inject_freeze_on_stabilize(html_path: Path) -> None:
    """Disable physics after the layout stabilizes.

    Once vis.js finishes computing the initial layout, this script
    turns physics off entirely. The result: dragged nodes stay exactly
    where the user places them — no spring pull-back, no gravity drift.

    Args:
        html_path: Path to the PyVis-generated HTML file.
    """
    script = """
<script type="text/javascript">
    // After stabilization, freeze layout so drag = permanent placement
    network.on("stabilizationIterationsDone", function() {
        network.setOptions({ physics: { enabled: false } });
    });
</script>
"""
    content = html_path.read_text(encoding="utf-8")
    content = content.replace("</body>", f"{script}\n</body>")
    html_path.write_text(content, encoding="utf-8")


def main():
    """CLI entry point for knowledge graph visualization export.

    Usage:
        python -m src.visualization.graph_export
        python -m src.visualization.graph_export --top-n 100 --metric pagerank
        python -m src.visualization.graph_export --output-dir /tmp/graphs
    """
    parser = argparse.ArgumentParser(
        description="Export interactive knowledge graph visualizations from Neo4j."
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of top nodes for the filtered graph (default: 15).",
    )
    parser.add_argument(
        "--metric",
        choices=["degree", "pagerank"],
        default="degree",
        help="Metric for filtering top nodes (default: degree).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for HTML files (default: docs/visualizations/).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also generate full graph HTML (can be very large).",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None

    outputs = export_graph(
        output_dir=output_dir,
        top_n=args.top_n,
        metric=args.metric,
        include_full=args.full,
    )

    if outputs:
        logger.info("Generated files:")
        for name, path in outputs.items():
            logger.info(f"  {name}: {path}")
    else:
        logger.warning("No files generated. Check Neo4j connection and data.")


if __name__ == "__main__":
    main()
