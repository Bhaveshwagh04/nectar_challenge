"""
connectivity_analysis.py - Task 5

Builds the asset dependency graph, runs data quality checks on the
connectivity table, and answers the standard graph queries the brief asks
for (connected assets, downstream impact of a failure, isolated assets).

networkx is used instead of standing up a graph database - for a few hundred
assets a DiGraph in memory is plenty, and it's a much smaller dependency than
running Neo4j/GrapQL for this exercise. Noted in the README as something
that'd change at real production scale (thousands of sites).

Run:
    python src/connectivity_analysis.py
"""

import json
import networkx as nx
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from preprocessing import load_raw

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
PLOT_DIR = OUT_DIR / "plots"


def build_graph(assets, connectivity):
    G = nx.DiGraph()

    for _, row in assets.iterrows():
        G.add_node(row.asset_id, asset_type=row.asset_type, site_id=row.site_id,
                   building_id=row.building_id, asset_name=row.asset_name)

    for _, row in connectivity.iterrows():
        if row.source_asset_id in G.nodes and row.target_asset_id in G.nodes:
            G.add_edge(row.source_asset_id, row.target_asset_id,
                       connection_type=row.connection_type,
                       weight=row.relationship_strength)

    return G


def data_quality_checks(assets, connectivity, G):
    findings = {}

    # duplicate edges
    dupe_mask = connectivity.duplicated(subset=["source_asset_id", "target_asset_id"], keep=False)
    findings["duplicate_connections"] = connectivity[dupe_mask].to_dict("records")

    # edges pointing to/from an asset that doesn't exist in metadata
    valid_ids = set(assets.asset_id)
    bad_source = connectivity[~connectivity.source_asset_id.isin(valid_ids)]
    bad_target = connectivity[~connectivity.target_asset_id.isin(valid_ids)]
    findings["invalid_parent_mappings"] = pd.concat([bad_source, bad_target]).drop_duplicates().to_dict("records")

    # orphan assets - no parent, and nothing else lists them as a parent either
    has_parent = set(assets[assets.parent_asset_id.notna()].asset_id)
    is_parent = set(assets.parent_asset_id.dropna())
    orphans = assets[(~assets.asset_id.isin(has_parent | is_parent))]
    findings["orphan_assets"] = orphans.asset_id.tolist()

    # assets with a parent_asset_id that doesn't actually exist
    invalid_parents = assets[assets.parent_asset_id.notna() & ~assets.parent_asset_id.isin(valid_ids)]
    findings["assets_with_invalid_parent"] = invalid_parents[["asset_id", "parent_asset_id"]].to_dict("records")

    # isolated nodes in the graph itself (no in or out edges)
    isolated = list(nx.isolates(G))
    findings["isolated_nodes_in_graph"] = isolated

    return findings


def downstream_impact(G, asset_id):
    """All assets reachable downstream if `asset_id` fails."""
    if asset_id not in G.nodes:
        return []
    return sorted(nx.descendants(G, asset_id))


def upstream_dependencies(G, asset_id):
    if asset_id not in G.nodes:
        return []
    return sorted(nx.ancestors(G, asset_id))


def query_assets_under_site(assets, site_id):
    return assets[assets.site_id == site_id].asset_id.tolist()


def query_connected_to(G, asset_id):
    if asset_id not in G.nodes:
        return []
    return sorted(set(G.predecessors(asset_id)) | set(G.successors(asset_id)))


def plot_hierarchy(G, assets, site_id, out_path):
    sub_nodes = assets[assets.site_id == site_id].asset_id.tolist()
    subG = G.subgraph(sub_nodes)

    try:
        pos = nx.nx_agraph.graphviz_layout(subG, prog="dot")
    except Exception:
        pos = nx.spring_layout(subG, k=0.9, seed=42)

    type_colors = {
        "Chiller": "#d1495b", "AHU": "#3b6ea5", "Pump": "#5b8c5a",
        "EnergyMeter": "#c17f3e", "EnvSensor": "#8a5b8c",
    }
    node_colors = [type_colors.get(subG.nodes[n].get("asset_type"), "#999999") for n in subG.nodes]

    fig, ax = plt.subplots(figsize=(11, 8))
    nx.draw(subG, pos, ax=ax, with_labels=True, node_color=node_colors, node_size=900,
            font_size=7, arrows=True, edge_color="#888888")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=10, label=t)
               for t, c in type_colors.items()]
    ax.legend(handles=handles, loc="upper right")
    ax.set_title(f"Asset dependency hierarchy - {site_id}")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run():
    telemetry, assets, connectivity = load_raw()
    G = build_graph(assets, connectivity)

    print(f"graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    dq = data_quality_checks(assets, connectivity, G)
    print("\n--- data quality findings ---")
    for k, v in dq.items():
        count = len(v) if isinstance(v, list) else "?"
        print(f"{k}: {count}")

    with open(OUT_DIR / "connectivity_data_quality.json", "w") as f:
        json.dump(dq, f, indent=2, default=str)

    # example queries the brief explicitly asks for
    example_chiller = assets[assets.asset_type == "Chiller"].asset_id.iloc[0]
    example_ahu = assets[assets.asset_type == "AHU"].asset_id.iloc[0]
    example_site = assets.site_id.iloc[0]

    q1 = query_connected_to(G, example_chiller)
    q2 = downstream_impact(G, example_ahu)
    q3 = query_assets_under_site(assets, example_site)
    q4 = list(nx.isolates(G)) + dq["orphan_assets"]

    print(f"\n[query] assets connected to {example_chiller}: {q1}")
    print(f"[query] downstream of {example_ahu} failing: {q2}")
    print(f"[query] assets under {example_site}: {len(q3)} assets")
    print(f"[query] isolated/orphan assets: {sorted(set(q4))}")

    # failure impact analysis on every chiller (the most upstream, highest-blast-radius asset type)
    impact_rows = []
    for _, row in assets[assets.asset_type == "Chiller"].iterrows():
        downstream = downstream_impact(G, row.asset_id)
        downstream_types = assets[assets.asset_id.isin(downstream)].asset_type.value_counts().to_dict()
        impact_rows.append({
            "asset_id": row.asset_id,
            "asset_name": row.asset_name,
            "site_id": row.site_id,
            "downstream_asset_count": len(downstream),
            "downstream_by_type": downstream_types,
        })
    impact_df = pd.DataFrame(impact_rows)
    print("\n--- chiller failure blast radius ---")
    print(impact_df.to_string(index=False))
    impact_df.to_json(OUT_DIR / "chiller_failure_impact.json", orient="records", indent=2)

    # network-level stats
    stats = {
        "num_assets": G.number_of_nodes(),
        "num_connections": G.number_of_edges(),
        "avg_out_degree": sum(dict(G.out_degree()).values()) / G.number_of_nodes(),
        "isolated_asset_count": len(list(nx.isolates(G))),
        "weakly_connected_components": nx.number_weakly_connected_components(G),
    }
    print("\nnetwork stats:", json.dumps(stats, indent=2))
    with open(OUT_DIR / "connectivity_network_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    plot_hierarchy(G, assets, example_site, PLOT_DIR / "13_asset_hierarchy.png")

    return G, dq, impact_df


if __name__ == "__main__":
    run()
