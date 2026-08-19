"""
pages/5_Asset_Connectivity.py

Interactive version of Task 5. The pyvis graph responds to a site
selector, and the query panel below implements the exact example queries
called out in the brief (connected-to, downstream-impact, assets-under-site,
isolated-assets) against the same networkx graph used in
src/connectivity_analysis.py - nothing here is re-derived by hand.
"""

import sys
import tempfile
from pathlib import Path

import networkx as nx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.ui import page_setup, ASSET_TYPE_COLORS
from utils.pipeline import get_graph, load_raw_data
import connectivity_analysis as ca

page_setup("Asset Connectivity Analysis", icon="🔗")

st.caption(
    "Assets and their parent/child relationships modeled as a directed graph. "
    "Edges point from parent to child - i.e. the direction a failure propagates."
)

telemetry, assets, connectivity = load_raw_data()
G, dq = get_graph()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Assets", G.number_of_nodes())
c2.metric("Connections", G.number_of_edges())
c3.metric("Isolated assets", len(list(nx.isolates(G))))
c4.metric("Data quality issues",
          len(dq["duplicate_connections"]) + len(dq["assets_with_invalid_parent"]))

st.divider()
tab1, tab2, tab3 = st.tabs(["Dependency graph", "Run a query", "Data quality report"])

# ---- tab 1: visual graph ----
with tab1:
    site_choice = st.selectbox("Site", sorted(assets.site_id.unique()))
    sub_nodes = assets[assets.site_id == site_choice].asset_id.tolist()
    subG = G.subgraph(sub_nodes)

    net = Network(height="600px", width="100%", directed=True, bgcolor="#ffffff", font_color="#222222")
    net.barnes_hut(gravity=-4000, spring_length=120)

    for node in subG.nodes:
        atype = subG.nodes[node].get("asset_type", "")
        name = subG.nodes[node].get("asset_name", node)
        color = ASSET_TYPE_COLORS.get(atype, "#999999")
        net.add_node(node, label=node, title=f"{name} ({atype})", color=color, size=18)

    for u, v, data in subG.edges(data=True):
        net.add_edge(u, v, title=data.get("connection_type", ""), arrows="to", color="#aaaaaa")

    html_path = Path(tempfile.gettempdir()) / "asset_graph.html"
    net.write_html(str(html_path), notebook=False)
    components.html(html_path.read_text(encoding="utf-8"), height=620, scrolling=False)

    legend = "  ".join(f"<span style='color:{c}'>●</span> {t}" for t, c in ASSET_TYPE_COLORS.items())
    st.markdown(legend, unsafe_allow_html=True)

# ---- tab 2: live query builder ----
with tab2:
    query_type = st.selectbox(
        "Query",
        [
            "Show all assets connected to <asset>",
            "Identify downstream assets impacted if <asset> fails",
            "List all assets under <site>",
            "Find isolated assets (no parent or child)",
        ],
    )

    if query_type == "Show all assets connected to <asset>":
        asset_choice = st.selectbox("Asset", sorted(G.nodes))
        result = sorted(set(G.predecessors(asset_choice)) | set(G.successors(asset_choice)))
        st.write(f"**Assets connected to `{asset_choice}`:**")
        st.write(result if result else "None - this asset has no connections.")

    elif query_type == "Identify downstream assets impacted if <asset> fails":
        asset_choice = st.selectbox("Asset", sorted(G.nodes))
        result = sorted(nx.descendants(G, asset_choice))
        st.write(f"**If `{asset_choice}` fails, these {len(result)} downstream assets are impacted:**")
        if result:
            impacted = assets[assets.asset_id.isin(result)][["asset_id", "asset_name", "asset_type"]]
            st.dataframe(impacted, use_container_width=True, hide_index=True)
            by_type = impacted.asset_type.value_counts()
            st.caption("Breakdown: " + ", ".join(f"{v} {k}" for k, v in by_type.items()))
        else:
            st.write("No downstream assets - this is a leaf node in the hierarchy.")

    elif query_type == "List all assets under <site>":
        site_choice2 = st.selectbox("Site", sorted(assets.site_id.unique()))
        result_df = assets[assets.site_id == site_choice2][["asset_id", "asset_name", "asset_type", "building_id"]]
        st.write(f"**{len(result_df)} assets under `{site_choice2}`:**")
        st.dataframe(result_df, use_container_width=True, hide_index=True)

    else:
        isolated = sorted(set(nx.isolates(G)) | set(dq["orphan_assets"]))
        st.write(f"**{len(isolated)} isolated / orphan assets found:**")
        st.write(isolated if isolated else "None - every asset has at least one connection.")

# ---- tab 3: data quality ----
with tab3:
    st.subheader("Duplicate connections")
    if dq["duplicate_connections"]:
        st.dataframe(pd.DataFrame(dq["duplicate_connections"]), use_container_width=True, hide_index=True)
    else:
        st.write("None found.")

    st.subheader("Invalid parent references")
    if dq["assets_with_invalid_parent"]:
        st.dataframe(pd.DataFrame(dq["assets_with_invalid_parent"]), use_container_width=True, hide_index=True)
    else:
        st.write("None found.")

    st.subheader("Orphan assets (no parent, and nothing lists them as a parent)")
    st.write(dq["orphan_assets"] if dq["orphan_assets"] else "None found.")

    st.subheader("Chiller failure blast radius")
    impact_rows = []
    for _, row in assets[assets.asset_type == "Chiller"].iterrows():
        downstream = ca.downstream_impact(G, row.asset_id)
        impact_rows.append({"asset_id": row.asset_id, "site_id": row.site_id,
                             "downstream_asset_count": len(downstream)})
    st.dataframe(pd.DataFrame(impact_rows), use_container_width=True, hide_index=True)
    st.caption("Chillers sit at the top of the hierarchy in every building, so they consistently "
               "have the largest blast radius of any asset type - worth prioritizing in maintenance scheduling.")
