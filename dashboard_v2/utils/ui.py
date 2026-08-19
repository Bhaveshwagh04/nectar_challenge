"""
utils/ui.py - small shared bits so every page looks consistent without
copy-pasting the same CSS block into all six files.
"""

import streamlit as st

BRAND_BLUE = "#2E4B6B"
BRAND_RED = "#d1495b"
BRAND_GREEN = "#5b8c5a"
BRAND_ORANGE = "#c17f3e"
BRAND_PURPLE = "#8a5b8c"

ASSET_TYPE_COLORS = {
    "Chiller": BRAND_RED,
    "AHU": "#3b6ea5",
    "Pump": BRAND_GREEN,
    "EnergyMeter": BRAND_ORANGE,
    "EnvSensor": BRAND_PURPLE,
}


def page_setup(title, icon="🏢"):
    st.set_page_config(page_title=f"Nectar | {title}", page_icon=icon, layout="wide")
    st.markdown(
        """
        <style>
        .block-container { padding-top: 2rem; }
        [data-testid="stMetricValue"] { font-size: 1.6rem; }
        h1 { color: #2E4B6B; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title(f"{icon} {title}")
