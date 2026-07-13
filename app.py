"""Insight Harness — decision intelligence on a governed semantic layer.

Run:  streamlit run app.py
"""
from __future__ import annotations

import os

import streamlit as st

from harness import semantic_layer as sl
from views import ask, causal_studio, help_page, monitoring, registry, reliability

st.set_page_config(page_title="Insight Harness", page_icon="🧭", layout="wide")

st.markdown("""
<style>
.tier-chip {display:inline-block; padding:2px 12px; border-radius:12px; color:white;
            font-size:0.78rem; font-weight:600; letter-spacing:.04em; margin-right:8px;}
.stamp {font-family:monospace; font-size:0.75rem; color:#6B7280; border-left:3px solid #0E7C7B;
        padding-left:10px; margin-top:4px;}
.caveat {background:#FDF6EC; border-left:3px solid #B07C0E; padding:8px 12px; margin:4px 0;
         font-size:0.86rem; border-radius:0 6px 6px 0;}
.refusal {background:#F1F2F4; border-left:3px solid #6B7280; padding:12px 14px;
          border-radius:0 6px 6px 0;}
</style>""", unsafe_allow_html=True)

PAGES = {
    "Ask": ask.render,
    "Monitoring": monitoring.render,
    "Causal Studio": causal_studio.render,
    "Semantic Layer": registry.render,
    "Reliability": reliability.render,
    "How answers are produced": help_page.render,
}

page = st.sidebar.radio("Insight Harness", list(PAGES), key="nav")
st.sidebar.caption(f"data version `{sl.data_version()}`")

with st.sidebar.expander("Language-model translation"):
    st.caption("Add an Anthropic API key to translate questions with a language model. "
               "The model only translates your question into a governed query — it never "
               "computes or answers. Every translation is checked against the metric "
               "registry, and anything invalid falls back to the built-in parser. The key "
               "stays in session memory only; you can also set the ANTHROPIC_API_KEY "
               "environment variable.")
    st.text_input("Anthropic API key", type="password", key="api_key")
    st.text_input("Model", value="claude-sonnet-4-6", key="llm_model")
    active = bool(st.session_state.get("api_key") or os.environ.get("ANTHROPIC_API_KEY"))
    st.markdown(f"Translator: **{'language model (registry-validated)' if active else 'built-in parser'}**")

PAGES[page]()
