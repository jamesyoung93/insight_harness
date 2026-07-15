"""Insight Harness — decision intelligence on a governed semantic layer.

Run:  streamlit run app.py
"""
from __future__ import annotations

import os

import streamlit as st

from harness import runtime_policy
from harness import semantic_layer as sl
from views import causal_studio, digest, help_page, home, monitoring, registry, reliability

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
    "Home": home.render,
    "Digest": digest.render,
    "Monitoring": monitoring.render,
    "Causal Studio": causal_studio.render,
    "Semantic Layer": registry.render,
    "Reliability": reliability.render,
    "How answers are produced": help_page.render,
}

page = st.sidebar.radio("Insight Harness", list(PAGES), key="nav")
st.sidebar.caption(f"data version `{sl.data_version()}`")

session_key = st.session_state.get("api_key")
deployment_key = os.environ.get("ANTHROPIC_API_KEY")
deployment_allowed = bool(deployment_key and runtime_policy.deployment_llm_enabled())
credential_source = "session" if session_key else "deployment" if deployment_allowed else None

if credential_source:
    st.sidebar.success("Language-model translation is enabled.")
    if credential_source == "session":
        st.sidebar.caption("Using the Anthropic API credential entered for this app session.")
    else:
        st.sidebar.caption("Using an Anthropic API credential configured by the deployment owner.")
else:
    st.sidebar.warning("Language-model translation is off.")
    if deployment_key and not deployment_allowed:
        st.sidebar.caption("A deployment credential exists but is not enabled for anonymous "
                           "sessions. Questions use the bounded built-in parser unless you "
                           "add your own Anthropic API key below.")
    else:
        st.sidebar.caption("No API credential is bundled with this app. Questions currently "
                           "use the bounded built-in parser. Add your own Anthropic API key "
                           "below to enable flexible model-backed translation.")

with st.sidebar.expander("Enable language-model translation (API key required)",
                         expanded=not credential_source):
    st.caption("A valid Anthropic API credential is required. The model only translates your "
               "question into a governed query; it never computes or writes the answer. Every "
               "translation is checked against the metric registry, and anything invalid "
               "falls back to the built-in parser.")
    st.text_input("Your Anthropic API key (required for LLM translation)", type="password",
                  key="api_key", help="Only enter a key in a deployment you trust.")
    models = runtime_policy.allowed_models()
    if st.session_state.get("llm_model") not in models:
        st.session_state["llm_model"] = models[0]
    st.selectbox("Model", models, key="llm_model",
                 help="The deployment owner controls this allowlist.")
    used = int(st.session_state.get("_model_calls_used", 0))
    limit = runtime_policy.session_model_call_limit()
    st.caption(f"Session model-call allowance: {max(0, limit - used)} of {limit} remaining.")
    st.caption("A key entered here remains in the current Streamlit session and is not written "
               "to disk by the app. The question and registered metric/dimension vocabulary "
               "are sent to Anthropic for intent translation; source rows are not. In a "
               "deployment you control, a server key also requires explicit public-use opt-in.")

PAGES[page]()
