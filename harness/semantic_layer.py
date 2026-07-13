"""Semantic layer: the governed foundation every answer is composed from.

One machine-readable definition per metric, with *named variants* as first-class
citizens, source registry with known limitations, and deterministic resolution
rules. The AI never improvises a definition; it resolves one, and discloses it.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
CONFIG_PATH = DATA_DIR / "governance_config.json"
GOVERNANCE_LOG = DATA_DIR / "governance_log.jsonl"

# --------------------------------------------------------------------------- #
# Source registry
# --------------------------------------------------------------------------- #
SOURCES = {
    "source_a": {
        "name": "Internal warehouse (Source A)",
        "kind": "transactional",
        "cadence": "monthly",
        "lag_months": 0,
        "account_grain": True,
        "notes": ["System of record for finance reporting.",
                  "Activity metrics (calls, new customers) collected here only."],
    },
    "source_b": {
        "name": "External panel feed (Source B)",
        "kind": "panel-projected",
        "cadence": "monthly",
        "lag_months": 1,
        "account_grain": False,
        "notes": ["Panel-projected: known small multiplicative bias vs. warehouse.",
                  "Latest month unavailable (1-month reporting lag).",
                  "Early history restated by the vendor."],
    },
}

# --------------------------------------------------------------------------- #
# Metric registry — variants are first-class, each with an owner and notes
# --------------------------------------------------------------------------- #
METRICS = {
    "revenue": {
        "label": "Revenue",
        "additive": True,
        "sources": ["source_a", "source_b"],
        "default_source": "source_a",
        "default_variant": "net",
        "variants": {
            "net":   {"column": "revenue_net",   "label": "Net revenue",
                      "owner": "Finance", "notes": "Gross less contractual segment discounts. Board-reported."},
            "gross": {"column": "revenue_gross", "label": "Gross revenue",
                      "owner": "Sales Ops", "notes": "Pre-discount bookings view used in territory scorecards."},
        },
    },
    "units": {
        "label": "Units",
        "additive": True,
        "sources": ["source_a", "source_b"],
        "default_source": "source_a",
        "default_variant": "std",
        "variants": {"std": {"column": "units", "label": "Units", "owner": "Sales Ops", "notes": ""}},
    },
    "calls": {
        "label": "Sales calls",
        "additive": True,
        "sources": ["source_a"],
        "default_source": "source_a",
        "default_variant": "std",
        "variants": {"std": {"column": "calls", "label": "Sales calls", "owner": "Sales Ops",
                             "notes": "Internally collected; uniform definition across regions."}},
    },
    "new_customers": {
        "label": "New customers",
        "additive": True,
        "sources": ["source_a"],
        "default_source": "source_a",
        "default_variant": "strict",
        "variants": {
            "strict": {"column": "new_cust_strict", "label": "New customers (strict)",
                       "owner": "Finance", "notes": "First purchase ever. Excludes reactivations."},
            "broad":  {"column": "new_cust_broad", "label": "New customers (broad)",
                       "owner": "Marketing", "notes": "Includes reactivated lapsed accounts. Campaign reporting."},
        },
    },
}

DIMENSIONS = ["region", "segment", "channel"]
MATERIALITY_REL = 0.02  # divergence above 2% is surfaced (registry default)


# --------------------------------------------------------------------------- #
# Governance configuration — admin-set overrides of registry defaults.
# Every change is logged: governance changes are provenance too.
# --------------------------------------------------------------------------- #
def _governance_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def materiality() -> float:
    v = _governance_config().get("materiality_rel")
    return float(v) if isinstance(v, (int, float)) and 0.005 <= v <= 0.20 else MATERIALITY_REL


def default_variant(metric: str) -> str:
    v = _governance_config().get("default_variants", {}).get(metric)
    return v if v in METRICS[metric]["variants"] else METRICS[metric]["default_variant"]


def set_governance(materiality_rel: float | None = None,
                   default_variants: dict | None = None) -> dict:
    """Apply and log a governance change; returns what actually changed."""
    cfg = _governance_config()
    change: dict = {}
    if materiality_rel is not None and 0.005 <= materiality_rel <= 0.20 \
            and materiality_rel != materiality():
        cfg["materiality_rel"] = float(materiality_rel)
        change["materiality_rel"] = float(materiality_rel)
    for m, v in (default_variants or {}).items():
        if m in METRICS and v in METRICS[m]["variants"] and v != default_variant(m):
            cfg.setdefault("default_variants", {})[m] = v
            change.setdefault("default_variants", {})[m] = v
    if not change:
        return change
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "change": change}
    with GOVERNANCE_LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return change


def governance_log() -> list[dict]:
    if not GOVERNANCE_LOG.exists():
        return []
    out = []
    for line in GOVERNANCE_LOG.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out

# keyword -> metric mapping used by the built-in parser and by the
# language-model translator's registry context
METRIC_KEYWORDS = {
    "revenue": "revenue", "sales": "revenue", "bookings": "revenue",
    "units": "units", "volume": "units",
    "calls": "calls", "activity": "calls",
    "new customers": "new_customers", "new customer": "new_customers", "acquisitions": "new_customers",
}

# --------------------------------------------------------------------------- #
# Event registry — feeds the causal design advisor
# --------------------------------------------------------------------------- #
EVENTS = {
    "east_program": {
        "name": "Partner enablement program (East)",
        "start": "2025-10",
        "scope": {"region": "East"},
        "metrics": ["revenue", "units"],
        "keywords": ["east", "program", "enablement", "partner program"],
        "candidate_controls": {"region": ["North", "South"]},
        "notes": "Rolled out region-wide in East; West excluded as control (concurrent shock).",
    },
    "west_shock": {
        "name": "Competitor entry (West / Enterprise)",
        "start": "2026-04",
        "scope": {"region": "West", "segment": "Enterprise"},
        "metrics": ["revenue"],
        "keywords": ["west", "competitor", "drop", "decline"],
        "candidate_controls": {"region": ["North", "South", "East"]},
        "notes": "Short post-period (limited months of data since event).",
    },
}


@dataclass
class Resolution:
    """The disclosed outcome of source/variant resolution for a question."""
    metric: str
    variant: str
    source: str
    reason: str
    alternates: list = field(default_factory=list)  # (kind, id) pairs


def resolve(metric: str, source: str | None = None, variant: str | None = None) -> Resolution:
    m = METRICS[metric]
    # overrides apply only when registered for this metric; anything else clamps
    # to the governed default and the clamp is disclosed in the resolution reason
    source_ok = source in m["sources"] if source else False
    variant_ok = variant in m["variants"] if variant else False
    chosen_source = source if source_ok else m["default_source"]
    chosen_variant = variant if variant_ok else default_variant(metric)
    alternates = [("variant", v) for v in m["variants"] if v != chosen_variant]
    alternates += [("source", s) for s in m["sources"] if s != chosen_source]
    clamped = [name for name, requested, ok in
               (("source", source, source_ok), ("variant", variant, variant_ok))
               if requested and not ok]
    if (source_ok or variant_ok) and not clamped:
        reason = "user override"
    elif source_ok or variant_ok:
        reason = (f"user override ({'source' if source_ok else 'variant'}); requested "
                  f"{' and '.join(clamped)} not registered for this metric — governed "
                  f"default used")
    elif clamped:
        reason = (f"requested {' and '.join(clamped)} not registered for this metric; "
                  f"governed default used ({m['variants'][chosen_variant]['owner']}-owned "
                  f"variant on {SOURCES[chosen_source]['name']})")
    else:
        reason = (f"default governed resolution ({m['variants'][chosen_variant]['owner']}-owned "
                  f"variant on {SOURCES[chosen_source]['name']})")
    return Resolution(metric, chosen_variant, chosen_source, reason, alternates)


@lru_cache(maxsize=8)
def load_fact(source: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"fact_{source}.csv")


@lru_cache(maxsize=1)
def load_accounts() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "accounts.csv")


@lru_cache(maxsize=1)
def ground_truth() -> dict:
    return json.loads((DATA_DIR / "ground_truth.json").read_text())


@lru_cache(maxsize=1)
def data_version() -> str:
    h = hashlib.sha256()
    for f in sorted(DATA_DIR.glob("*.csv")):
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def months(source: str = "source_a") -> list[str]:
    return sorted(load_fact(source)["month"].unique())


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Filter values may be a scalar or a list of registered values."""
    for dim, val in filters.items():
        if dim in df.columns:
            df = df[df[dim].isin(val)] if isinstance(val, (list, tuple)) else df[df[dim] == val]
    return df


def scope_string(filters: dict) -> str:
    parts = [f"{k} in [{', '.join(v)}]" if isinstance(v, (list, tuple)) else f"{k}={v}"
             for k, v in filters.items()]
    return ", ".join(parts) or "all scopes"


def column_for(metric: str, variant: str) -> str:
    return METRICS[metric]["variants"][variant]["column"]
