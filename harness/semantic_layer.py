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
        "name": "Direct/DDD + specialty pharmacy feed",
        "kind": "transactional",
        "cadence": "monthly",
        "lag_months": 0,
        "account_grain": True,
        "notes": ["System of record at account grain for monthly synthetic data.",
                  "Details, samples, programs, and writer activity are collected here only."],
    },
    "source_b": {
        "name": "Projected retail panel",
        "kind": "panel-projected",
        "cadence": "monthly",
        "lag_months": 1,
        "account_grain": False,
        "notes": ["Panel-projected: known small multiplicative bias vs. direct/DDD feeds.",
                  "Latest month unavailable (1-month reporting lag).",
                  "Early history restated by the vendor."],
    },
}

# --------------------------------------------------------------------------- #
# Metric registry — variants are first-class, each with an owner and notes
# --------------------------------------------------------------------------- #
METRICS = {
    "trx": {
        "label": "TRx", "family": "prescriptions", "kind": "additive", "additive": True,
        "sources": ["source_a", "source_b"], "default_source": "source_a",
        "default_variant": "units",
        "variants": {
            "units": {"column": "trx_units", "label": "TRx units", "comparison_group": "volume", "owner": "Commercial Analytics", "notes": "Total prescriptions in unit equivalents."},
            "dollars": {"column": "trx_dollars", "label": "TRx dollars", "comparison_group": "currency", "owner": "Finance", "notes": "Gross prescription value before rebates."},
            "normalized": {"column": "trx_normalized", "label": "TRx normalized equivalents", "comparison_group": "volume", "owner": "Commercial Analytics", "notes": "Pack-size normalized equivalent units."},
        },
    },
    "nrx": {
        "label": "NRx", "family": "new_prescriptions", "kind": "additive", "additive": True,
        "sources": ["source_a", "source_b"], "default_source": "source_a",
        "default_variant": "units",
        "variants": {"units": {"column": "nrx", "label": "NRx", "owner": "Brand Analytics", "notes": "New prescriptions in the month."}},
    },
    "nbrx": {
        "label": "NBRx", "family": "new_to_brand", "kind": "additive", "additive": True,
        "sources": ["source_a", "source_b"], "default_source": "source_a",
        "default_variant": "units",
        "variants": {"units": {"column": "nbrx", "label": "NBRx", "owner": "Brand Analytics", "notes": "New-to-brand prescriptions in the month."}},
    },
    "trx_share": {
        "label": "TRx market share", "family": "share", "kind": "ratio", "additive": False,
        "sources": ["source_a", "source_b"], "default_source": "source_a",
        "default_variant": "brand_market",
        "variants": {"brand_market": {"numerator": "trx_units", "denominator": "market_trx",
                                        "label": "TRx market share", "owner": "Brand Analytics",
                                        "notes": "Brand TRx divided by total market TRx; descriptive only."}},
    },
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
        "label": "Details",
        "family": "field_effort", "kind": "additive",
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
    "samples": {
        "label": "Samples dropped", "family": "field_effort", "kind": "additive", "additive": True,
        "sources": ["source_a"], "default_source": "source_a", "default_variant": "units",
        "variants": {"units": {"column": "samples", "label": "Sample units", "owner": "Sales Operations", "notes": "Recorded sample units."}},
    },
    "speaker_attendance": {
        "label": "Speaker attendance", "family": "programs", "kind": "additive", "additive": True,
        "sources": ["source_a"], "default_source": "source_a", "default_variant": "attendees",
        "variants": {"attendees": {"column": "speaker_attendance", "label": "Speaker-program attendance", "owner": "Marketing Operations", "notes": "Recorded HCP attendance."}},
    },
    "new_writers": {
        "label": "New writers", "family": "writers", "kind": "additive", "additive": True,
        "sources": ["source_a"], "default_source": "source_a", "default_variant": "strict",
        "variants": {"strict": {"column": "new_writers", "label": "New writers", "owner": "Brand Analytics", "notes": "First observed brand prescription in the measured history."}},
    },
}

DIMENSIONS = ["territory", "district", "region", "specialty", "payer_channel",
              "segment", "channel"]
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
    "trx market share": "trx_share", "market share": "trx_share", "share": "trx_share",
    "total prescriptions": "trx", "total scripts": "trx", "trx": "trx", "scripts": "trx",
    "new-to-brand": "nbrx", "new to brand": "nbrx", "nbrx": "nbrx",
    "new prescriptions": "nrx", "nrx": "nrx",
    "details": "calls", "detail calls": "calls",
    "samples": "samples", "sample units": "samples",
    "speaker attendance": "speaker_attendance", "speaker programs": "speaker_attendance",
    "new writers": "new_writers", "writers": "new_writers",
    "revenue": "revenue", "sales": "revenue", "bookings": "revenue",
    "units": "units", "volume": "units",
    "calls": "calls", "activity": "calls",
    "new customers": "new_customers", "new customer": "new_customers", "acquisitions": "new_customers",
}

# --------------------------------------------------------------------------- #
# Event registry — feeds the causal design advisor
# --------------------------------------------------------------------------- #
EVENTS = {
    "speaker_launch": {
        "name": "Speaker-program launch (East)", "start": "2025-10",
        "scope": {"region": "East"}, "metrics": ["trx", "nrx", "nbrx"],
        "keywords": ["east", "speaker", "speaker program", "program launch"],
        "candidate_controls": {"region": ["North", "South"]},
        "notes": "Region-wide program; West excluded because of a concurrent competitor shock.",
    },
    "formulary_win": {
        "name": "Medicare Part D formulary win (South)", "start": "2026-01",
        "scope": {"region": "South"}, "metrics": ["trx", "nrx"],
        "keywords": ["south", "formulary", "medicare", "payer win"],
        "candidate_controls": {"region": ["North", "East"]},
        "notes": "Monthly synthetic event used to test a payer-channel hypothesis.",
    },
    "competitor_launch": {
        "name": "Competitor launch (West / Cardiology)", "start": "2026-04",
        "scope": {"region": "West", "specialty": "Cardiology"},
        "metrics": ["trx", "nrx", "nbrx"],
        "keywords": ["west", "competitor", "cardiology", "competitor launch"],
        "candidate_controls": {"region": ["North", "South", "East"]},
        "notes": "Short post-period; treated specialty is explicitly registered.",
    },
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
    chosen_group = m["variants"][chosen_variant].get("comparison_group")
    alternates = [("variant", v) for v, spec in m["variants"].items()
                  if v != chosen_variant
                  and (chosen_group is None or spec.get("comparison_group") == chosen_group)]
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
    spec = METRICS[metric]["variants"][variant]
    return spec.get("column") or spec["numerator"]


def metric_kind(metric: str) -> str:
    return METRICS[metric].get("kind", "additive")


def aggregate_metric(df: pd.DataFrame, metric: str, variant: str) -> float:
    """Aggregate additive and ratio metrics without ever summing percentages."""
    spec = METRICS[metric]["variants"][variant]
    if metric_kind(metric) == "ratio":
        denominator = float(df[spec["denominator"]].sum())
        return float(df[spec["numerator"]].sum()) / denominator if denominator else 0.0
    return float(df[spec["column"]].sum())


def monthly_metric(df: pd.DataFrame, metric: str, variant: str) -> pd.Series:
    spec = METRICS[metric]["variants"][variant]
    if metric_kind(metric) == "ratio":
        numerator = df.groupby("month")[spec["numerator"]].sum()
        denominator = df.groupby("month")[spec["denominator"]].sum()
        return numerator.div(denominator.where(denominator != 0)).fillna(0.0).sort_index()
    return df.groupby("month")[spec["column"]].sum().sort_index()
