"""Generate placeholder demo data with documented ground truth.

This is a stand-in for the full simulated benchmark dataset. It deliberately
bakes in the classes of issues the harness must handle, and exports the
ground truth to ground_truth.json so the evaluation page can score against it.

Baked-in structure (all documented, all recoverable):
  1. TREND + SEASONALITY  : mild upward trend, Q4 seasonal bump.
  2. CAUSAL EVENT (lift)  : "Partner enablement program" in East, from 2025-10,
                            true effect = +8% on revenue & units (DiD target).
  3. CAUSAL EVENT (shock) : "Competitor entry" hits West/Enterprise from 2026-04,
                            true effect = -22% revenue (decomposition/anomaly target).
  4. SOURCE DISAGREEMENT  : source_b is a panel-projected external feed:
                            +3% avg multiplicative bias (varies by region),
                            1-month reporting lag (latest month missing),
                            first 3 months restated (+1.5%).
  5. METRIC VARIANTS      : revenue_net vs revenue_gross (segment discount rates);
                            new_cust_strict vs new_cust_broad (reactivations).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RNG = np.random.default_rng(42)

MONTHS = pd.period_range("2024-07", "2026-06", freq="M").astype(str).tolist()
REGIONS = ["North", "South", "East", "West"]
SEGMENTS = ["Enterprise", "MidMarket", "SMB"]
CHANNELS = ["Direct", "Partner"]
SPECIALTY = {"Enterprise": "Cardiology", "MidMarket": "Endocrinology",
             "SMB": "Primary Care"}
PAYER = {("Enterprise", "Direct"): "Commercial",
         ("Enterprise", "Partner"): "Medicare Part D",
         ("MidMarket", "Direct"): "Commercial",
         ("MidMarket", "Partner"): "Medicaid",
         ("SMB", "Direct"): "Cash",
         ("SMB", "Partner"): "Medicare Part D"}

BASE = {("Enterprise",): 90.0, ("MidMarket",): 55.0, ("SMB",): 30.0}
REGION_MULT = {"North": 1.15, "South": 0.9, "East": 1.0, "West": 1.05}
CHANNEL_SHARE = {"Direct": 0.62, "Partner": 0.38}
DISCOUNT = {"Enterprise": 0.18, "MidMarket": 0.12, "SMB": 0.07}  # gross -> net

EAST_PROGRAM_START = "2025-10"   # true lift +8% (revenue, units) in East
EAST_LIFT = 0.08
WEST_SHOCK_START = "2026-04"     # true drop -22% on West/Enterprise revenue
WEST_SHOCK = -0.22
SOURCE_B_BIAS = {"North": 1.025, "South": 1.035, "East": 1.030, "West": 1.032}
SOURCE_B_RESTATED = MONTHS[:3]   # restated history
SOURCE_B_LAG = 1                 # latest month absent
FORMULARY_START = "2026-01"
FORMULARY_LIFT = 0.10


def month_index(m: str) -> int:
    return MONTHS.index(m)


def build_fact() -> pd.DataFrame:
    rows = []
    for mi, m in enumerate(MONTHS):
        trend = 1.0 + 0.004 * mi
        season = 1.10 if m[-2:] in ("10", "11", "12") else 1.0
        for r in REGIONS:
            for s in SEGMENTS:
                for c in CHANNELS:
                    mu = BASE[(s,)] * REGION_MULT[r] * CHANNEL_SHARE[c] * trend * season
                    noise = RNG.normal(1.0, 0.045)
                    rev = mu * noise
                    # causal event 1: East program lift
                    if r == "East" and month_index(m) >= month_index(EAST_PROGRAM_START):
                        rev *= 1 + EAST_LIFT
                    # causal event 2: West/Enterprise competitor shock
                    if r == "West" and s == "Enterprise" and month_index(m) >= month_index(WEST_SHOCK_START):
                        rev *= 1 + WEST_SHOCK
                    units = rev / (2.4 if s == "Enterprise" else 1.6 if s == "MidMarket" else 0.9)
                    units *= RNG.normal(1.0, 0.02)
                    if r == "South" and c == "Partner" and month_index(m) >= month_index(FORMULARY_START):
                        units *= 1 + FORMULARY_LIFT
                    calls = max(0, RNG.normal(46 if c == "Direct" else 24, 5)) * REGION_MULT[r]
                    ncs = max(0, RNG.poisson(6 if s == "SMB" else 3))
                    district = f"{r} District {'1' if s == 'Enterprise' else '2'}"
                    territory = f"{r[:1]}-{s[:2].upper()}-{'A' if c == 'Direct' else 'B'}"
                    trx = max(0.0, units * 10.0)
                    nrx = trx * (0.16 if s == "Enterprise" else 0.20 if s == "MidMarket" else 0.24)
                    nbrx = nrx * (0.68 if c == "Direct" else 0.76)
                    market_trx = trx / (0.105 + 0.012 * (r == "East") + 0.006 * (s == "Enterprise"))
                    rows.append({
                        "month": m, "region": r, "segment": s, "channel": c,
                        "district": district, "territory": territory,
                        "specialty": SPECIALTY[s], "payer_channel": PAYER[(s, c)],
                        "revenue_gross": round(rev, 3),
                        "revenue_net": round(rev * (1 - DISCOUNT[s]), 3),
                        "units": round(units, 2),
                        "calls": round(calls, 1),
                        "new_cust_strict": int(ncs),
                        "new_cust_broad": int(ncs + RNG.poisson(1.4)),
                        "trx_units": round(trx, 2),
                        "trx_dollars": round(trx * (118.0 if s == "Enterprise" else 94.0), 2),
                        "trx_normalized": round(trx * (0.96 if c == "Partner" else 1.0), 2),
                        "nrx": round(nrx, 2), "nbrx": round(nbrx, 2),
                        "market_trx": round(market_trx, 2),
                        "samples": int(max(0, calls * RNG.normal(3.2, 0.35))),
                        "speaker_attendance": int(max(0, RNG.poisson(4 if c == "Partner" else 2))),
                        "new_writers": int(max(0, round(ncs * RNG.normal(1.35, 0.15)))),
                    })
    return pd.DataFrame(rows)


def build_source_b(a: pd.DataFrame) -> pd.DataFrame:
    b = a.copy()
    for col in ("revenue_gross", "revenue_net", "units", "trx_units", "trx_dollars",
                "trx_normalized", "nrx", "nbrx", "market_trx"):
        b[col] = b.apply(lambda x, c=col: x[c] * SOURCE_B_BIAS[x["region"]] * RNG.normal(1.0, 0.01), axis=1)
    restate = b["month"].isin(SOURCE_B_RESTATED)
    for col in ("revenue_gross", "revenue_net"):
        b.loc[restate, col] = b.loc[restate, col] * 1.015
    b = b[~b["month"].isin(MONTHS[-SOURCE_B_LAG:])]  # reporting lag
    b = b.drop(columns=["calls", "new_cust_strict", "new_cust_broad", "samples",
                        "speaker_attendance", "new_writers"])  # not collected externally
    return b.round(3)


def build_accounts(a: pd.DataFrame) -> pd.DataFrame:
    rows = []
    last12 = MONTHS[-12:]
    for i in range(240):
        r = RNG.choice(REGIONS)
        s = RNG.choice(SEGMENTS, p=[0.25, 0.35, 0.40])
        base = {"Enterprise": 950, "MidMarket": 380, "SMB": 110}[s] * RNG.lognormal(0, 0.5)
        rows.append({
            "account_id": f"A{i+1:04d}",
            "name": f"Dr. Morgan {i+1:03d}",
            "region": r, "segment": s,
            "district": f"{r} District {'1' if s == 'Enterprise' else '2'}",
            "territory": f"{r[:1]}-{s[:2].upper()}-A",
            "specialty": SPECIALTY[s],
            "payer_channel": PAYER[(s, "Direct")],
            "revenue_ttm": round(base, 1),
            "revenue_prev_ttm": round(base * RNG.normal(0.97, 0.12), 1),
            "trx_ttm": round(base * 8.5, 1),
            "calls_90d": int(max(0, RNG.normal(7, 5))),
            "months_since_activity": int(RNG.choice(range(0, 9), p=[.32, .2, .14, .1, .08, .06, .05, .03, .02])),
        })
    df = pd.DataFrame(rows)
    df["months_since_rx"] = df["months_since_activity"]
    df["decile"] = pd.qcut(df["revenue_ttm"], 10, labels=False) + 1
    df["_l12"] = ",".join(last12)
    return df


def main():
    a = build_fact()
    b = build_source_b(a)
    accounts = build_accounts(a)
    a.to_csv(HERE / "fact_source_a.csv", index=False)
    b.to_csv(HERE / "fact_source_b.csv", index=False)
    accounts.drop(columns=["_l12"]).to_csv(HERE / "accounts.csv", index=False)

    h = hashlib.sha256()
    for path in (HERE / "accounts.csv", HERE / "fact_source_a.csv", HERE / "fact_source_b.csv"):
        h.update(path.read_bytes())
    version = h.hexdigest()[:12]
    gt = {
        "data_version": version,
        "months": MONTHS,
        "events": {
            "speaker_launch": {"start": EAST_PROGRAM_START, "scope": {"region": "East"},
                               "true_effect_pct": EAST_LIFT, "metrics": ["trx", "nrx", "nbrx"],
                               "mechanism": "multiplicative regional prescription lift"},
            "formulary_win": {"start": FORMULARY_START, "scope": {"region": "South"},
                              "true_effect_pct": FORMULARY_LIFT, "metrics": ["trx", "nrx"],
                              "mechanism": "lift on South partner/payer-channel volume"},
            "competitor_launch": {"start": WEST_SHOCK_START,
                                  "scope": {"region": "West", "specialty": "Cardiology"},
                                  "true_effect_pct": WEST_SHOCK, "metrics": ["trx", "nrx", "nbrx"],
                                  "mechanism": "multiplicative West/Cardiology shock"},
            "east_program": {"start": EAST_PROGRAM_START, "scope": {"region": "East"},
                             "true_effect_pct": EAST_LIFT, "metrics": ["revenue", "units"],
                             "mechanism": "multiplicative lift on all East cells"},
            "west_shock": {"start": WEST_SHOCK_START, "scope": {"region": "West", "segment": "Enterprise"},
                           "true_effect_pct": WEST_SHOCK, "metrics": ["revenue"],
                           "mechanism": "multiplicative shock on West/Enterprise cells"},
        },
        "source_b_issues": {"bias_by_region": SOURCE_B_BIAS, "lag_months": SOURCE_B_LAG,
                            "restated_months": SOURCE_B_RESTATED, "restatement_factor": 1.015},
        "variant_definitions": {"revenue": {"net": "gross * (1 - segment discount)", "discounts": DISCOUNT},
                                "trx": {"units": "total prescriptions", "dollars": "gross prescription value",
                                        "normalized": "pack-size normalized equivalent units"},
                                "trx_share": {"numerator": "brand TRx units", "denominator": "market TRx units"},
                                "new_customers": {"strict": "first purchase ever",
                                                  "broad": "strict + reactivated lapsed accounts"}},
    }
    (HERE / "ground_truth.json").write_text(json.dumps(gt, indent=2))
    print(f"data written; version={version}; rows a={len(a)} b={len(b)} accounts={len(accounts)}")


if __name__ == "__main__":
    main()
