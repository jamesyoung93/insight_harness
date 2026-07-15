# Insight Harness

Insight Harness is a Streamlit decision-intelligence workbench for a synthetic
pharma commercial dataset. It starts with persona-specific KPI tiles, then lets
the user interrogate the same governed metrics through natural language,
monitoring, a top-three digest, and registered causal designs.

The trust boundary is simple: a language model may translate a question or
rephrase a digest headline, but it never computes an answer. Registry-validated
intents enter deterministic engines, and every result is exported as a
provenance-stamped artifact with its source, variant, data version, code, tier,
and stable hash.

> [!IMPORTANT]
> All bundled records and effects are deterministic synthetic demo data. The
> dataset is monthly, not weekly, and contains no patient, prescriber, or other
> real-world data. Do not use this repository to make clinical or production
> commercial decisions without replacing and validating the full data contract.

## Quick start

Python 3.12 is the CI runtime.

```bash
python -m venv .venv
# PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -c constraints.txt -r requirements-dev.txt
python data/generate_demo_data.py
python -m streamlit run app.py
```

Run the verification suite separately:

```bash
python -m compileall -q app.py harness views tests
python -m pytest -q
```

The app is fully usable without an API key. In that mode, questions use the
bounded rule parser and digest cards use deterministic templates.

## What is implemented

| Surface | Capability |
|---|---|
| **Home** | Persona defaults; live KPI, diagnostic, and retrieval tiles; R3M/R6M/R12M and MoM/QoQ/YoY controls; one scope selector spanning region, district, territory, specialty, and payer channel; governed source/variant overrides; material-fork disclosure; Watch, Open, Break down, and JSON download actions; natural-language exploration below the tiles |
| **Digest** | Three deterministic signals ranked from movements, session watches, material source/definition forks, and registered-event overlap; scope precedence, novelty, and one item per metric family; artifact downloads and resolution-preserving drill-through; optional validated model phrasing |
| **Monitoring** | Session-owned watched insights plus an impact-ranked anomaly feed, with breakdown drill-through that retains the saved comparison and resolution context |
| **Causal Studio** | Difference-in-differences proposals for three registered synthetic events, including treated/control scopes, computed pre-trend and sensitivity checks, and caveats; analyst sign-off requires administrator-token authentication and adds a provenance stamp without promoting the Hypothesis tier |
| **Semantic Layer** | Public read-only metric and source registries; server-token-gated administration of the materiality threshold and default variants; atomic configuration writes with embedded audit history and a synced local JSONL mirror |
| **Reliability** | A pharma-only independent golden set covering values, ratios, retrieval, decomposition, causal effects, divergence, refusals, and source/data contracts; every case runs twice for hash reproducibility, while the broader test suite enforces watches and tile parity |
| **How answers are produced** | In-product explanation of tiers, provenance, and the authority boundary between translation and computation |

### Persona presets

The five presets are Sales Rep, District Manager, Brand Marketing, Market
Access, and Executive. Sales Rep starts at territory scope, District Manager at
district scope, and the other presets at national scope. Each preset supplies a
tile set, scope, window, comparison basis, and digest scope.

Tile add/remove/reorder choices, saved defaults, watches, and public digest
history are owned by the current Streamlit session. They are intentionally not
written to a shared profile file, so one anonymous viewer cannot change another
viewer's workspace. They expire with the session; production persistence needs
authenticated user identity and an external store.

## Governed pharma contract

The registry exposes these monthly metrics:

- TRx with units, gross-dollar, and payer-normalized variants
- NRx and NBRx
- TRx market share, calculated as brand TRx divided by market TRx
- details delivered, call plan, and call-plan attainment
- samples dropped, speaker attendance, and new writers

The registered dimensions are `territory`, `district`, `region`, `specialty`,
and `payer_channel`. The HCP universe also carries derived decile, trailing Rx,
recency, and 90-day activity fields for governed retrieval such as whitespace
HCPs.

The two source products deliberately disagree in controlled ways:

- `source_a` — **Direct/DDD + specialty pharmacy feed**, at one
  `account_id` × `month` row, covering prescriptions and field activity with no
  reporting lag.
- `source_b` — **Projected retail panel**, aggregated by month and the five
  registered dimensions, with a one-month lag, registered regional projection
  factors, and an early-history restatement. It does not claim HCP or
  field-activity grain.

Source comparisons use the common available window. Missing panel coverage is
reported as a coverage gap rather than misclassified as a value fork. Ratio
metrics are aggregated from numerator and denominator; an undefined denominator
stays undefined rather than becoming a false zero. Ratio decomposition is
deliberately refused because no governed share-decomposition method is
registered.

Three synthetic events are registered with exact treated and control scopes:

- speaker-program launch in two East territories, with matched North territory
  controls;
- South Medicare Part D formulary win, with North/East Medicare Part D
  controls; and
- competitor launch in West Cardiology, with Cardiology controls from the
  other three regions.

`data/ground_truth.json` records the injected effects, source pathologies,
grains, and invariants used by the independent evaluation path.

## Answer pipeline

```text
question or saved spec
  -> triage / validated model translation       harness/triage.py
  -> governed source + variant resolution       harness/semantic_layer.py
  -> deterministic engine                       harness/engines/
  -> common-window divergence + caveats         harness/services.py
  -> provenance-stamped AnswerArtifact          harness/provenance.py
```

Question classes are Retrieval, Descriptive, Diagnostic, Causal, Predictive,
and Out of scope. Predictive requests are refused because the registry contains
no governed forecasting model. Causal requests execute only when they match a
registered event and preserve its treated population; otherwise the system
returns a scoped reframe.

A tile is an immutable saved-question specification, not a separate dashboard
calculation. Tile rendering, Ask, Monitoring drill-through, and digest
drill-through all enter the same answer pipeline. Cache identity includes the
materialized spec, effective scope, governance configuration, and data version.

## Optional language-model features

Users may enter their own Anthropic API key in the sidebar for the current
session. A deployment-owned `ANTHROPIC_API_KEY` is disabled for anonymous use
unless the operator also sets:

```text
INSIGHT_HARNESS_ALLOW_PUBLIC_LLM=true
```

The deployment controls an allowlist of model IDs and a per-session model-call
limit. Model translation is registry-validated before execution; invalid output
falls back to rules. Digest rewriting receives a bounded fact payload and is
accepted only if metric, scope, entities, direction, units, numeric claims, and
non-causal wording validate exactly. The deterministic template wins on any
failure.

See [SECURITY.md](SECURITY.md) before enabling model access or governance writes
on a public deployment.

Feedback question identifiers are keyed HMACs rather than plain hashes. By
default, a random per-process key prevents identifiers from being correlated
across restarts. Operators that need stable, controlled correlation may set
`INSIGHT_HARNESS_TELEMETRY_HASH_KEY`; treat that value as a secret and define a
retention purpose before enabling it.

## Synthetic data and reproducibility

`data/generate_demo_data.py` uses seed `42` to produce:

- 240 synthetic HCPs across 24 months (`2024-07` through `2026-06`);
- 5,760 account-month rows in source A;
- 1,104 aggregated, one-month-lagged panel rows in source B; and
- an HCP table derived and reconciled from source A.

Generated CSV files use explicit UTF-8/LF output. CI regenerates all four data
artifacts and fails if they differ from the committed benchmark:

```bash
python data/generate_demo_data.py
git diff --exit-code -- data/accounts.csv data/fact_source_a.csv \
  data/fact_source_b.csv data/ground_truth.json
```

The in-app label “Daily digest” describes the intended workflow, not fake daily
rotation. With the committed monthly snapshot, the digest changes only when its
data version, governance, persona/scope, watches, or identity-scoped history
changes. A true morning digest requires a real refreshed feed.

## Evaluation and CI

The test suite includes deterministic engine tests, data-contract and golden-set
checks, tile/spec/cache parity, saved-insight migrations, session isolation,
digest ranking and narrator attacks, provenance hashing, runtime policy, and
headless Streamlit page flows.

GitHub Actions runs on pull requests and pushes to `main` with read-only
repository permissions. It resolves the declared dependency ranges against the
tested direct-version pins in `constraints.txt`, checks the dependency graph,
regenerates the benchmark, compiles the app, and runs the full test suite.

For operational configuration, state-file behavior, deployment, and the release
checklist, see [docs/OPERATIONS.md](docs/OPERATIONS.md).

## Repository map

```text
app.py                          Streamlit router and runtime-policy controls
views/
  home.py                       persona tile band and exploration workspace
  digest.py                     deterministic top-three digest
  monitoring.py                 session watches and anomaly feed
  causal_studio.py              registered causal design proposals
  registry.py                   registry browser and authenticated governance
  reliability.py                golden-set accuracy record
  common.py                     artifact rendering and drill-through helpers
harness/
  semantic_layer.py             metrics, dimensions, sources, events, governance
  tiles.py / tile_runtime.py     saved-question specs, identity, and execution
  saved_insights.py             schema-migrated session watch/tile store
  triage.py / llm_translator.py rule and optional model translation
  engines/                      descriptive, retrieval, decomposition, causal
  services.py                   divergence, monitoring, feedback, caveats
  digest.py / digest_store.py   ranking, artifacts, and history stores
  digest_narrator.py            optional bounded/validated phrasing
  pipeline.py                   orchestration and independent golden set
  provenance.py                 stable answer artifacts and hashes
  runtime_policy.py             model, quota, privacy, and admin policy
data/
  generate_demo_data.py         deterministic pharma benchmark generator
  ground_truth.json             independent event/source/data contract
  fact_source_a.csv             account-month system-of-record facts
  fact_source_b.csv             aggregated projected panel
  accounts.csv                  derived HCP universe
tests/                          unit, contract, golden, and AppTest coverage
```
