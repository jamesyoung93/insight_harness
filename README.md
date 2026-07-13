# Insight Harness

A Streamlit decision-intelligence workbench built on a governed semantic layer:
question triage → deterministic resolution against a metric registry →
deterministic engines → computed divergence & caveats → provenance-stamped
answer artifacts, with a standing accuracy record where abstention and
reproducibility are scored behaviors.

## Run

```bash
pip install -r requirements-dev.txt
python data/generate_demo_data.py   # regenerates the dataset
streamlit run app.py
pytest                              # headless UI + harness suite
```

## Deploy on Streamlit Community Cloud

This repository is ready to deploy from its root: `app.py` is the entrypoint,
`requirements.txt` declares the Python dependencies, `.streamlit/config.toml`
contains the theme, and the generated evaluation data is committed with the
application.

1. Connect the GitHub repository to [Streamlit Community Cloud](https://share.streamlit.io/).
2. Create an app using branch `main` and entrypoint `app.py`.
3. Use Python 3.12 in Advanced settings.
4. Deploy without a model key to use the deterministic parser. To enable the
   optional language-model translator, add this root-level secret in Advanced
   settings rather than committing it:

   ```toml
   ANTHROPIC_API_KEY = "your-key"
   ```

Root-level Streamlit secrets are available as environment variables, which is
the interface used by this app. See Streamlit's official guides to
[file organisation](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization),
[deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy),
and [secrets management](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management).

## Pages

| Page | What it does |
|---|---|
| **Ask** | Natural-language questions over governed metrics, with session history and replay, grouped suggestions, time windows ("last 6 months", "Q1 2026"), multi-value filters ("East and West"), user-selectable comparison basis for breakdowns, scoped refusals with clickable reframes, JSON/CSV artifact downloads, a copyable result hash, 👍/🚩 correction telemetry, and a Watch action |
| **Monitoring** | A Watched section (user-pinned metric+scope, evaluated with the same materiality logic) plus an impact-ranked anomaly feed; every row drills through to its decomposition with one click |
| **Causal Studio** | Attribution as a review surface: registered events, difference-in-differences proposals rendered as a structured brief (design → computed assumption checks → estimate vs. naive read → variant sensitivity), and an analyst sign-off that stamps the artifact and is recorded in telemetry |
| **Semantic Layer** | Read-only registry browsing for everyone (metrics, variants with owners, sources with known limitations) plus an admin expander that edits the materiality threshold and default variants — persisted to `data/governance_config.json`, reloaded by the layer, every change logged to `data/governance_log.jsonl` |
| **Reliability** | The accuracy record: a standing question set with independently computed expected answers, run twice per question with hash-equality reproducibility, correct-refusal scoring, correction rate from telemetry, trends persisted to `data/eval_history.jsonl`, and session counters for language-model translations |
| **How answers are produced** | User-voiced help: what the tiers mean, how to read the provenance stamp, when to trust a number outright |

## Architecture

```
question ─▶ TRIAGE (classify + parse)          harness/triage.py
              │  Retrieval / Descriptive / Diagnostic / Causal / Predictive / OOS
              │  windows, comparison bases, multi-value filters
              ▼
          RESOLVE (source + variant)           harness/semantic_layer.py
              │  governed defaults (admin-configurable), disclosed; overrides labeled
              ▼
          ENGINE (deterministic)               harness/engines/
              │  descriptive · retrieval · decomposition · causal advisor
              ▼
          DIVERGENCE + CAVEATS                 harness/services.py
              │  engine-aware computed forks; registry-metadata caveats
              ▼
          ANSWER ARTIFACT                      harness/provenance.py
                 code · data version · result hash · tier · stamp · JSON export
```

Design decisions worth knowing:

- **The LLM only translates.** By default the parser is deterministic rules
  (`harness/triage.py`). Supplying an Anthropic API key (sidebar, or
  `ANTHROPIC_API_KEY`) swaps in a model behind the identical `Intent` contract
  (`harness/llm_translator.py`). Every translation is validated against the
  registry — unknown metrics, dimension values, events, windows, or bases are
  rejected and the pipeline falls back to rules, visibly, with the raw error
  preserved in the artifact JSON. Translation latency is recorded in the stamp.
- **Answers are artifacts, not strings.** `harness/provenance.py` — every
  answer carries the code executed, resolved source/variant, data version,
  tier, and a stable result hash, and exports as JSON. Reproducibility is
  enforced, not hoped for. Drill-through steps are real questions, so every
  step of the notice → quantify → localize → attribute loop reproduces.
- **Caveats are computed, not narrated** — built from registry metadata (lag,
  restatements, variant existence, metric substitutions) and calculated
  sensitivity, never free text.
- **Divergence is engine-aware.** Alternate sources/variants are recomputed
  the same way the answer was (level vs level, delta vs delta, same window);
  a fork that can't be recomputed like-for-like is skipped, never approximated.
  Causal designs disclose variant forks via their computed sensitivity instead.
- **Refusals carry reframes.** Predictive questions, unmappable metrics,
  causal questions without a registered event, and out-of-range windows are
  declined with computed re-askable questions — refusing correctly is a scored
  behavior on the Reliability page.
- **Governance changes are provenance.** Admin edits to materiality/default
  variants are validated, persisted to a small JSON config, and logged with
  timestamps; the golden set's expected answers follow the configured defaults
  so the accuracy record stays valid under configuration changes.

## Dataset

`data/generate_demo_data.py` generates the dataset and documents its contract:
baked-in causal effects with true magnitudes (exported to `ground_truth.json`
so the causal advisor can be scored), source pathologies (bias, lag,
restatements) so divergence detection has real work to do, and metric variants
owned by different functions. To use your own data, replace the CSVs, extend
`ground_truth.json` and `pipeline.GOLDEN`, and re-run the Reliability check.

## Tests

`tests/` runs headless via `pytest`:

- `test_ui.py` — `streamlit.testing.v1.AppTest`: every page renders, every
  question class round-trips, drill-through navigation, downloads, history
  replay, sign-off, watchlists, admin, and the accuracy check through the UI.
- `test_copy_audit.py` — AST-based scan of every UI string literal for
  self-narration and phase language.
- `test_golden.py` / `test_capabilities.py` / `test_harness.py` — the golden
  set (100% pass / 100% reproducible, always), windows, bases, multi-filters,
  watchlists, governance, translator validation, divergence unit-consistency.

The suite is hermetic: it strips `ANTHROPIC_API_KEY` and redirects every
mutable state file (`feedback_log.jsonl`, `watchlist.json`,
`governance_config.json`, `governance_log.jsonl`, `eval_history.jsonl`) to a
temp dir.

## Files

```
app.py                          router + chrome
views/
  common.py                     shared answer renderer: chips, stamp, waterfall,
                                causal brief, refusal panel, artifact actions
  ask.py                        home: question box, history rail, suggestions
  monitoring.py                 watched scopes + anomaly feed, drill-through
  causal_studio.py              design proposals + analyst sign-off
  registry.py                   semantic layer browser + governance admin
  reliability.py                accuracy record + run history + telemetry
  help_page.py                  "How answers are produced"
harness/
  semantic_layer.py             sources, metrics + variants, events, resolution,
                                governance config
  triage.py                     classification + intent parsing (rule parser),
                                windows, bases, multi-value filters
  llm_translator.py             optional LLM drop-in behind the same contract
  pipeline.py                   orchestrator + golden set + run history
  provenance.py                 AnswerArtifact: code, hashes, tiers, JSON export
  services.py                   divergence, caveats, anomaly feed, watchlists,
                                telemetry
  engines/
    basic.py                    descriptive (windows, bases) + retrieval
    decomposition.py            contribution analysis (bases, waterfall data)
    causal_advisor.py           DiD design proposals with computed checks
data/
  generate_demo_data.py         dataset generator; doubles as the data contract
  ground_truth.json             documented true effects and data issues
tests/                          pytest suite (headless AppTest + harness units)
```

Note: `harness/__init__.py` contains environment hardening (thread stack size,
pyarrow thread pinning) needed in restricted sandboxes; it is harmless on a
normal workstation. Keep it.
