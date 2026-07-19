# LEM Dataset Loop Design (lem-native training data)

**Status:** DRAFT for review — nothing below is approved until Snider signs off.

**Goal:** The complete training-data loop — capture → score → review → export — as lem features: DuckDB-backed under `~/.lem/`, driven from the TUI and `lem data` verbs, feeding `lem train --sft` and `lem ssd` directly. No external labelling platform; Argilla is retired (2026-07-19, anti-piecemeal: we code the features we need and carry no extras).

**Why now:** the training weeks need the loop, and every stage but the middle already exists in-tree. This design adds the dataset store and the review surface, and strings the existing parts together; it deliberately invents nothing that has a proven in-repo precedent to reuse.

---

## What already exists (reused, not rebuilt)

| Stage | Existing seam | Where |
|---|---|---|
| Capture doctrine | Capture-first (#97): "the model's response IS the data point" — raw text captured at birth, scores attached later, never inline | `go/train/capture.go` (`CaptureRow{Step, Prompt, Text, At}`, JSONL sidecar) |
| Sampling | `lem ssd` self-distillation sampling (sample the frozen base, capture the trace); `lem train --sft` eval captures | `cli/main.go`, `go/train/` |
| Scoring (heuristic) | The lem-scorer: in-process, non-LLM, `ScorePair`/`DiffResult` — its package doc names "the training-data validation path" as a first-class consumer. Axes: LEK composite 0–100, Sycophancy, Imprint, Hostility, Suggestions | `go/eval/score/lek/` |
| Scoring (cascade) | The checkpoint-oracle cascade over score-vector time series | `go/train/score_cascade.go` |
| Judge scoring | A local model behind the shared serial lane (one resident model; jobs queue) | the TUI's scheduler-wrapped lane |
| Welfare screen | `go/welfare` mediate + slurs matcher (catalogue pending #29) | `go/welfare/` |
| Store pattern | Versioned DuckDB migrations + typed records + repository, medium-backed `~/.lem` contract | `cli/tui/{records,migrations,repository,paths}.go` |
| Conversation source | Per-user chat history DuckDB (sessions/turns) | `go/serving/chathistory/` (`~/Lethean/lem/users/<id>/chats.duckdb`) |
| Export consumer | `lem train --sft` eats `{messages}` JSONL; `lem ssd` traces | `cli/`, `go/train/` |
| Provenance rule | "The model's response IS the data point — fingerprint at generation; never re-score an old response against a later model state" | house rule; model fingerprints exist in the engine |

## Architecture

Two-module split, the proven `orchestrator.Store` shape from the CoreAgent port:

- **Root module — `go/dataset`** (engine-free, DuckDB-free, portable): domain types, the `dataset.Store` interface, ingest normalisation, scoring orchestration contracts, export writers, manifest hashing. Imports `go/eval/score/lek`, `go/welfare`, `go/train` types where needed. Never imports the TUI or a SQL driver. Public fallible functions return `core.Result`.
- **CLI module** — the DuckDB `dataset.Store` implementation (schema migration version alongside the existing TUI migrations), the `lem data` verbs, and the TUI review surface. Consumes the root package; supplies the driver, exactly as `cli/tui/agentstore.go` does for the orchestrator.

## Storage

**Decision: a separate `~/.lem/datasets.duckdb`**, not the orchestration `lem.duckdb`.

Rationale: dataset bulk (10⁵–10⁶ rows of text) has a different lifecycle from orchestration state — backed up, copied between boxes, potentially handed to another machine whole; a bloated or damaged dataset file must never take the agent/TUI state down with it. The alternative (one shared `lem.duckdb`, the CoreAgent design's choice for orchestration state) is rejected for bulk data on those grounds. Same medium-backed path contract as everything else under `~/.lem/`; only the DuckDB adapter receives the resolved filename.

## Domain model

All rows carry `created_at`; nothing is ever hard-deleted from normal actions — archive by flag + timestamp (the house rule). Lookup-heavy JSON payloads live in DuckDB JSON columns; load-bearing fields are real columns.

- **Dataset** — id (uuid), slug (unique, human), title, purpose note, created/archived.
- **Item** — id, dataset_id, **kind** (`pair` = prompt/response · `messages` = full chat turns · `trace` = SSD trace), **content** (JSON, kind-shaped), **source** (`capture:serve`, `capture:train`, `import:chathistory`, `import:jsonl`, `ssd`), source_ref (e.g. chat session/turn id), **model_fingerprint** (of the GENERATING model, stamped at birth; empty for human/imported text), content_hash (dedupe), parent_item_id (edit lineage), archived flag.
- **Score** — id, item_id, **kind** (`lek`, `hostility`, `sycophancy`, `judge:<name>`, …), **value** (float), payload (full JSON — e.g. the whole `ScoreResult`), **scorer identity** (scorer name + version OR judge model fingerprint), created_at. **Append-only**: re-scoring adds a row, never overwrites — the score history IS the record, and a score always names what produced it (the fingerprint rule).
- **Review** — item_id, **status** (`pending` → `approved` | `rejected` | `quarantined`), reviewer (`snider` | `auto:welfare` | `auto:threshold`), note, created_at. Latest row wins; history kept. **Edit-and-approve** never mutates: it creates a derived Item (`parent_item_id` set), archives the original, approves the child.
- **Export** — id, dataset_id, format, filter description, item count, output path, **manifest hash** (sha256 over the ordered item hashes — the receipt that a training run can name exactly what it saw), created_at.

## Ingest paths

1. **Serve capture (opt-in, default OFF):** `lem serve --capture <dataset-slug>` tees each completed (prompt, response) into the dataset with the serving model's fingerprint. Privacy default: no flag, no capture — capture is a deliberate act.
2. **Chathistory import:** `lem data import --chats <user> [--session <id>]` reads the existing per-user chats.duckdb read-only and normalises turns → `messages` items.
3. **JSONL import:** `lem data import --jsonl <path>` accepting `{messages}` and `{prompt, response}` shapes; also the existing `CaptureRow` sidecar shape (`{step, prompt, text, at_unix}`) so historic training captures load unchanged.
4. **SSD traces:** `lem ssd` gains `--dataset <slug>` to land its sampled traces directly.

Every ingest path runs the **welfare screen** (`go/welfare`) at the door: a hit sets status `quarantined` with `reviewer=auto:welfare` — visible, reviewable, never silently dropped. (The slur catalogue itself remains #29 — Snider-curated.)

## Scoring

`lem data score <dataset> [--kind lek|judge:<name>] [--filter …]`:

- **Heuristic tier (default, no model needed):** `lek.ScorePair` per item → one `lek` Score row (composite in `value`, full `ScoreResult` in payload) plus derived `hostility`/`sycophancy` rows for direct filtering.
- **Judge tier (optional):** a named judge prompt template run through the shared serial model lane (scoring queues like any generation job; the TUI stays responsive; only one model resident). Judge scores store the judge model's fingerprint as scorer identity.
- Auto-review thresholds are **explicit, never implicit**: `lem data score --auto-approve 'lek>=80' --auto-reject 'hostility>=0.7'` writes Review rows with `reviewer=auto:threshold` and the expression in the note. No silent policy.

## Review surface (TUI)

**Decision: a fifth primary panel, `Data`**, beside Chat/Work/Models/Service.

Rationale: during training weeks, review is a primary daily workflow, not an inspector detail; the Work panel proved the pattern (list + detail + contextual actions + palette mirroring). Alternative considered: an inspector mode under an existing panel — rejected as burying the loop's main human step. (This extends `panelID`; the four-panel geometry generalises — flagged for review since it touches the approved TUI design's "four primary panels" line.)

- **List:** items with status, kind, source, top scores; filters by dataset / status / score expression / source; sort by score/date. Keyboard-first (the j/k + action-keys idiom).
- **Detail:** rendered prompt/response or turn stack (existing Glamour path), full score breakdown (axes from payload), lineage (parent/derived), welfare flags.
- **Actions:** approve / reject / quarantine-clear (requires note) / **edit** (opens the editor seam → derived item, original archived) / tag / bulk-apply-to-current-filter (with count confirmation).
- Palette mirrors every available action; unavailable states render honestly (the agentcap pattern).

## Export

`lem data export <dataset> --format sft-jsonl [--filter 'status=approved'] --out <path>`:

- **Formats v1:** `sft-jsonl` (`{messages}` — the `lem train --sft` contract), `pairs-jsonl` (`{prompt, response}`), `capture-jsonl` (round-trips the CaptureRow shape).
- Deterministic ordering (created_at, id) → **manifest** row + sidecar JSON: counts, filter expression, per-item content hashes, the manifest hash. A training run's provenance is one hash.
- Default filter `status=approved`; exporting anything else requires the explicit filter (no accidental training on unreviewed data).

## Verbs (summary)

`lem data create|list|stats|import|score|export|archive` — plus `lem data review` which opens the TUI on the Data panel. `lem serve --capture <slug>`, `lem ssd --dataset <slug>`.

## Non-goals (v1)

Web UI · multi-user/remote sync · DPO/RLHF pair formats (v2 when training needs them) · dataset packing into `.model` packs (the vindex sibling — future) · Argilla data salvage (qa box is down; revisit only if the old data proves needed).

## Constraints (house, binding)

`~/.lem/` only, medium-backed paths, no `--home` override · `dappco.re/go` wrappers; `core.Result`; no banned stdlib imports in production code · root module stays portable (no DuckDB/TUI deps); CLI owns the drivers · archive-by-flag, never hard-delete · one test per symbol per variant, `_Good/_Bad/_Ugly`, Examples for public symbols; behavioural test before implementation · every score names its scorer + fingerprint; no re-scoring in place · welfare screen at every ingest door · truthful partials: a stage that can't land honestly refuses loudly and the plan records the boundary.

## Open questions for review

1. **Data panel vs inspector mode** — the panel is my recommendation (primary workflow), but it amends the approved four-panel TUI design.
2. **Separate `datasets.duckdb`** vs folding into `lem.duckdb` — separate is my recommendation (bulk/lifecycle/blast-radius), stated above.
3. **Serve capture default** — OFF (explicit `--capture`) is my recommendation; confirm the privacy stance.
4. **Judge prompts** — where do the named judge templates live: in-repo (versioned, shared) or `~/.lem/judges/` (user-owned)? My lean: in-repo defaults + user overrides in `~/.lem/judges/`.
