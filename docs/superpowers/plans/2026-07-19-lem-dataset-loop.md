# LEM Dataset Loop Implementation Plan

> **For agentic workers:** work task-by-task; each task's focused tests pass before its commit, and each commit contains only that task's files. Checkbox (`- [ ]`) syntax tracks completion. A stage that cannot land honestly refuses loudly and this plan records the boundary — a truthful partial beats a fake whole.

**Goal:** The lem-native training-data loop — capture → score → review → export — per the approved design [`../specs/2026-07-19-lem-dataset-loop-design.md`](../specs/2026-07-19-lem-dataset-loop-design.md). Root package `go/dataset` (portable), DuckDB store + verbs + TUI Data panel in the CLI module.

**Architecture:** the CoreAgent split — `go/dataset` owns domain, store contract, ingest normalisation, scoring orchestration, export + manifests; the CLI module supplies the DuckDB driver (`~/.lem/datasets.duckdb`), the `lem data` verbs, the serve/ssd capture taps, and the fifth TUI panel. Root lands and gates before the CLI consumes it.

**Approved decisions (binding):** fifth TUI `Data` panel · separate `~/.lem/datasets.duckdb` · serve capture default OFF (explicit `--capture <slug>`) · judge templates in-repo with `~/.lem/judges/` overrides.

## Global Constraints

- The design doc is authority; implementation pressure that conflicts with it stops and updates the design first.
- Production code: `dappco.re/go` wrappers (no banned stdlib imports), `core.Result` with `OK` inspected before `Value`, UK English, EUPL-1.2 SPDX headers on new files.
- `go/dataset` never imports the TUI, a SQL driver, or engine/metal. The CLI module never re-implements domain logic.
- All state below `~/.lem/` through the medium-backed path contract; only the DuckDB adapter receives a resolved filename. No `--home` override.
- Archive-by-flag everywhere; no hard deletes from normal actions. Scores append-only with scorer identity + model fingerprint; no in-place re-scoring.
- Welfare screen at every ingest door; hits quarantine visibly, never drop silently.
- Tests: behavioural test before each implementation change; one test per symbol per variant, `_Good/_Bad/_Ugly`, beside the source; Example tests for public symbols. Coverage target 95% for `go/dataset`.
- A task is not complete until its focused tests pass and its commit contains only that task's files.

## Target File Map

Root module (`go/dataset/`): `dataset.go` (Dataset/Item/Score/Review/Export types, kinds, statuses, content shapes), `hash.go` (content + manifest hashing), `store.go` (the Store interface + query filters), `ingest.go` (JSONL/CaptureRow/chathistory normalisation), `screen.go` (welfare-at-the-door), `score.go` (heuristic tier + auto-threshold expressions + judge contract), `export.go` (writers + manifests), `doc.go` — each with sibling `_test.go` and Example tests.

CLI module: `tui/datasetstore.go` (DuckDB Store impl + migrations for `datasets.duckdb`), `tui/paths.go` (add the datasets DB + judges paths), `data.go` (the `lem data` verb family), `serve.go`/`ssd.go` (capture taps), `tui/tabs.go` + `tui/datapanel.go` (+ palette/keymap/app wiring), `judges/` (in-repo default templates), READMEs.

---

### Task 1: Domain types, content shapes, hashing (`go/dataset` core)

- [x] Dataset/Item/Score/Review/Export types with kinds (`pair`/`messages`/`trace`), sources, statuses, reviewer identities per the design; content-shape validation per kind (a `pair` item must carry prompt+response; `messages` must carry role-alternating turns; `trace` opaque-but-nonempty).
- [x] `content_hash` (canonicalised content) + manifest hash (sha256 over ordered item hashes); deterministic across runs.
- [x] Tests: shape validation Good/Bad/Ugly per kind; hash determinism + inequality cases; Examples.

### Task 2: Store contract + ingest normalisation

- [ ] `Store` interface: dataset CRUD-by-flag, item append/get/list with filters (dataset, status, kind, source, score expressions, archived), score append, review append (latest-wins reads), export record append. An in-memory fake Store for root-module tests.
- [ ] Ingest: `{messages}` JSONL, `{prompt,response}` JSONL, `CaptureRow` JSONL (`{step,prompt,text,at_unix}`), and chathistory session/turn rows → normalised Items with source refs + fingerprints; dedupe by content_hash within a dataset (a duplicate ingest is a counted no-op, not an error).
- [ ] Tests: each shape round-trips; malformed rows are loud per-row errors with honest counts (ingested/skipped/dedup); Examples.

### Task 3: Welfare screen + scoring orchestration

- [ ] Every ingest path routes through the welfare screen; a hit writes `quarantined` review (`auto:welfare`) and the item still lands (visible, reviewable).
- [ ] Heuristic tier: `lek.ScorePair` per item → `lek` score row (composite value + full payload) + derived `hostility`/`sycophancy` rows; scorer identity = scorer name + version.
- [ ] Auto-threshold expressions (`lek>=80`, `hostility>=0.7`): a tiny explicit parser (field, op, number — nothing more), applied ONLY when flags are passed; review rows carry `auto:threshold` + the expression text.
- [ ] Judge contract: the interface a CLI-side driver implements (template name → prompt render → generation → parsed score); root defines + tests the contract with a fake judge, drivers live CLI-side.
- [ ] Tests: quarantine visibility, score append-only (re-score adds), threshold parser Good/Bad/Ugly (reject anything beyond the grammar), fake-judge round-trip; Examples.

### Task 4: Export writers + manifests

- [ ] `sft-jsonl` (`{messages}` — pairs converted via the standard user/assistant wrap), `pairs-jsonl`, `capture-jsonl` writers; default filter `status=approved`, anything else requires the explicit filter.
- [ ] Deterministic ordering (created_at, id); manifest row + sidecar JSON (counts, filter expression, per-item hashes, manifest hash).
- [ ] Tests: each format golden-checked; the approved-only default proven; manifest hash stability; Examples. Root-module gate: `go build ./... && go vet ./dataset/ && go test -count=1 ./dataset/` clean, coverage ≥95%.

### Task 5: DuckDB store + paths (CLI)

- [ ] `datasets.duckdb` at the medium-backed path; its own migration versioning (v1) following the `cli/tui/migrations.go` pattern; typed records mirroring the domain; implements `dataset.Store`.
- [ ] Score-expression filters compile to SQL over the latest-score-per-kind view; latest-review-wins views.
- [ ] Tests: full Store conformance against the root fake's behaviour (same test table driven over both), temp-root isolation, migration idempotence.

### Task 6: `lem data` verbs

- [ ] `create|list|stats|import|score|export|archive` + `review` (prints how to open the TUI Data panel); `--json` for scripting; honest per-row import counts; score/export flags per the design.
- [ ] Wires the concrete lek scorer + welfare screen + DuckDB store; exit codes truthful.
- [ ] Tests: verb-level Good/Bad/Ugly against a temp `~/.lem` root (the existing CLI test idiom).

### Task 7: Capture taps

- [ ] `lem serve --capture <slug>`: tee completed (prompt, response) pairs with the serving model's fingerprint; OFF without the flag; capture failures log-and-continue (never break serving).
- [ ] `lem ssd --dataset <slug>`: land sampled traces as `trace` items.
- [ ] Tests: default-off proven; tap writes with fingerprint; serve path unaffected when capture errors.

### Task 8: TUI Data panel

- [ ] Fifth `panelID` + tab; list with filters (dataset/status/score/source) + keyboard idiom; detail view (Glamour-rendered content, score breakdown, lineage, welfare flags).
- [ ] Actions: approve / reject / quarantine-clear (note required) / edit-as-derived (original archived) / tag / bulk-apply-to-filter with count confirmation; palette mirroring; honest unavailable states.
- [ ] Tests: panel state machine, action wiring, bulk-confirm; render snapshots per the existing layout-test idiom.

### Task 9: Judge tier driver

- [ ] In-repo default templates under `judges/` + `~/.lem/judges/` overrides (medium-backed); the driver renders, runs through the shared serial model lane (queued jobs, TUI stays live), parses, stores with the judge model fingerprint.
- [ ] `lem data score --kind judge:<name>` wired end-to-end.
- [ ] Tests: template resolution order (override wins), driver against a stub lane, malformed judge output is a loud per-item error.

### Task 10: Docs + plan completion

- [ ] `cli/tui/README.md` + `docs/cmd-lem.md` sections; design doc kept in sync with anything learned.
- [ ] Coverage receipt for `go/dataset` (≥95%) + CLI additions measured and recorded honestly (below-bar numbers recorded, not hidden).
- [ ] Plan boxes ticked to delivered reality; completion commit.

## Execution order

Tasks 1→4 are one root-module campaign (single lane, sequential commits). Task 5 gates on 4; 6–7 on 5; 8–9 parallel after 6; 10 last. GPU is never required; judge-tier live runs are orchestrator merge gates.
