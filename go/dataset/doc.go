// SPDX-Licence-Identifier: EUPL-1.2

// Package dataset is the lem-native training-data loop's root domain
// package: capture → score → review → export, per
// docs/superpowers/specs/2026-07-19-lem-dataset-loop-design.md.
//
// This package is engine-free and portable: no DuckDB driver, no TUI, no
// engine/metal. It owns the domain types (Dataset/Item/Score/Review/
// Export), the [Store] contract a driver implements, ingest
// normalisation, the welfare screen at the ingest door, heuristic +
// judge-tier scoring orchestration, and export writers with manifest
// hashing. The CLI module supplies the DuckDB [Store] implementation
// (~/.lem/datasets.duckdb), the `lem data` verbs, and the TUI review
// surface — this package never imports any of that; it only defines the
// shape the CLI plugs into, exactly as go/agent/orchestrator does for the
// agent orchestration store.
//
// Reused seams (never rebuilt here): [dappco.re/go/inference/eval/score/lek]
// for heuristic scoring (ScorePair), [dappco.re/go/inference/welfare] for
// the ingest-door hostility/slur screen, and the CaptureRow JSONL wire
// shape from dappco.re/go/inference/train (mirrored by field name here,
// not imported, so this package never pulls in train's engine
// dependencies).
//
//	store := dataset.NewMemoryStore()
//	ds := dataset.Dataset{ID: dataset.NewID(), Slug: "evening-vents", Title: "Evening vents", CreatedAt: time.Now()}
//	_ = store.DatasetCreate(ds)
//	report := dataset.IngestPairsJSONL(store, ds.ID, reader, dataset.IngestOptions{})
package dataset
