// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"bytes"
	"context"
	"io"
	"iter"
	"os"
	"path/filepath"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/dataset"
)

// ---- fakes ----

// captureFakeModel is a minimal inference.TextModel double — mirrors
// go/serving/welfare_guard_test.go's welfareFakeModel shape (the same
// TextModel-decorator wrapping this file's captureTextModel is modelled on),
// reimplemented here since it is unexported in another package. Chat records
// every call and replays convoTokens.
type captureFakeModel struct {
	convoTokens []string
	convoCalls  [][]inference.Message
}

func fakeTokenSeq(texts ...string) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		for i, text := range texts {
			if !yield(inference.Token{ID: int32(i + 1), Text: text}) {
				return
			}
		}
	}
}

func (f *captureFakeModel) Chat(_ context.Context, messages []inference.Message, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	f.convoCalls = append(f.convoCalls, append([]inference.Message(nil), messages...))
	return fakeTokenSeq(f.convoTokens...)
}
func (f *captureFakeModel) Generate(context.Context, string, ...inference.GenerateOption) iter.Seq[inference.Token] {
	return fakeTokenSeq(f.convoTokens...)
}
func (f *captureFakeModel) Classify(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Ok(nil)
}
func (f *captureFakeModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Ok(nil)
}
func (f *captureFakeModel) ModelType() string                  { return "capture-fake" }
func (f *captureFakeModel) Info() inference.ModelInfo          { return inference.ModelInfo{} }
func (f *captureFakeModel) Metrics() inference.GenerateMetrics { return inference.GenerateMetrics{} }
func (f *captureFakeModel) Err() core.Result                   { return core.Ok(nil) }
func (f *captureFakeModel) Close() core.Result                 { return core.Ok(nil) }

var _ inference.TextModel = (*captureFakeModel)(nil)

// captureFakeSchedulerModel adds inference.SchedulerModel to captureFakeModel
// — wrapCapture must pick the captureSchedulerModel variant for a model
// shaped like this one, exactly as go/serving/welfare_guard.go's own
// SchedulerModel-detection wrap does.
type captureFakeSchedulerModel struct {
	*captureFakeModel
	scheduleCalls []inference.ScheduledRequest
}

func (f *captureFakeSchedulerModel) Schedule(_ context.Context, req inference.ScheduledRequest) (inference.RequestHandle, <-chan inference.ScheduledToken, error) {
	f.scheduleCalls = append(f.scheduleCalls, req)
	out := make(chan inference.ScheduledToken, len(f.convoTokens))
	for i, text := range f.convoTokens {
		out <- inference.ScheduledToken{RequestID: req.ID, Token: inference.Token{ID: int32(i + 1), Text: text}}
	}
	close(out)
	return inference.RequestHandle{ID: req.ID}, out, nil
}

var _ inference.SchedulerModel = (*captureFakeSchedulerModel)(nil)

// failingStore wraps a dataset.Store and makes ItemAppend always fail — the
// fake "a failing store leaves serving working" proves capture failures
// never propagate into the token stream.
type failingStore struct {
	dataset.Store
}

func (failingStore) ItemAppend(dataset.Item) core.Result {
	return core.Fail(core.NewError("simulated store failure"))
}

func newSeededMemoryStore(t *testing.T, datasetID string) *dataset.MemoryStore {
	t.Helper()
	store := dataset.NewMemoryStore()
	if r := store.DatasetCreate(dataset.Dataset{ID: datasetID, Slug: "capture-fake-ds", Title: "capture fake", CreatedAt: time.Now()}); !r.OK {
		t.Fatalf("seed dataset: %v", r.Value)
	}
	return store
}

func drainTokens(seq iter.Seq[inference.Token]) []string {
	var texts []string
	for tok := range seq {
		texts = append(texts, tok.Text)
	}
	return texts
}

// ---- modelFingerprint ----

func TestModelFingerprint(t *testing.T) {
	t.Run("Good/absolute path", func(t *testing.T) {
		got := modelFingerprint("/models/gemma-4-e2b-it-4bit")
		if got != "path:/models/gemma-4-e2b-it-4bit" {
			t.Errorf("modelFingerprint = %q", got)
		}
	})
	t.Run("Good/relative path resolves absolute", func(t *testing.T) {
		got := modelFingerprint("relative/model/dir")
		if !core.HasPrefix(got, "path:") || !core.PathIsAbs(got[len("path:"):]) {
			t.Errorf("modelFingerprint(relative) = %q, want an absolute path: prefix", got)
		}
	})
	t.Run("Bad/empty path", func(t *testing.T) {
		if got := modelFingerprint(""); got != "" {
			t.Errorf("modelFingerprint(\"\") = %q, want empty", got)
		}
	})
	t.Run("Ugly/same path is stable", func(t *testing.T) {
		a := modelFingerprint("/models/x")
		b := modelFingerprint("/models/x")
		if a != b {
			t.Errorf("modelFingerprint not stable: %q != %q", a, b)
		}
	})
}

// ---- wrapCapture / captureTextModel.Chat ----

func TestWrapCapture_Chat_Good(t *testing.T) {
	const datasetID = "ds-good"
	store := newSeededMemoryStore(t, datasetID)
	fake := &captureFakeModel{convoTokens: []string{"hel", "lo "}}
	wrapped := wrapCapture(fake, store, datasetID, "path:/models/x", io.Discard)

	texts := drainTokens(wrapped.Chat(context.Background(), []inference.Message{{Role: "user", Content: "hi"}}))
	if got := core.Join("", texts...); got != "hello " {
		t.Fatalf("streamed tokens = %q, want %q (must pass through unchanged)", got, "hello ")
	}

	items := core.MustCast[[]dataset.Item](store.Items(dataset.ItemFilter{DatasetID: datasetID}))
	if len(items) != 1 {
		t.Fatalf("items landed = %d, want 1", len(items))
	}
	item := items[0]
	if item.Kind != dataset.KindPair {
		t.Errorf("Kind = %s, want pair", item.Kind)
	}
	if item.Source != dataset.SourceCaptureServe {
		t.Errorf("Source = %s, want %s", item.Source, dataset.SourceCaptureServe)
	}
	if item.ModelFingerprint != "path:/models/x" {
		t.Errorf("ModelFingerprint = %q, want path:/models/x", item.ModelFingerprint)
	}
	var pc dataset.PairContent
	if r := core.JSONUnmarshal(item.Content, &pc); !r.OK {
		t.Fatalf("decode content: %v", r.Value)
	}
	if pc.Response != "hello " {
		t.Errorf("captured response = %q, want %q", pc.Response, "hello ")
	}
	if !core.Contains(pc.Prompt, "hi") {
		t.Errorf("captured prompt = %q, want it to contain the user turn", pc.Prompt)
	}
}

func TestWrapCapture_Chat_EmptyMessages(t *testing.T) {
	const datasetID = "ds-empty"
	store := newSeededMemoryStore(t, datasetID)
	fake := &captureFakeModel{convoTokens: []string{"still streams"}}
	wrapped := wrapCapture(fake, store, datasetID, "path:/models/x", io.Discard)

	texts := drainTokens(wrapped.Chat(context.Background(), nil))
	if got := core.Join("", texts...); got != "still streams" {
		t.Fatalf("streamed tokens = %q, want pass-through even with no messages", got)
	}
	items := core.MustCast[[]dataset.Item](store.Items(dataset.ItemFilter{DatasetID: datasetID}))
	if len(items) != 0 {
		t.Fatalf("items landed = %d, want 0 (no user content to capture)", len(items))
	}
}

// TestWrapCapture_Chat_FailingStore is the "a failing store (fake) leaves
// serving working" proof: every token still reaches the caller even though
// every dataset write fails.
func TestWrapCapture_Chat_FailingStore(t *testing.T) {
	const datasetID = "ds-failing"
	inner := newSeededMemoryStore(t, datasetID)
	store := failingStore{Store: inner}
	fake := &captureFakeModel{convoTokens: []string{"serving ", "still ", "works"}}

	var log bytes.Buffer
	wrapped := wrapCapture(fake, store, datasetID, "path:/models/x", &log)

	texts := drainTokens(wrapped.Chat(context.Background(), []inference.Message{{Role: "user", Content: "hi"}}))
	if got := core.Join("", texts...); got != "serving still works" {
		t.Fatalf("streamed tokens = %q, want the full reply despite the store failing", got)
	}
	if !core.Contains(log.String(), "capture write failed") {
		t.Errorf("log = %q, want a capture-failed notice", log.String())
	}
}

func TestWrapCapture_Unwrap(t *testing.T) {
	const datasetID = "ds-unwrap"
	store := newSeededMemoryStore(t, datasetID)
	fake := &captureFakeModel{}
	wrapped := wrapCapture(fake, store, datasetID, "fp", io.Discard)

	unwrappable, ok := wrapped.(interface{ Unwrap() inference.TextModel })
	if !ok {
		t.Fatal("wrapped model does not implement Unwrap")
	}
	if unwrappable.Unwrap() != inference.TextModel(fake) {
		t.Error("Unwrap did not return the inner model")
	}
}

// TestWrapCapture_SchedulerModel proves a scheduled model is wrapped so the
// tap still fires — the exact bug class go/serving/welfare_guard.go's own
// doc comment documents: inference.As walks Unwrap to reach
// SchedulerModel.Schedule, so a plain (non-Schedule) wrapper would be
// bypassed entirely for every scheduled request.
func TestWrapCapture_SchedulerModel(t *testing.T) {
	const datasetID = "ds-sched"
	store := newSeededMemoryStore(t, datasetID)
	fake := &captureFakeSchedulerModel{captureFakeModel: &captureFakeModel{convoTokens: []string{"sched", "uled"}}}

	wrapped := wrapCapture(fake, store, datasetID, "path:/models/sched", io.Discard)
	sched, ok := inference.As[inference.SchedulerModel](wrapped)
	if !ok {
		t.Fatal("wrapped model does not resolve as a SchedulerModel — the scheduled path would silently skip capture")
	}

	_, stream, err := sched.Schedule(context.Background(), inference.ScheduledRequest{
		ID: "req-1", Messages: []inference.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("Schedule: %v", err)
	}
	var got core.Builder
	for tok := range stream {
		got.WriteString(tok.Token.Text)
	}
	if got.String() != "scheduled" {
		t.Fatalf("streamed tokens = %q, want %q", got.String(), "scheduled")
	}

	items := core.MustCast[[]dataset.Item](store.Items(dataset.ItemFilter{DatasetID: datasetID}))
	if len(items) != 1 {
		t.Fatalf("items landed = %d, want 1", len(items))
	}
	if items[0].Source != dataset.SourceCaptureServe {
		t.Errorf("Source = %s, want %s", items[0].Source, dataset.SourceCaptureServe)
	}
}

// ---- resolveCaptureDataset / buildCaptureLoader (real store, HOME-redirected) ----

func TestResolveCaptureDataset(t *testing.T) {
	dataTestHome(t)
	t.Run("Good", func(t *testing.T) {
		mustCreate(t, "resolve-me")
		store, ds, err := resolveCaptureDataset("resolve-me")
		if err != nil {
			t.Fatalf("resolveCaptureDataset: %v", err)
		}
		defer store.Close()
		if ds.Slug != "resolve-me" {
			t.Errorf("Slug = %q, want resolve-me", ds.Slug)
		}
	})
	t.Run("Bad/unknown slug", func(t *testing.T) {
		store, _, err := resolveCaptureDataset("does-not-exist-at-all")
		if err == nil {
			t.Fatal("resolveCaptureDataset over an unknown slug succeeded, want an error")
		}
		if store != nil {
			t.Error("resolveCaptureDataset returned a non-nil store alongside an error")
		}
	})
}

// TestBuildCaptureLoader_DefaultOff proves --capture's empty value never
// opens the dataset store at all — "OFF without the flag (the approved
// privacy default)" — checked structurally: datasets.duckdb must not exist
// on disk after the call.
func TestBuildCaptureLoader_DefaultOff(t *testing.T) {
	home := dataTestHome(t)
	loader, store, err := buildCaptureLoader("", io.Discard)
	if err != nil || loader != nil || store != nil {
		t.Fatalf("buildCaptureLoader(\"\") = (%v, %v, %v), want (nil, nil, nil)", loader, store, err)
	}
	if _, statErr := os.Stat(filepath.Join(home, ".lem", "datasets.duckdb")); !os.IsNotExist(statErr) {
		t.Fatalf("datasets.duckdb exists after a capture-off call: stat err = %v", statErr)
	}
}

func TestBuildCaptureLoader_Good(t *testing.T) {
	dataTestHome(t)
	mustCreate(t, "capture-target")
	loader, store, err := buildCaptureLoader("capture-target", io.Discard)
	if err != nil {
		t.Fatalf("buildCaptureLoader: %v", err)
	}
	defer store.Close()
	if loader == nil {
		t.Fatal("loader is nil for a resolved --capture slug")
	}
	if store == nil {
		t.Fatal("store is nil for a resolved --capture slug")
	}
}

func TestBuildCaptureLoader_Bad(t *testing.T) {
	dataTestHome(t)
	loader, store, err := buildCaptureLoader("no-such-dataset", io.Discard)
	if err == nil {
		t.Fatal("buildCaptureLoader over a missing dataset succeeded, want an error")
	}
	if loader != nil || store != nil {
		t.Fatalf("buildCaptureLoader on failure = (%v, %v), want (nil, nil)", loader, store)
	}
}

// ---- ingestTraceItem / ingestSSDTrace ----

func TestIngestTraceItem(t *testing.T) {
	const datasetID = "ds-trace"
	store := newSeededMemoryStore(t, datasetID)
	content := []byte(`{"sample":"trace-row"}`)

	t.Run("Good", func(t *testing.T) {
		item, deduped, err := ingestTraceItem(store, datasetID, content, dataset.SourceSSD, "1", "path:/models/base")
		if err != nil {
			t.Fatalf("ingestTraceItem: %v", err)
		}
		if deduped {
			t.Error("first ingest reported deduped")
		}
		if item.Kind != dataset.KindTrace {
			t.Errorf("Kind = %s, want trace", item.Kind)
		}
		if item.ModelFingerprint != "path:/models/base" {
			t.Errorf("ModelFingerprint = %q", item.ModelFingerprint)
		}
	})

	t.Run("Good/dedupes identical content", func(t *testing.T) {
		_, deduped, err := ingestTraceItem(store, datasetID, content, dataset.SourceSSD, "2", "path:/models/base")
		if err != nil {
			t.Fatalf("ingestTraceItem (repeat): %v", err)
		}
		if !deduped {
			t.Error("repeat ingest of identical content was not reported as deduped")
		}
	})

	t.Run("Bad/non-JSON content is rejected", func(t *testing.T) {
		_, _, err := ingestTraceItem(store, datasetID, []byte("not json"), dataset.SourceSSD, "3", "path:/models/base")
		if err == nil {
			t.Fatal("ingestTraceItem accepted non-JSON trace content")
		}
	})
}

func writeSSDCaptureFixture(t *testing.T, dir string, lines ...string) {
	t.Helper()
	var buf bytes.Buffer
	for _, line := range lines {
		buf.WriteString(line)
		buf.WriteByte('\n')
	}
	if err := os.WriteFile(filepath.Join(dir, ssdCaptureSidecarName), buf.Bytes(), 0o644); err != nil {
		t.Fatalf("write ssd capture fixture: %v", err)
	}
}

func TestIngestSSDTrace_Good(t *testing.T) {
	dataTestHome(t)
	mustCreate(t, "ssd-trace-ds")
	dir := t.TempDir()
	writeSSDCaptureFixture(t, dir, `{"step":0,"prompt":"p1","text":"r1"}`, `{"step":1,"prompt":"p2","text":"r2"}`)

	var log bytes.Buffer
	if err := ingestSSDTrace(dir, "ssd-trace-ds", "/models/ssd-base", &log); err != nil {
		t.Fatalf("ingestSSDTrace: %v", err)
	}
	if !core.Contains(log.String(), "2 trace items ingested") {
		t.Errorf("log = %q, want a 2-ingested summary", log.String())
	}

	store, closeStore := openTestStore(t)
	defer closeStore()
	ds, err := resolveDatasetSlug(store, "ssd-trace-ds")
	if err != nil {
		t.Fatalf("resolve: %v", err)
	}
	items := core.MustCast[[]dataset.Item](store.Items(dataset.ItemFilter{DatasetID: ds.ID}))
	if len(items) != 2 {
		t.Fatalf("items landed = %d, want 2", len(items))
	}
	for _, item := range items {
		if item.Kind != dataset.KindTrace {
			t.Errorf("Kind = %s, want trace", item.Kind)
		}
		if item.Source != dataset.SourceSSD {
			t.Errorf("Source = %s, want %s", item.Source, dataset.SourceSSD)
		}
		if !core.HasPrefix(item.ModelFingerprint, "path:") {
			t.Errorf("ModelFingerprint = %q, want a path: prefix", item.ModelFingerprint)
		}
	}
}

func TestIngestSSDTrace_Bad(t *testing.T) {
	dataTestHome(t)
	t.Run("unknown dataset", func(t *testing.T) {
		dir := t.TempDir()
		writeSSDCaptureFixture(t, dir, `{"step":0,"prompt":"p","text":"r"}`)
		if err := ingestSSDTrace(dir, "does-not-exist", "/models/x", io.Discard); err == nil {
			t.Fatal("ingestSSDTrace against an unknown dataset succeeded")
		}
	})
	t.Run("missing sidecar file", func(t *testing.T) {
		mustCreate(t, "ssd-trace-missing-sidecar")
		if err := ingestSSDTrace(t.TempDir(), "ssd-trace-missing-sidecar", "/models/x", io.Discard); err == nil {
			t.Fatal("ingestSSDTrace against a missing sidecar succeeded")
		}
	})
}
