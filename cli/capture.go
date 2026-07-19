// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"bufio"
	"context"
	"io"
	"iter"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/cli/tui"
	"dappco.re/go/inference/dataset"
	"dappco.re/go/inference/serving"
)

// capture.go wires the two `lem serve --capture <slug>` / `lem ssd --dataset
// <slug>` taps (Task 7 of docs/superpowers/plans/2026-07-19-lem-dataset-loop.md)
// on top of go/dataset's exported ingest primitives — nothing here reaches into
// go/serving or go/train internals, since neither exposes a completion hook of
// its own (ServeConfig carries no OnComplete-shaped field; RunSSDCommand
// returns only an error). Both taps stay entirely CLI-side:
//
//   - serve: a inference.TextModel decorator injected through
//     serving.ServeConfig.Loader, mirroring go/serving/welfare_guard.go's
//     TextModel-decorator shape one layer up (see wrapCapture's doc for why
//     the SchedulerModel variant exists — the exact bug class welfare_guard.go
//     already documents and fixes the same way).
//   - ssd: read back the capture sidecar train.RunSSDCommand already writes
//     (<checkpoint-dir>/ssd-captures.jsonl — the filename cli/ssd.go's own
//     --checkpoint-dir help text already documents) once the run succeeds, and
//     land each row as a KindTrace Item.

// modelFingerprint derives the ModelFingerprint the dataset design's
// provenance rule requires ("a score always names what produced it — never
// re-score an old response against a later model state") from what cmd/lem
// already has in hand at load time: the resolved model directory path.
// Hashing the full weights on every load is not viable (checkpoints run
// multi-GB); the path is the identity this repo's own conventions already
// treat as stable — a meaningfully different model state gets a NEW
// directory (see quant.go's defaultOutDir) rather than overwriting one in
// place. Both taps use this same derivation so a row captured either way
// names the same fingerprint for the same checkpoint. The "path:" prefix
// names the fingerprint's own provenance honestly, the same way judge-tier
// score kinds are namespaced ("judge:<name>") — this is a path-derived
// identity, not a content hash.
func modelFingerprint(path string) string {
	trimmed := core.Trim(path)
	if trimmed == "" {
		return ""
	}
	if r := core.PathAbs(trimmed); r.OK {
		if abs, ok := r.Value.(string); ok && abs != "" {
			return core.Concat("path:", abs)
		}
	}
	return core.Concat("path:", trimmed)
}

// resolveCaptureDataset opens the shared dataset store and resolves slug to
// its Dataset — the preflight both capture taps need before they can tee
// anything. The store is returned OPEN; the caller closes it, including on a
// resolution failure (this function closes it itself in that one case, so a
// store opened then immediately failing to resolve never leaks the DuckDB
// handle into the caller's error path).
func resolveCaptureDataset(slug string) (tui.DatasetStore, dataset.Dataset, error) {
	opened := tui.OpenDatasetStore()
	if !opened.OK {
		return nil, dataset.Dataset{}, opened.Err()
	}
	store, ok := opened.Value.(tui.DatasetStore)
	if !ok {
		return nil, dataset.Dataset{}, core.NewError("dataset store: unexpected OpenDatasetStore result type")
	}
	dsResult := store.DatasetBySlug(slug)
	if !dsResult.OK {
		_ = store.Close()
		return nil, dataset.Dataset{}, core.E("main.resolveCaptureDataset",
			core.Sprintf("dataset %q not found — create it first with `lem data create %s`", slug, slug), dsResult.Err())
	}
	ds, ok := dsResult.Value.(dataset.Dataset)
	if !ok {
		_ = store.Close()
		return nil, dataset.Dataset{}, core.NewError("dataset store: unexpected DatasetBySlug result type")
	}
	return store, ds, nil
}

// ---- serve tap ----

// buildCaptureLoader resolves --capture into a wired serving.ModelLoader plus
// the dataset store it captures into, or returns (nil, nil, nil) when
// captureSlug is empty — "OFF without the flag (the approved privacy
// default)": no store is ever opened, nothing under ~/.lem is ever touched,
// when capture is not requested. The caller is responsible for closing the
// returned store (nil-safe) once serving stops.
func buildCaptureLoader(captureSlug string, log io.Writer) (serving.ModelLoader, tui.DatasetStore, error) {
	slug := core.Trim(captureSlug)
	if slug == "" {
		return nil, nil, nil
	}
	store, ds, err := resolveCaptureDataset(slug)
	if err != nil {
		return nil, nil, err
	}
	return captureModelLoader(store, ds.ID, log), store, nil
}

// captureModelLoader builds the serving.ModelLoader --capture installs:
// resolve through the registered backend exactly as
// go/serving's own unexported defaultTextModelLoader does (inference.LoadModel
// is that function's one exported seam), then wrap the result with the
// capture decorator. Mirrors defaultTextModelLoader's error handling so a
// captured load fails exactly the same way an uncaptured one would.
func captureModelLoader(store dataset.Store, datasetID string, log io.Writer) serving.ModelLoader {
	return func(path string, opts ...inference.LoadOption) (inference.TextModel, error) {
		result := inference.LoadModel(path, opts...)
		if !result.OK {
			if err, ok := result.Value.(error); ok {
				return nil, err
			}
			return nil, core.E("main.captureModelLoader", "registered backend failed to load model", nil)
		}
		model, ok := result.Value.(inference.TextModel)
		if !ok || model == nil {
			return nil, core.E("main.captureModelLoader", "registered backend returned non-TextModel value", nil)
		}
		return wrapCapture(model, store, datasetID, modelFingerprint(path), log), nil
	}
}

// captureTextModel decorates an inference.TextModel with a completed-turn tee
// into a dataset. Only Chat is overridden — the compat mux funnels every
// OpenAI/Anthropic/Ollama CHAT route through TextModel.Chat
// (go/serving/welfare_guard.go's welfareTextModel makes the identical scope
// call, in its own doc comment); the legacy text-completions route
// (model.Generate) is not captured, the same accepted scope boundary the
// welfare guard already carries.
type captureTextModel struct {
	inference.TextModel
	store       dataset.Store
	datasetID   string
	fingerprint string
	log         io.Writer
}

// wrapCapture decorates model with the capture tap. A model that routes
// through a request scheduler (inference.SchedulerModel — the -scheduler
// serve modes) is wrapped as a captureSchedulerModel instead: the compat mux
// prefers inference.SchedulerModel.Schedule over Chat
// (go/serving/compat/mux.go), and inference.As walks Unwrap to find it — so a
// plain captureTextModel would be unwrapped straight past and never see a
// scheduled request. This is the exact bug class
// go/serving/welfare_guard.go's own doc comment documents (and fixes the
// same way) for the welfare guard; capture would silently miss every
// scheduled turn without this.
func wrapCapture(model inference.TextModel, store dataset.Store, datasetID, fingerprint string, log io.Writer) inference.TextModel {
	c := &captureTextModel{TextModel: model, store: store, datasetID: datasetID, fingerprint: fingerprint, log: log}
	if sched, ok := inference.As[inference.SchedulerModel](model); ok {
		return &captureSchedulerModel{captureTextModel: c, inner: sched}
	}
	return c
}

// Chat tees the completed turn once the stream drains — the tokens reach the
// client first, capture is a side effect of draining, never a delay on the
// hot path.
func (m *captureTextModel) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	seq := m.TextModel.Chat(ctx, messages, opts...)
	prompt := reduceMessagesToPrompt(messages)
	if core.Trim(prompt) == "" {
		return seq
	}
	return func(yield func(inference.Token) bool) {
		var reply core.Builder
		for tok := range seq {
			reply.WriteString(tok.Text)
			if !yield(tok) {
				return
			}
		}
		m.captureCompleted(prompt, reply.String())
	}
}

// captureCompleted tees one completed turn into the dataset. Errors are
// logged and swallowed — never returned, never panicked: "capture failures
// log-and-continue — serving must never break because the dataset write
// failed" is a hard rule for a live server.
func (m *captureTextModel) captureCompleted(prompt, response string) {
	if core.Trim(response) == "" {
		return
	}
	r := dataset.IngestPair(m.store, prompt, response, dataset.IngestRequest{
		DatasetID: m.datasetID, Source: dataset.SourceCaptureServe, ModelFingerprint: m.fingerprint,
	})
	if !r.OK {
		core.Print(m.log, "serve: capture write failed (serving continues): %s", r.Error())
	}
}

// Unwrap exposes the wrapped model so the serving layer can still reach
// optional capabilities this tap does not itself re-declare (embeddings,
// rerank, vision, audio) — without it, a captured model would silently lose
// those routes purely because it got wrapped. Mirrors welfareTextModel.Unwrap
// exactly.
func (m *captureTextModel) Unwrap() inference.TextModel { return m.TextModel }

// captureSchedulerModel is captureTextModel extended with the
// inference.SchedulerModel surface — see wrapCapture's doc for why this type
// exists. Mirrors go/serving/welfare_guard.go's welfareSchedulerModel.
type captureSchedulerModel struct {
	*captureTextModel
	inner inference.SchedulerModel
}

var _ inference.SchedulerModel = (*captureSchedulerModel)(nil)

// Schedule tees the completed turn through the scheduled path — the
// ScheduledToken twin of Chat, folding the streamed reply and capturing once
// the inner stream drains. Mirrors welfareSchedulerModel.Schedule's forwarding
// shape.
func (m *captureSchedulerModel) Schedule(ctx context.Context, req inference.ScheduledRequest) (inference.RequestHandle, <-chan inference.ScheduledToken, error) {
	handle, stream, err := m.inner.Schedule(ctx, req)
	if err != nil {
		return handle, stream, err
	}
	prompt := reduceMessagesToPrompt(req.Messages)
	if core.Trim(prompt) == "" {
		return handle, stream, err
	}
	out := make(chan inference.ScheduledToken, cap(stream))
	go func() {
		defer close(out)
		var reply core.Builder
		for tok := range stream {
			reply.WriteString(tok.Token.Text)
			out <- tok
		}
		m.captureCompleted(prompt, reply.String())
	}()
	return handle, out, nil
}

// reduceMessagesToPrompt flattens a chat request's messages into the flat
// prompt string dataset.IngestPair's KindPair shape wants: every non-empty
// turn's content, newline-joined in order. Mirrors the same content-only (no
// role prefix) reduction go/dataset.MessagesContent.LastExchange applies when
// it turns a full conversation into a pair, so a captured item's prompt text
// has the same shape a dataset-side messages-to-pair reduction would
// produce.
func reduceMessagesToPrompt(messages []inference.Message) string {
	var sb core.Builder
	for _, message := range messages {
		if core.Trim(message.Content) == "" {
			continue
		}
		sb.WriteString(message.Content)
		sb.WriteString("\n")
	}
	return sb.String()
}

// ---- ssd tap ----

// ssdCaptureSidecarName is the capture sidecar filename train.RunSSDCommand
// writes under --checkpoint-dir whenever CheckpointDir is non-empty (its own
// internal default, applied unconditionally at the command layer — see
// go/train/command.go/ssd.go); cli/ssd.go's own --checkpoint-dir flag help
// text already documents this exact name, so this constant matches a
// published, stable convention rather than a private implementation detail.
const ssdCaptureSidecarName = "ssd-captures.jsonl"

// ingestSSDTrace reads the capture sidecar a completed `lem ssd
// --checkpoint-dir` run wrote and lands each row as a KindTrace Item in the
// named dataset. Errors are returned to the caller rather than logged here —
// cli/ssd.go decides how loudly to treat a tap failure (log-and-continue: an
// hours-long sampling run's checkpoint is already safe on disk regardless of
// whether this last step succeeds).
func ingestSSDTrace(checkpointDir, slug, modelPath string, log io.Writer) error {
	store, ds, err := resolveCaptureDataset(slug)
	if err != nil {
		return err
	}
	defer store.Close()

	sidecarPath := core.PathJoin(checkpointDir, ssdCaptureSidecarName)
	opened := core.Open(sidecarPath)
	if !opened.OK {
		return core.E("main.ingestSSDTrace", core.Concat("open capture sidecar ", sidecarPath), opened.Err())
	}
	file, ok := opened.Value.(*core.OSFile)
	if !ok {
		return core.E("main.ingestSSDTrace", "unexpected capture sidecar file result", nil)
	}
	defer file.Close()

	fingerprint := modelFingerprint(modelPath)
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64*1024), 4*1024*1024)
	ingested, deduped, skipped := 0, 0, 0
	row := 0
	for scanner.Scan() {
		row++
		line := core.Trim(scanner.Text())
		if line == "" {
			continue
		}
		_, wasDuplicate, itemErr := ingestTraceItem(store, ds.ID, []byte(line), dataset.SourceSSD, core.Sprintf("%d", row), fingerprint)
		if itemErr != nil {
			skipped++
			continue
		}
		if wasDuplicate {
			deduped++
		} else {
			ingested++
		}
	}
	if scanErr := scanner.Err(); scanErr != nil {
		return core.E("main.ingestSSDTrace", "scan capture sidecar", scanErr)
	}
	core.Print(log, "ssd: dataset %q — %d trace items ingested, %d deduped, %d skipped", slug, ingested, deduped, skipped)
	return nil
}

// ingestTraceItem lands one KindTrace item, deduping by content hash within
// the dataset — the one dedupe-then-append shape go/dataset's own unexported
// ingestContent primitive applies to every OTHER kind, reimplemented here
// only for KindTrace, which go/dataset exports no dedicated ingest helper
// for (the design's ingest paths are pair/messages/capture-row/chathistory;
// SSD traces land through the Store contract directly). A KindTrace item is
// never welfare-screened either way — go/dataset/screen.go's screenItem has
// no KindTrace case (an opaque trace carries no user-authored text) — so
// skipping that step here mirrors go/dataset's own behaviour exactly, not a
// shortcut around it.
func ingestTraceItem(store dataset.Store, datasetID string, content []byte, source dataset.ItemSource, sourceRef, fingerprint string) (dataset.Item, bool, error) {
	hashResult := dataset.ContentHash(dataset.KindTrace, content)
	if !hashResult.OK {
		return dataset.Item{}, false, hashResult.Err()
	}
	hash, ok := hashResult.Value.(string)
	if !ok {
		return dataset.Item{}, false, core.NewError("dataset: unexpected ContentHash result type")
	}

	existingResult := store.Items(dataset.ItemFilter{DatasetID: datasetID, ContentHash: hash, IncludeArchived: true})
	if !existingResult.OK {
		return dataset.Item{}, false, existingResult.Err()
	}
	if existing, ok := existingResult.Value.([]dataset.Item); ok && len(existing) > 0 {
		return existing[0], true, nil
	}

	item := dataset.Item{
		ID: dataset.NewID(), DatasetID: datasetID, Kind: dataset.KindTrace, Content: content,
		Source: source, SourceRef: sourceRef, ModelFingerprint: fingerprint, ContentHash: hash, CreatedAt: time.Now(),
	}
	appendResult := store.ItemAppend(item)
	if !appendResult.OK {
		return dataset.Item{}, false, appendResult.Err()
	}
	appended, ok := appendResult.Value.(dataset.Item)
	if !ok {
		return dataset.Item{}, false, core.NewError("dataset: unexpected ItemAppend result type")
	}
	return appended, false, nil
}
