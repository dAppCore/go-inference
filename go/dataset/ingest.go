// SPDX-Licence-Identifier: EUPL-1.2

package dataset

import (
	"bufio"
	"sort"
	"time"

	core "dappco.re/go"
)

// IngestOptions carries the ingest-wide settings that do not vary
// per-row — currently just the fallback ModelFingerprint for rows that
// don't carry their own provenance.
type IngestOptions struct {
	// ModelFingerprint is stamped on every ingested Item when the
	// source rows carry no provenance of their own (JSONL import is
	// "empty for human/imported text" by default — set this only when
	// the caller genuinely knows the generating model).
	ModelFingerprint string
}

// IngestRequest is one item's ingest inputs — shared by every ingest
// path in this file, and by [IngestPair] for single-item callers like a
// `lem serve --capture` tap.
type IngestRequest struct {
	DatasetID        string
	Source           ItemSource
	SourceRef        string
	ModelFingerprint string
	// CreatedAt is the item's provenance timestamp — the zero Time
	// means "now" (ingestContent fills it in). Set this explicitly when
	// the source data carries its own birth time (a CaptureRow's
	// at_unix, a chathistory session's started_at) so exports order by
	// true generation time, not import wall-clock time.
	CreatedAt time.Time
}

// IngestOutcome is what happened to one ingested row.
type IngestOutcome struct {
	Item Item
	// Deduped is true when an item with the same content hash already
	// existed in the dataset — a counted no-op, not an error; Item is
	// the PRE-EXISTING item in this case.
	Deduped bool
	// Quarantined is true when the welfare screen flagged the item at
	// the door — the item still landed (Item is valid and stored), it
	// is simply also reviewable as quarantined.
	Quarantined bool
}

// IngestSkip records one row this package could not ingest — never a
// silent drop.
type IngestSkip struct {
	// Row is the 1-based position of the row within its source (JSONL
	// line number, or session index for chathistory ingest).
	Row    int
	Reason string
}

// IngestReport is the honest per-call tally every ingest path in this
// file returns.
type IngestReport struct {
	Ingested    int
	Deduped     int
	Quarantined int
	Skipped     []IngestSkip
}

// tally folds one row's outcome into the report. Quarantined items still
// count as Ingested — they landed, they are just also flagged.
func (report *IngestReport) tally(outcome IngestOutcome) {
	switch {
	case outcome.Deduped:
		report.Deduped++
	case outcome.Quarantined:
		report.Quarantined++
		report.Ingested++
	default:
		report.Ingested++
	}
}

// ingestContent is the shared primitive every ingest path in this
// package funnels through: validate the content shape, compute its
// canonicalised hash, dedupe within the dataset, run the welfare screen,
// and append — "every ingest path runs the welfare screen at the door:
// a hit sets status quarantined ... and the item still lands (visible,
// reviewable)" per the design. screen is shared across a whole bulk
// ingest call (see newWelfareScreen) rather than rebuilt per row.
func ingestContent(store Store, kind ItemKind, content []byte, req IngestRequest, screen *welfareScreen) (IngestOutcome, error) {
	if r := ValidateItemContent(kind, content); !r.OK {
		return IngestOutcome{}, r.Err()
	}
	hashResult := ContentHash(kind, content)
	if !hashResult.OK {
		return IngestOutcome{}, hashResult.Err()
	}
	hash := hashResult.Value.(string)

	existingResult := store.Items(ItemFilter{DatasetID: req.DatasetID, ContentHash: hash, IncludeArchived: true})
	if !existingResult.OK {
		return IngestOutcome{}, existingResult.Err()
	}
	if existing, ok := existingResult.Value.([]Item); ok && len(existing) > 0 {
		return IngestOutcome{Item: existing[0], Deduped: true}, nil
	}

	createdAt := req.CreatedAt
	if createdAt.IsZero() {
		createdAt = time.Now()
	}
	triggered, det := screen.screenItem(kind, content)

	item := Item{
		ID:               NewID(),
		DatasetID:        req.DatasetID,
		Kind:             kind,
		Content:          content,
		Source:           req.Source,
		SourceRef:        req.SourceRef,
		ModelFingerprint: req.ModelFingerprint,
		ContentHash:      hash,
		CreatedAt:        createdAt,
	}
	appendResult := store.ItemAppend(item)
	if !appendResult.OK {
		return IngestOutcome{}, appendResult.Err()
	}
	stored := appendResult.Value.(Item)

	if triggered {
		review := Review{ItemID: stored.ID, Status: StatusQuarantined, Reviewer: ReviewerAutoWelfare, Note: describeWelfareHit(det), CreatedAt: createdAt}
		if r := store.ReviewAppend(review); !r.OK {
			return IngestOutcome{}, r.Err()
		}
	}
	return IngestOutcome{Item: stored, Quarantined: triggered}, nil
}

// IngestPair ingests one prompt/response pair as a KindPair Item —
// dedupe, welfare-screen, validate, append. The primitive a `lem serve
// --capture` tap (Task 7) calls per completed turn.
//
//	r := dataset.IngestPair(store, "hi", "hello", dataset.IngestRequest{
//	    DatasetID: ds.ID, Source: dataset.SourceCaptureServe, ModelFingerprint: fp,
//	})
func IngestPair(store Store, prompt, response string, req IngestRequest) core.Result {
	if store == nil {
		return core.Fail(core.NewError("dataset: ingest requires a store"))
	}
	if core.Trim(req.DatasetID) == "" {
		return core.Fail(core.NewError("dataset: ingest requires a dataset id"))
	}
	if !knownItemSource(req.Source) {
		return core.Fail(core.NewError("dataset: ingest requires a known source"))
	}
	content := core.JSONMarshal(PairContent{Prompt: prompt, Response: response})
	if !content.OK {
		return core.Fail(core.E("dataset.IngestPair", "marshal pair content", content.Err()))
	}
	outcome, err := ingestContent(store, KindPair, content.Value.([]byte), req, newWelfareScreen())
	if err != nil {
		return core.Fail(core.E("dataset.IngestPair", "ingest pair", err))
	}
	return core.Ok(outcome)
}

// jsonlRow is the union of every row shape IngestJSONL auto-detects:
// OpenAI-style messages, a bare prompt/response pair, or the
// train.CaptureRow sidecar shape (step/prompt/text/at_unix) — matched by
// field name only, this package never imports dappco.re/go/inference/train
// (see the package doc).
type jsonlRow struct {
	Prompt   string        `json:"prompt"`
	Response string        `json:"response"`
	Text     string        `json:"text"`
	Step     int           `json:"step"`
	At       int64         `json:"at_unix"`
	Messages []MessageTurn `json:"messages"`
}

// normalise reduces a decoded row to (kind, content, createdAt, ok).
// createdAt is the zero Time unless the row carried its own birth time
// (CaptureRow's at_unix). ok is false when the row matches none of the
// recognised shapes.
func (row jsonlRow) normalise() (ItemKind, []byte, time.Time, bool) {
	if len(row.Messages) > 0 {
		encoded := core.JSONMarshal(MessagesContent{Messages: row.Messages})
		if !encoded.OK {
			return "", nil, time.Time{}, false
		}
		return KindMessages, encoded.Value.([]byte), time.Time{}, true
	}
	response := core.Trim(row.Response)
	if response == "" {
		response = core.Trim(row.Text)
	}
	if core.Trim(row.Prompt) == "" || response == "" {
		return "", nil, time.Time{}, false
	}
	encoded := core.JSONMarshal(PairContent{Prompt: row.Prompt, Response: response})
	if !encoded.OK {
		return "", nil, time.Time{}, false
	}
	createdAt := time.Time{}
	if row.At > 0 {
		createdAt = time.Unix(row.At, 0).UTC()
	}
	return KindPair, encoded.Value.([]byte), createdAt, true
}

// IngestJSONL reads newline-delimited JSON rows from reader and ingests
// each as a KindMessages or KindPair Item, auto-detecting the row shape
// per line — {"messages":[...]}, {"prompt":...,"response":...}, or the
// CaptureRow sidecar shape {"step":...,"prompt":...,"text":...,
// "at_unix":...} — so historic training captures load unchanged. A
// CaptureRow row's at_unix, when present, becomes the ingested Item's
// CreatedAt: provenance is the original generation time, not the import
// time.
//
// Every row is independent — a malformed or unrecognised row is a
// counted skip (see IngestReport.Skipped), never a call-aborting error.
//
//	r := dataset.IngestJSONL(store, ds.ID, reader, dataset.IngestOptions{})
//	report := r.Value.(dataset.IngestReport)
func IngestJSONL(store Store, datasetID string, reader core.Reader, opts IngestOptions) core.Result {
	if store == nil {
		return core.Fail(core.NewError("dataset: ingest requires a store"))
	}
	if core.Trim(datasetID) == "" {
		return core.Fail(core.NewError("dataset: ingest requires a dataset id"))
	}
	if reader == nil {
		return core.Fail(core.NewError("dataset: ingest requires a reader"))
	}

	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 64*1024), 4*1024*1024)
	screen := newWelfareScreen()

	report := IngestReport{}
	rowNum := 0
	for scanner.Scan() {
		rowNum++
		raw := core.Trim(scanner.Text())
		if raw == "" {
			continue
		}

		var row jsonlRow
		if r := core.JSONUnmarshalString(raw, &row); !r.OK {
			report.Skipped = append(report.Skipped, IngestSkip{Row: rowNum, Reason: "invalid JSON"})
			continue
		}

		kind, content, createdAt, ok := row.normalise()
		if !ok {
			report.Skipped = append(report.Skipped, IngestSkip{Row: rowNum, Reason: "no recognised messages/prompt-response/capture-row shape"})
			continue
		}

		req := IngestRequest{
			DatasetID:        datasetID,
			Source:           SourceImportJSONL,
			ModelFingerprint: opts.ModelFingerprint,
			CreatedAt:        createdAt,
		}
		outcome, err := ingestContent(store, kind, content, req, screen)
		if err != nil {
			report.Skipped = append(report.Skipped, IngestSkip{Row: rowNum, Reason: err.Error()})
			continue
		}
		report.tally(outcome)
	}
	if err := scanner.Err(); err != nil {
		return core.Fail(core.E("dataset.IngestJSONL", "scan failed", err))
	}
	return core.Ok(report)
}

// ChatTurn is one neutral turn inside a [ChatSession] — mirrors
// dappco.re/go/inference/serving/chathistory.Turn's shape without
// importing that package's DuckDB reader (see the package doc); the CLI
// reads chats.duckdb and feeds already-loaded rows here.
type ChatTurn struct {
	Role    string
	Content string
	Ordinal int
}

// ChatSession is one neutral chathistory conversation — mirrors
// dappco.re/go/inference/serving/chathistory's session + turns shape.
type ChatSession struct {
	ID        string
	Title     string
	ModelID   string
	StartedAt time.Time
	Turns     []ChatTurn
}

// IngestChatSessions normalises chathistory sessions into one
// KindMessages Item per session (ordered by Turn.Ordinal), deduped and
// welfare-screened at the door like every other ingest path. ModelID is
// recorded as ModelFingerprint: chathistory only carries the wire model
// name, not a precise weights fingerprint, but it is the best available
// provenance.
//
//	r := dataset.IngestChatSessions(store, ds.ID, sessions)
func IngestChatSessions(store Store, datasetID string, sessions []ChatSession) core.Result {
	if store == nil {
		return core.Fail(core.NewError("dataset: ingest requires a store"))
	}
	if core.Trim(datasetID) == "" {
		return core.Fail(core.NewError("dataset: ingest requires a dataset id"))
	}

	report := IngestReport{}
	screen := newWelfareScreen()
	for i, session := range sessions {
		row := i + 1
		if len(session.Turns) == 0 {
			report.Skipped = append(report.Skipped, IngestSkip{Row: row, Reason: "session has no turns"})
			continue
		}
		turns := make([]ChatTurn, len(session.Turns))
		copy(turns, session.Turns)
		sort.SliceStable(turns, func(a, b int) bool { return turns[a].Ordinal < turns[b].Ordinal })

		messageTurns := make([]MessageTurn, 0, len(turns))
		for _, turn := range turns {
			messageTurns = append(messageTurns, MessageTurn{Role: turn.Role, Content: turn.Content})
		}
		content := core.JSONMarshal(MessagesContent{Messages: messageTurns})
		if !content.OK {
			report.Skipped = append(report.Skipped, IngestSkip{Row: row, Reason: "marshal messages content"})
			continue
		}

		req := IngestRequest{
			DatasetID:        datasetID,
			Source:           SourceImportChathistory,
			SourceRef:        session.ID,
			ModelFingerprint: session.ModelID,
			CreatedAt:        session.StartedAt,
		}
		outcome, err := ingestContent(store, KindMessages, content.Value.([]byte), req, screen)
		if err != nil {
			report.Skipped = append(report.Skipped, IngestSkip{Row: row, Reason: err.Error()})
			continue
		}
		report.tally(outcome)
	}
	return core.Ok(report)
}
