// SPDX-Licence-Identifier: EUPL-1.2

package dataset

import (
	core "dappco.re/go"
)

func TestIngestReport_tally(t *core.T) {
	report := IngestReport{}
	report.tally(IngestOutcome{})
	report.tally(IngestOutcome{Deduped: true})
	report.tally(IngestOutcome{Quarantined: true})
	core.AssertEqual(t, 2, report.Ingested, "a plain ingest and a quarantined ingest both land")
	core.AssertEqual(t, 1, report.Deduped)
	core.AssertEqual(t, 1, report.Quarantined)
}

func TestJsonlRow_normalise_Good(t *core.T) {
	messagesRow := jsonlRow{Messages: []MessageTurn{{Role: "user", Content: "hi"}, {Role: "assistant", Content: "hello"}}}
	kind, content, createdAt, ok := messagesRow.normalise()
	core.AssertTrue(t, ok)
	core.AssertEqual(t, KindMessages, kind)
	core.AssertTrue(t, createdAt.IsZero(), "messages rows carry no birth time")
	core.AssertContains(t, string(content), "hello")

	pairRow := jsonlRow{Prompt: "hi", Response: "hello"}
	kind, content, createdAt, ok = pairRow.normalise()
	core.AssertTrue(t, ok)
	core.AssertEqual(t, KindPair, kind)
	core.AssertTrue(t, createdAt.IsZero())
	core.AssertContains(t, string(content), "hello")

	captureRow := jsonlRow{Step: 3, Prompt: "hi", Text: "hello", At: 1700000000}
	kind, content, createdAt, ok = captureRow.normalise()
	core.AssertTrue(t, ok)
	core.AssertEqual(t, KindPair, kind)
	core.AssertEqual(t, int64(1700000000), createdAt.Unix(), "a CaptureRow's at_unix becomes the item's birth time")
	core.AssertContains(t, string(content), "hello")
}

func TestJsonlRow_normalise_Bad(t *core.T) {
	_, _, _, ok := jsonlRow{}.normalise()
	core.AssertFalse(t, ok, "an empty row matches no shape")

	_, _, _, ok = jsonlRow{Prompt: "hi"}.normalise()
	core.AssertFalse(t, ok, "a prompt with no response/text matches no shape")
}

func TestJsonlRow_normalise_Ugly(t *core.T) {
	// response takes priority over text when both are present.
	row := jsonlRow{Prompt: "hi", Response: "explicit", Text: "fallback"}
	_, content, _, ok := row.normalise()
	core.AssertTrue(t, ok)
	core.AssertContains(t, string(content), "explicit")
	core.AssertNotContains(t, string(content), "fallback")
}

func TestIngestPair_Good(t *core.T) {
	store, ds, _ := seededStore(t)
	r := IngestPair(store, "explain your reasoning", "the sky is blue", IngestRequest{DatasetID: ds.ID, Source: SourceCaptureServe, ModelFingerprint: "fp-1"})
	core.AssertTrue(t, r.OK)
	outcome := r.Value.(IngestOutcome)
	core.AssertFalse(t, outcome.Deduped)
	core.AssertEqual(t, "fp-1", outcome.Item.ModelFingerprint)
	core.AssertEqual(t, SourceCaptureServe, outcome.Item.Source)
}

func TestIngestPair_Bad(t *core.T) {
	store, _, _ := seededStore(t)
	r := IngestPair(store, "hi", "hello", IngestRequest{DatasetID: "missing-dataset", Source: SourceCaptureServe})
	core.AssertFalse(t, r.OK, "an unknown dataset id must fail")

	r = IngestPair(nil, "hi", "hello", IngestRequest{DatasetID: "ds-1", Source: SourceCaptureServe})
	core.AssertFalse(t, r.OK, "a nil store must fail, not panic")

	r = IngestPair(store, "hi", "hello", IngestRequest{DatasetID: "", Source: SourceCaptureServe})
	core.AssertFalse(t, r.OK, "an empty dataset id must fail")
}

func TestIngestPair_Ugly(t *core.T) {
	store, ds, _ := seededStore(t)

	r := IngestPair(store, "", "", IngestRequest{DatasetID: ds.ID, Source: SourceCaptureServe})
	core.AssertFalse(t, r.OK, "an empty prompt/response must fail content validation")

	r = IngestPair(store, "hi", "hello", IngestRequest{DatasetID: ds.ID, Source: ItemSource("bogus")})
	core.AssertFalse(t, r.OK, "an unknown source must fail")

	// Dedupe: ingesting the same pair twice is a counted no-op, not an
	// error, and returns the pre-existing item.
	first := IngestPair(store, "dup prompt", "dup response", IngestRequest{DatasetID: ds.ID, Source: SourceCaptureServe})
	core.RequireTrue(t, first.OK)
	second := IngestPair(store, "dup prompt", "dup response", IngestRequest{DatasetID: ds.ID, Source: SourceCaptureServe})
	core.AssertTrue(t, second.OK)
	core.AssertTrue(t, second.Value.(IngestOutcome).Deduped)
	core.AssertEqual(t, first.Value.(IngestOutcome).Item.ID, second.Value.(IngestOutcome).Item.ID)
}

func TestIngestJSONL_Good(t *core.T) {
	store, ds, _ := seededStore(t)
	body := "" +
		`{"prompt":"hi","response":"hello"}` + "\n" +
		`{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]}` + "\n" +
		`{"step":1,"prompt":"hi again","text":"hello again","at_unix":1700000000}` + "\n"
	r := IngestJSONL(store, ds.ID, core.NewReader(body), IngestOptions{})
	core.AssertTrue(t, r.OK)
	report := r.Value.(IngestReport)
	core.AssertEqual(t, 3, report.Ingested)
	core.AssertEqual(t, 0, len(report.Skipped))
}

func TestIngestJSONL_Bad(t *core.T) {
	store, ds, _ := seededStore(t)
	body := "" +
		`not json at all` + "\n" +
		`{"prompt":"only a prompt"}` + "\n" +
		`` + "\n" + // blank line, silently skipped, not counted
		`{"prompt":"hi","response":"hello"}` + "\n"
	r := IngestJSONL(store, ds.ID, core.NewReader(body), IngestOptions{})
	core.AssertTrue(t, r.OK, "row-level failures do not fail the whole call")
	report := r.Value.(IngestReport)
	core.AssertEqual(t, 1, report.Ingested)
	core.AssertEqual(t, 2, len(report.Skipped))
	core.AssertEqual(t, 1, report.Skipped[0].Row)
	core.AssertEqual(t, 2, report.Skipped[1].Row)
}

func TestIngestJSONL_Ugly(t *core.T) {
	r := IngestJSONL(nil, "ds-1", core.NewReader(""), IngestOptions{})
	core.AssertFalse(t, r.OK, "a nil store must fail")

	store, _, _ := seededStore(t)
	r = IngestJSONL(store, "", core.NewReader(""), IngestOptions{})
	core.AssertFalse(t, r.OK, "an empty dataset id must fail")

	r = IngestJSONL(store, "ds-1", nil, IngestOptions{})
	core.AssertFalse(t, r.OK, "a nil reader must fail")

	// Malformed content that IS valid top-level JSON but fails the pair
	// shape check downstream (welfare/dedupe/append) is still a counted
	// skip, not a call-aborting error.
	store2, ds2, _ := seededStore(t)
	body := `{"prompt":"hi","response":"hello"}` + "\n" + `{"prompt":"hi","response":"hello"}` + "\n"
	r = IngestJSONL(store2, ds2.ID, core.NewReader(body), IngestOptions{})
	core.AssertTrue(t, r.OK)
	report := r.Value.(IngestReport)
	core.AssertEqual(t, 1, report.Ingested)
	core.AssertEqual(t, 1, report.Deduped, "the second identical row dedupes against the first")
}

// failingReader emits data then a non-EOF error — exercises the
// bufio.Scanner error path IngestJSONL surfaces via scanner.Err().
type failingReader struct {
	data []byte
	pos  int
	err  error
}

func (r *failingReader) Read(p []byte) (int, error) {
	if r.pos >= len(r.data) {
		return 0, r.err
	}
	n := copy(p, r.data[r.pos:])
	r.pos += n
	return n, nil
}

func TestIngestJSONL_ScanError(t *core.T) {
	store, ds, _ := seededStore(t)
	r := IngestJSONL(store, ds.ID, &failingReader{data: []byte(`{"prompt":"hi","response":"hello"}`), err: core.NewError("disk hiccup")}, IngestOptions{})
	core.AssertFalse(t, r.OK, "a genuine read error must fail the call, unlike a per-row parse error")
}

func TestIngestChatSessions_Good(t *core.T) {
	store, ds, _ := seededStore(t)
	sessions := []ChatSession{
		{
			ID: "sess-1", Title: "evening vent", ModelID: "lemer-lite", StartedAt: timeAt(9000),
			Turns: []ChatTurn{
				{Role: "assistant", Content: "hi there", Ordinal: 2},
				{Role: "user", Content: "hey lemma", Ordinal: 1},
			},
		},
	}
	r := IngestChatSessions(store, ds.ID, sessions)
	core.AssertTrue(t, r.OK)
	report := r.Value.(IngestReport)
	core.AssertEqual(t, 1, report.Ingested)

	items := core.MustCast[[]Item](store.Items(ItemFilter{DatasetID: ds.ID, Kind: KindMessages}))
	core.AssertLen(t, items, 1)
	core.AssertEqual(t, "sess-1", items[0].SourceRef)
	core.AssertEqual(t, "lemer-lite", items[0].ModelFingerprint)

	var mc MessagesContent
	core.RequireTrue(t, core.JSONUnmarshal(items[0].Content, &mc).OK)
	core.AssertEqual(t, "user", mc.Messages[0].Role, "turns are re-ordered by Ordinal, out-of-order input notwithstanding")
}

func TestIngestChatSessions_Bad(t *core.T) {
	store, _, _ := seededStore(t)
	r := IngestChatSessions(store, "", nil)
	core.AssertFalse(t, r.OK, "an empty dataset id must fail")

	r = IngestChatSessions(nil, "ds-1", nil)
	core.AssertFalse(t, r.OK, "a nil store must fail")
}

func TestIngestChatSessions_Ugly(t *core.T) {
	store, ds, _ := seededStore(t)
	sessions := []ChatSession{
		{ID: "empty-sess", Turns: nil},
	}
	r := IngestChatSessions(store, ds.ID, sessions)
	core.AssertTrue(t, r.OK)
	report := r.Value.(IngestReport)
	core.AssertEqual(t, 0, report.Ingested)
	core.AssertEqual(t, 1, len(report.Skipped))
	core.AssertEqual(t, "session has no turns", report.Skipped[0].Reason)

	// A session whose turns fail the role-alternating shape rule (two
	// adjacent user turns, no assistant reply between them) is a
	// counted skip too, surfaced from ingestContent's validation step.
	malformed := []ChatSession{{
		ID: "malformed-sess", ModelID: "lemer-lite", StartedAt: timeAt(9500),
		Turns: []ChatTurn{
			{Role: "user", Content: "one", Ordinal: 1},
			{Role: "user", Content: "two", Ordinal: 2},
		},
	}}
	r = IngestChatSessions(store, ds.ID, malformed)
	core.AssertTrue(t, r.OK)
	report = r.Value.(IngestReport)
	core.AssertEqual(t, 0, report.Ingested)
	core.AssertEqual(t, 1, len(report.Skipped))
}
