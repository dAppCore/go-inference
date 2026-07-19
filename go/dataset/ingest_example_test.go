// SPDX-Licence-Identifier: EUPL-1.2

package dataset_test

import (
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/dataset"
)

// ExampleIngestPair ingests one prompt/response pair — the primitive a
// `lem serve --capture` tap uses per completed turn.
func ExampleIngestPair() {
	store := dataset.NewMemoryStore()
	ds := dataset.Dataset{ID: "ds-1", Slug: "evening-vents", Title: "Evening vents", CreatedAt: time.Unix(1000, 0)}
	core.MustCast[dataset.Dataset](store.DatasetCreate(ds))

	r := dataset.IngestPair(store, "hi", "hello", dataset.IngestRequest{
		DatasetID: ds.ID, Source: dataset.SourceCaptureServe, ModelFingerprint: "fp-1",
	})
	outcome := r.Value.(dataset.IngestOutcome)
	core.Println(r.OK, outcome.Deduped, outcome.Item.ModelFingerprint)
	// Output:
	// true false fp-1
}

// ExampleIngestJSONL ingests a mixed-shape JSONL corpus in one call,
// auto-detecting messages/pair/CaptureRow rows per line.
func ExampleIngestJSONL() {
	store := dataset.NewMemoryStore()
	ds := dataset.Dataset{ID: "ds-1", Slug: "evening-vents", Title: "Evening vents", CreatedAt: time.Unix(1000, 0)}
	core.MustCast[dataset.Dataset](store.DatasetCreate(ds))

	body := `{"prompt":"hi","response":"hello"}` + "\n" +
		`{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]}` + "\n"
	r := dataset.IngestJSONL(store, ds.ID, core.NewReader(body), dataset.IngestOptions{})
	report := r.Value.(dataset.IngestReport)
	core.Println(report.Ingested, len(report.Skipped))
	// Output:
	// 2 0
}

// ExampleIngestChatSessions normalises one chathistory session into a
// single KindMessages Item, ordered by Turn.Ordinal.
func ExampleIngestChatSessions() {
	store := dataset.NewMemoryStore()
	ds := dataset.Dataset{ID: "ds-1", Slug: "evening-vents", Title: "Evening vents", CreatedAt: time.Unix(1000, 0)}
	core.MustCast[dataset.Dataset](store.DatasetCreate(ds))

	sessions := []dataset.ChatSession{{
		ID: "sess-1", ModelID: "lemer-lite", StartedAt: time.Unix(2000, 0),
		Turns: []dataset.ChatTurn{
			{Role: "user", Content: "hey lemma", Ordinal: 1},
			{Role: "assistant", Content: "hi there", Ordinal: 2},
		},
	}}
	r := dataset.IngestChatSessions(store, ds.ID, sessions)
	report := r.Value.(dataset.IngestReport)
	core.Println(report.Ingested)
	// Output:
	// 1
}
