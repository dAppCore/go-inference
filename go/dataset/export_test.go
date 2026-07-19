// SPDX-Licence-Identifier: EUPL-1.2

package dataset

import (
	goio "io"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

// failingItemsStore wraps a real MemoryStore but forces Items to fail —
// exercises ExportDataset's items-fetch-failure branch, which a real
// Store implementation can hit (a query error) even though MemoryStore's
// own Items never fails once DatasetID is set.
type failingItemsStore struct {
	*MemoryStore
}

func (s *failingItemsStore) Items(ItemFilter) core.Result {
	return core.Fail(core.NewError("simulated items query failure"))
}

// failingWriteMedium wraps a real MemoryMedium but forces WriteStream to
// fail — exercises exportRows' open-failure branch, which a real medium
// legitimately hits on disk-full/permission-denied in production but
// which coreio.MemoryMedium's Create never fails (it has no failure
// mode of its own to simulate against).
type failingWriteMedium struct {
	*coreio.MemoryMedium
}

func (m *failingWriteMedium) WriteStream(string) (goio.WriteCloser, error) {
	return nil, core.NewError("simulated write failure")
}

// failingManifestWriteMedium wraps a real MemoryMedium, lets the JSONL
// WriteStream succeed, but forces the manifest's plain Write to fail —
// exercises ExportDataset's manifest-write-failure branch specifically
// (distinct from the JSONL open failure failingWriteMedium simulates).
type failingManifestWriteMedium struct {
	*coreio.MemoryMedium
}

func (m *failingManifestWriteMedium) Write(string, string) error {
	return core.NewError("simulated manifest write failure")
}

// ---- row builders ----

func TestItemAsPair_Good(t *core.T) {
	pc, ok := itemAsPair(fixtureItem("item-1", "ds-1", 1000))
	core.AssertTrue(t, ok)
	core.AssertEqual(t, "hi", pc.Prompt)

	messagesItem := Item{Kind: KindMessages, Content: core.JSONMarshal(MessagesContent{Messages: []MessageTurn{
		{Role: "user", Content: "hi"}, {Role: "assistant", Content: "hello"},
	}}).Value.([]byte)}
	pc, ok = itemAsPair(messagesItem)
	core.AssertTrue(t, ok)
	core.AssertEqual(t, "hello", pc.Response)
}

func TestItemAsPair_Bad(t *core.T) {
	_, ok := itemAsPair(Item{Kind: KindTrace, Content: []byte(`{}`)})
	core.AssertFalse(t, ok, "an opaque trace is never representable as a pair")
}

func TestItemAsPair_Ugly(t *core.T) {
	_, ok := itemAsPair(Item{Kind: KindPair, Content: []byte("not json")})
	core.AssertFalse(t, ok)

	noAssistant := core.JSONMarshal(MessagesContent{Messages: []MessageTurn{{Role: "user", Content: "hi"}}}).Value.([]byte)
	_, ok = itemAsPair(Item{Kind: KindMessages, Content: noAssistant})
	core.AssertFalse(t, ok, "a conversation with no assistant turn has no response")
}

func TestWriteSFTRow_Good(t *core.T) {
	row, ok := writeSFTRow(fixtureItem("item-1", "ds-1", 1000), 0)
	core.AssertTrue(t, ok)
	core.AssertEqual(t, `{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello-item-1"}]}`, string(row))

	messagesItem := Item{Kind: KindMessages, Content: core.JSONMarshal(MessagesContent{Messages: []MessageTurn{
		{Role: "user", Content: "hi"}, {Role: "assistant", Content: "hello"},
	}}).Value.([]byte)}
	row, ok = writeSFTRow(messagesItem, 0)
	core.AssertTrue(t, ok)
	core.AssertEqual(t, `{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]}`, string(row))
}

func TestWriteSFTRow_Bad(t *core.T) {
	_, ok := writeSFTRow(Item{Kind: KindTrace, Content: []byte(`{}`)}, 0)
	core.AssertFalse(t, ok)
}

func TestWriteSFTRow_Ugly(t *core.T) {
	_, ok := writeSFTRow(Item{Kind: KindPair, Content: []byte("not json")}, 0)
	core.AssertFalse(t, ok)
	_, ok = writeSFTRow(Item{Kind: KindMessages, Content: []byte("not json")}, 0)
	core.AssertFalse(t, ok)
}

func TestWritePairsRow_Good(t *core.T) {
	row, ok := writePairsRow(fixtureItem("item-1", "ds-1", 1000), 0)
	core.AssertTrue(t, ok)
	core.AssertEqual(t, `{"prompt":"hi","response":"hello-item-1"}`, string(row))
}

func TestWritePairsRow_Bad(t *core.T) {
	_, ok := writePairsRow(Item{Kind: KindTrace, Content: []byte(`{}`)}, 0)
	core.AssertFalse(t, ok)
}

func TestWritePairsRow_Ugly(t *core.T) {
	_, ok := writePairsRow(Item{Kind: KindPair, Content: []byte("not json")}, 0)
	core.AssertFalse(t, ok)
}

func TestWriteCaptureRow_Good(t *core.T) {
	item := fixtureItem("item-1", "ds-1", 1700000000)
	row, ok := writeCaptureRow(item, 3)
	core.AssertTrue(t, ok)
	core.AssertEqual(t, `{"step":3,"prompt":"hi","text":"hello-item-1","at_unix":1700000000}`, string(row))
}

func TestWriteCaptureRow_Bad(t *core.T) {
	_, ok := writeCaptureRow(Item{Kind: KindTrace, Content: []byte(`{}`)}, 0)
	core.AssertFalse(t, ok)
}

func TestWriteCaptureRow_Ugly(t *core.T) {
	_, ok := writeCaptureRow(Item{Kind: KindPair, Content: []byte("not json")}, 0)
	core.AssertFalse(t, ok)
}

func TestWriterFor_Good(t *core.T) {
	core.AssertNotNil(t, writerFor(FormatSFTJSONL))
	core.AssertNotNil(t, writerFor(FormatPairsJSONL))
	core.AssertNotNil(t, writerFor(FormatCaptureJSONL))
}

func TestWriterFor_Bad(t *core.T) {
	core.AssertNil(t, writerFor(ExportFormat("bogus")))
}

func TestDescribeFilter_Good(t *core.T) {
	core.AssertEqual(t, "status=approved", describeFilter(ItemFilter{Status: StatusApproved}))
	core.AssertEqual(t, "status=approved,kind=pair", describeFilter(ItemFilter{Status: StatusApproved, Kind: KindPair}))
}

func TestDescribeFilter_Bad(t *core.T) {
	core.AssertEqual(t, "none", describeFilter(ItemFilter{}))
}

func TestDescribeFilter_Ugly(t *core.T) {
	filter := ItemFilter{Source: SourceImportJSONL, Score: &ScoreExpression{Kind: ScoreKindLEK, Op: OpGTE, Threshold: 80}, IncludeArchived: true}
	core.AssertEqual(t, "source=import:jsonl,lek>=80,include_archived=true", describeFilter(filter))
}

// ---- exportRows ----

func TestExportRows_Good(t *core.T) {
	medium := coreio.NewMemoryMedium()
	items := []Item{fixtureItem("item-1", "ds-1", 1000), fixtureItem("item-2", "ds-1", 2000)}
	written, skipped, hashes, err := exportRows(medium, "out.jsonl", items, writePairsRow)
	core.AssertNoError(t, err)
	core.AssertEqual(t, 2, written)
	core.AssertEqual(t, 0, skipped)
	core.AssertLen(t, hashes, 2)

	content, readErr := medium.Read("out.jsonl")
	core.AssertNoError(t, readErr)
	core.AssertEqual(t, "{\"prompt\":\"hi\",\"response\":\"hello-item-1\"}\n{\"prompt\":\"hi\",\"response\":\"hello-item-2\"}\n", content)
}

func TestExportRows_Bad(t *core.T) {
	items := []Item{{Kind: KindTrace, Content: []byte(`{}`), ContentHash: "x"}}
	medium := coreio.NewMemoryMedium()
	written, skipped, hashes, err := exportRows(medium, "out.jsonl", items, writePairsRow)
	core.AssertNoError(t, err, "an unrepresentable row is a counted skip, not an error")
	core.AssertEqual(t, 0, written)
	core.AssertEqual(t, 1, skipped)
	core.AssertLen(t, hashes, 0)
}

func TestExportRows_Ugly(t *core.T) {
	medium := &failingWriteMedium{MemoryMedium: coreio.NewMemoryMedium()}
	_, _, _, err := exportRows(medium, "out.jsonl", nil, writePairsRow)
	core.AssertError(t, err, "simulated write failure")
}

// ---- ExportDataset: golden checks per format ----

func exportSeededStore(t *core.T) (*MemoryStore, Dataset) {
	store := NewMemoryStore()
	ds := fixtureDataset("ds-1", "evening-vents")
	core.RequireTrue(t, store.DatasetCreate(ds).OK)

	approved := fixtureItem("item-approved", ds.ID, 1000)
	core.RequireTrue(t, store.ItemAppend(approved).OK)
	core.RequireTrue(t, store.ReviewAppend(fixtureReview(approved.ID, StatusApproved, 1500)).OK)

	pending := Item{
		ID: "item-pending", DatasetID: ds.ID, Kind: KindPair,
		Content: fixturePairContent("still", "pending"), Source: SourceImportJSONL,
		ContentHash: ContentHash(KindPair, fixturePairContent("still", "pending")).Value.(string),
		CreatedAt:   timeAt(2000),
	}
	core.RequireTrue(t, store.ItemAppend(pending).OK)
	// pending has no Review row at all — implicitly StatusPending.

	return store, ds
}

func TestExportDataset_Good(t *core.T) {
	store, ds := exportSeededStore(t)

	// pairs-jsonl
	medium := coreio.NewMemoryMedium()
	r := ExportDataset(store, medium, ExportRequest{DatasetID: ds.ID, Format: FormatPairsJSONL, OutputPath: "out.jsonl"})
	core.AssertTrue(t, r.OK)
	export := r.Value.(Export)
	core.AssertEqual(t, 1, export.ItemCount, "only the approved item — the default filter excludes pending")
	core.AssertEqual(t, "status=approved", export.FilterDescription)

	content := core.MustCast[string](core.ResultOf(medium.Read("out.jsonl")))
	core.AssertEqual(t, "{\"prompt\":\"hi\",\"response\":\"hello-item-approved\"}\n", content)

	manifestText := core.MustCast[string](core.ResultOf(medium.Read("out.jsonl.manifest.json")))
	var manifest ExportManifest
	core.RequireTrue(t, core.JSONUnmarshalString(manifestText, &manifest).OK)
	core.AssertEqual(t, 1, manifest.ItemCount)
	core.AssertEqual(t, export.ManifestHash, manifest.ManifestHash)
	core.AssertLen(t, manifest.ContentHashes, 1)

	stored := core.MustCast[[]Export](store.Exports(ds.ID))
	core.AssertLen(t, stored, 1)

	// sft-jsonl
	medium2 := coreio.NewMemoryMedium()
	r = ExportDataset(store, medium2, ExportRequest{DatasetID: ds.ID, Format: FormatSFTJSONL, OutputPath: "out.jsonl"})
	core.AssertTrue(t, r.OK)
	content = core.MustCast[string](core.ResultOf(medium2.Read("out.jsonl")))
	core.AssertEqual(t, "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"},{\"role\":\"assistant\",\"content\":\"hello-item-approved\"}]}\n", content)

	// capture-jsonl
	medium3 := coreio.NewMemoryMedium()
	r = ExportDataset(store, medium3, ExportRequest{DatasetID: ds.ID, Format: FormatCaptureJSONL, OutputPath: "out.jsonl"})
	core.AssertTrue(t, r.OK)
	content = core.MustCast[string](core.ResultOf(medium3.Read("out.jsonl")))
	core.AssertEqual(t, "{\"step\":0,\"prompt\":\"hi\",\"text\":\"hello-item-approved\",\"at_unix\":1000}\n", content)
}

func TestExportDataset_Bad(t *core.T) {
	store, ds := exportSeededStore(t)
	medium := coreio.NewMemoryMedium()

	core.AssertFalse(t, ExportDataset(nil, medium, ExportRequest{DatasetID: ds.ID, Format: FormatPairsJSONL, OutputPath: "out.jsonl"}).OK, "a nil store must fail")
	core.AssertFalse(t, ExportDataset(store, nil, ExportRequest{DatasetID: ds.ID, Format: FormatPairsJSONL, OutputPath: "out.jsonl"}).OK, "a nil medium must fail")
	core.AssertFalse(t, ExportDataset(store, medium, ExportRequest{Format: FormatPairsJSONL, OutputPath: "out.jsonl"}).OK, "an empty dataset id must fail")
	core.AssertFalse(t, ExportDataset(store, medium, ExportRequest{DatasetID: ds.ID, Format: FormatPairsJSONL}).OK, "an empty output path must fail")
	core.AssertFalse(t, ExportDataset(store, medium, ExportRequest{DatasetID: ds.ID, Format: ExportFormat("bogus"), OutputPath: "out.jsonl"}).OK, "an unknown format must fail")
	core.AssertFalse(t, ExportDataset(store, medium, ExportRequest{DatasetID: "missing-dataset", Format: FormatPairsJSONL, OutputPath: "out.jsonl"}).OK, "an unknown dataset must fail")

	failing := &failingItemsStore{MemoryStore: store}
	core.AssertFalse(t, ExportDataset(failing, medium, ExportRequest{DatasetID: ds.ID, Format: FormatPairsJSONL, OutputPath: "out.jsonl"}).OK, "an items-query failure must propagate")

	failingRows := &failingWriteMedium{MemoryMedium: coreio.NewMemoryMedium()}
	core.AssertFalse(t, ExportDataset(store, failingRows, ExportRequest{DatasetID: ds.ID, Format: FormatPairsJSONL, OutputPath: "out.jsonl"}).OK, "a row-write failure must propagate")

	failingManifest := &failingManifestWriteMedium{MemoryMedium: coreio.NewMemoryMedium()}
	core.AssertFalse(t, ExportDataset(store, failingManifest, ExportRequest{DatasetID: ds.ID, Format: FormatPairsJSONL, OutputPath: "out.jsonl"}).OK, "a manifest-write failure must propagate")
}

func TestExportDataset_Ugly(t *core.T) {
	store, ds := exportSeededStore(t)

	// Exporting anything other than the default requires an explicit
	// status — here, everything (approved + pending).
	medium := coreio.NewMemoryMedium()
	r := ExportDataset(store, medium, ExportRequest{
		DatasetID: ds.ID, Format: FormatPairsJSONL, OutputPath: "out.jsonl",
		Filter: ItemFilter{Status: StatusPending}, FilterDescription: "status=pending",
	})
	core.AssertTrue(t, r.OK)
	core.AssertEqual(t, 1, r.Value.(Export).ItemCount, "only the pending item matches this explicit filter")
	core.AssertEqual(t, "status=pending", r.Value.(Export).FilterDescription)

	// A dataset with nothing matching the filter is an honest empty
	// export, not an error — the manifest hash is the empty-string hash.
	medium2 := coreio.NewMemoryMedium()
	r = ExportDataset(store, medium2, ExportRequest{DatasetID: ds.ID, Format: FormatPairsJSONL, OutputPath: "out.jsonl", Filter: ItemFilter{Status: StatusRejected}, FilterDescription: "status=rejected"})
	core.AssertTrue(t, r.OK)
	export := r.Value.(Export)
	core.AssertEqual(t, 0, export.ItemCount)
	core.AssertEqual(t, core.SHA256HexString(""), export.ManifestHash)

	// A format-incompatible item (trace, no prompt/response) is a
	// counted skip, not a call-aborting error.
	traceItem := Item{
		ID: "item-trace", DatasetID: ds.ID, Kind: KindTrace, Content: []byte(`{"logits":[1]}`),
		Source: SourceSSD, ContentHash: ContentHash(KindTrace, []byte(`{"logits":[1]}`)).Value.(string), CreatedAt: timeAt(3000),
	}
	core.RequireTrue(t, store.ItemAppend(traceItem).OK)
	core.RequireTrue(t, store.ReviewAppend(fixtureReview(traceItem.ID, StatusApproved, 3500)).OK)
	medium3 := coreio.NewMemoryMedium()
	r = ExportDataset(store, medium3, ExportRequest{DatasetID: ds.ID, Format: FormatPairsJSONL, OutputPath: "out.jsonl"})
	core.AssertTrue(t, r.OK)
	core.AssertEqual(t, 1, r.Value.(Export).ItemCount, "the trace item is skipped; the approved pair item still exports")
}
