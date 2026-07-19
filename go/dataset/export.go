// SPDX-Licence-Identifier: EUPL-1.2

package dataset

import (
	"bufio"
	"time"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

// captureExportRow mirrors train.CaptureRow's JSON shape by field name —
// see ingest.go's jsonlRow for the same convention — not imported, to
// keep this package free of train's engine dependency. Step is the
// item's zero-based ordinal position within this export (a general
// dataset Item carries no training-loop step counter of its own).
type captureExportRow struct {
	Step   int    `json:"step"`
	Prompt string `json:"prompt"`
	Text   string `json:"text"`
	At     int64  `json:"at_unix"`
}

// itemAsPair reduces an Item to a PairContent regardless of Kind: a
// KindPair item's content directly, a KindMessages item's
// [MessagesContent.LastExchange] reduction. A KindTrace item is opaque
// and never representable as a pair.
func itemAsPair(item Item) (PairContent, bool) {
	switch item.Kind {
	case KindPair:
		var pc PairContent
		if r := core.JSONUnmarshal(item.Content, &pc); !r.OK {
			return PairContent{}, false
		}
		return pc, true
	case KindMessages:
		var mc MessagesContent
		if r := core.JSONUnmarshal(item.Content, &mc); !r.OK {
			return PairContent{}, false
		}
		return mc.LastExchange()
	default:
		return PairContent{}, false
	}
}

// writeSFTRow renders item as one sft-jsonl row: {"messages":[...]} —
// the `lem train --sft` contract. A KindMessages item's content is
// already this shape; a KindPair item is wrapped via the standard
// user/assistant turns. KindTrace is not representable.
func writeSFTRow(item Item, _ int) ([]byte, bool) {
	switch item.Kind {
	case KindMessages:
		var mc MessagesContent
		if r := core.JSONUnmarshal(item.Content, &mc); !r.OK {
			return nil, false
		}
		encoded := core.JSONMarshal(mc)
		if !encoded.OK {
			return nil, false
		}
		return encoded.Bytes(), true
	case KindPair:
		pc, ok := itemAsPair(item)
		if !ok {
			return nil, false
		}
		mc := MessagesContent{Messages: []MessageTurn{
			{Role: "user", Content: pc.Prompt},
			{Role: "assistant", Content: pc.Response},
		}}
		encoded := core.JSONMarshal(mc)
		if !encoded.OK {
			return nil, false
		}
		return encoded.Bytes(), true
	default:
		return nil, false
	}
}

// writePairsRow renders item as one pairs-jsonl row:
// {"prompt":...,"response":...}.
func writePairsRow(item Item, _ int) ([]byte, bool) {
	pc, ok := itemAsPair(item)
	if !ok {
		return nil, false
	}
	encoded := core.JSONMarshal(pc)
	if !encoded.OK {
		return nil, false
	}
	return encoded.Bytes(), true
}

// writeCaptureRow renders item as one capture-jsonl row, round-tripping
// train.CaptureRow's field names: {"step":...,"prompt":...,"text":...,
// "at_unix":...}. text is the response — CaptureRow's own doc calls it
// "the raw return text". at_unix is the item's CreatedAt (its true
// generation time when known — see ingest.go's CaptureRow ingest path).
func writeCaptureRow(item Item, ordinal int) ([]byte, bool) {
	pc, ok := itemAsPair(item)
	if !ok {
		return nil, false
	}
	encoded := core.JSONMarshal(captureExportRow{Step: ordinal, Prompt: pc.Prompt, Text: pc.Response, At: item.CreatedAt.Unix()})
	if !encoded.OK {
		return nil, false
	}
	return encoded.Bytes(), true
}

// rowWriter renders one Item as one export row, or reports it cannot be
// represented in this format (a counted skip, never a call-aborting
// error — the same honesty rule ingest.go's malformed rows follow).
// ordinal is the item's zero-based position among the rows written so
// far (writeCaptureRow's synthetic Step; ignored by the other writers).
type rowWriter func(item Item, ordinal int) ([]byte, bool)

// writerFor resolves the rowWriter for format, or nil for an unknown
// format.
func writerFor(format ExportFormat) rowWriter {
	switch format {
	case FormatSFTJSONL:
		return writeSFTRow
	case FormatPairsJSONL:
		return writePairsRow
	case FormatCaptureJSONL:
		return writeCaptureRow
	default:
		return nil
	}
}

// describeFilter renders filter's set dimensions as a compact,
// deterministic string — the manifest's human-readable filter
// description when the caller does not supply one explicitly.
func describeFilter(filter ItemFilter) string {
	var parts []string
	if filter.Status != "" {
		parts = append(parts, core.Concat("status=", string(filter.Status)))
	}
	if filter.Kind != "" {
		parts = append(parts, core.Concat("kind=", string(filter.Kind)))
	}
	if filter.Source != "" {
		parts = append(parts, core.Concat("source=", string(filter.Source)))
	}
	if filter.Score != nil {
		parts = append(parts, filter.Score.String())
	}
	if filter.IncludeArchived {
		parts = append(parts, "include_archived=true")
	}
	if len(parts) == 0 {
		return "none"
	}
	return core.Join(",", parts...)
}

// ExportManifest is the sidecar JSON a training run reads to know
// exactly what it saw — written alongside the JSONL at
// "<OutputPath>.manifest.json".
type ExportManifest struct {
	DatasetID         string    `json:"dataset_id"`
	Format            string    `json:"format"`
	FilterDescription string    `json:"filter"`
	ItemCount         int       `json:"item_count"`
	SkippedCount      int       `json:"skipped_count"`
	ContentHashes     []string  `json:"content_hashes"`
	ManifestHash      string    `json:"manifest_hash"`
	CreatedAt         time.Time `json:"created_at"`
}

// ExportRequest is one dataset export's inputs. Filter.DatasetID is
// overwritten from DatasetID. Filter.Status left "" defaults to
// StatusApproved — "no accidental training on unreviewed data"; set it
// explicitly to export anything else. FilterDescription is the
// manifest's human-readable filter text — auto-derived from the
// resolved filter when left "".
type ExportRequest struct {
	DatasetID         string
	Format            ExportFormat
	Filter            ItemFilter
	FilterDescription string
	OutputPath        string
}

// exportRows writes items to outputPath through medium using writer,
// returning the written/skipped counts and the written items' content
// hashes in write order. The file is flushed and closed before this
// returns.
func exportRows(medium coreio.Medium, outputPath string, items []Item, writer rowWriter) (written, skipped int, hashes []string, err error) {
	wc, openErr := medium.WriteStream(outputPath)
	if openErr != nil {
		return 0, 0, nil, core.E("dataset.Export", "open output path", openErr)
	}
	defer func() { _ = wc.Close() }()

	buffered := bufio.NewWriter(wc)
	hashes = make([]string, 0, len(items))
	for _, item := range items {
		row, ok := writer(item, written)
		if !ok {
			skipped++
			continue
		}
		if _, writeErr := buffered.Write(row); writeErr != nil {
			return written, skipped, hashes, core.E("dataset.Export", "write row", writeErr)
		}
		if writeErr := buffered.WriteByte('\n'); writeErr != nil {
			return written, skipped, hashes, core.E("dataset.Export", "write row", writeErr)
		}
		hashes = append(hashes, item.ContentHash)
		written++
	}
	if flushErr := buffered.Flush(); flushErr != nil {
		return written, skipped, hashes, core.E("dataset.Export", "flush output", flushErr)
	}
	return written, skipped, hashes, nil
}

// ExportDataset runs one dataset export end to end: resolves the
// default filter (status=approved unless the caller set an explicit
// Status), fetches matching items from store, orders them
// deterministically by (created_at, id), writes the format-specific
// JSONL to medium at req.OutputPath, writes the sidecar
// [ExportManifest] JSON at "<OutputPath>.manifest.json", records an
// [Export] receipt via store.ExportAppend, and returns it.
//
//	medium := io.NewMemoryMedium() // or io.Local in production
//	r := dataset.ExportDataset(store, medium, dataset.ExportRequest{
//	    DatasetID: ds.ID, Format: dataset.FormatPairsJSONL, OutputPath: "/out/train.jsonl",
//	})
func ExportDataset(store Store, medium coreio.Medium, req ExportRequest) core.Result {
	if store == nil {
		return core.Fail(core.NewError("dataset: export requires a store"))
	}
	if medium == nil {
		return core.Fail(core.NewError("dataset: export requires a medium"))
	}
	if core.Trim(req.DatasetID) == "" {
		return core.Fail(core.NewError("dataset: export requires a dataset id"))
	}
	if core.Trim(req.OutputPath) == "" {
		return core.Fail(core.NewError("dataset: export requires an output path"))
	}
	writer := writerFor(req.Format)
	if writer == nil {
		return core.Fail(core.NewError("dataset: export requires a known format"))
	}

	filter := req.Filter
	filter.DatasetID = req.DatasetID
	if filter.Status == "" {
		filter.Status = StatusApproved
	}
	filterDescription := req.FilterDescription
	if filterDescription == "" {
		filterDescription = describeFilter(filter)
	}

	itemsResult := store.Items(filter)
	if !itemsResult.OK {
		return itemsResult
	}
	items := itemsResult.Value.([]Item)
	sortByCreated(items, func(item Item) (time.Time, string) { return item.CreatedAt, item.ID })

	written, skipped, hashes, err := exportRows(medium, req.OutputPath, items, writer)
	if err != nil {
		return core.Fail(err)
	}

	manifestHash := ManifestHash(hashes)
	manifest := ExportManifest{
		DatasetID: req.DatasetID, Format: string(req.Format), FilterDescription: filterDescription,
		ItemCount: written, SkippedCount: skipped, ContentHashes: hashes, ManifestHash: manifestHash, CreatedAt: time.Now(),
	}
	manifestBytes := core.JSONMarshalIndent(manifest, "", "  ")
	if !manifestBytes.OK {
		return core.Fail(core.E("dataset.Export", "marshal manifest", manifestBytes.Err()))
	}
	manifestPath := core.Concat(req.OutputPath, ".manifest.json")
	if writeErr := medium.Write(manifestPath, core.AsString(manifestBytes.Bytes())); writeErr != nil {
		return core.Fail(core.E("dataset.Export", "write manifest", writeErr))
	}

	export := Export{
		ID: NewID(), DatasetID: req.DatasetID, Format: req.Format, FilterDescription: filterDescription,
		ItemCount: written, OutputPath: req.OutputPath, ManifestHash: manifestHash, CreatedAt: time.Now(),
	}
	if r := ValidateExport(export); !r.OK {
		return r
	}
	return store.ExportAppend(export)
}
