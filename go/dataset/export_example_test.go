// SPDX-Licence-Identifier: EUPL-1.2

package dataset_test

import (
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/dataset"
	coreio "dappco.re/go/io"
)

// ExampleExportDataset writes an approved-only pairs-jsonl export plus
// its sidecar manifest, and records the receipt.
func ExampleExportDataset() {
	store := dataset.NewMemoryStore()
	ds := dataset.Dataset{ID: "ds-1", Slug: "evening-vents", Title: "Evening vents", CreatedAt: time.Unix(1000, 0)}
	core.MustCast[dataset.Dataset](store.DatasetCreate(ds))

	content := core.JSONMarshal(dataset.PairContent{Prompt: "hi", Response: "hello"}).Value.([]byte)
	hash := core.MustCast[string](dataset.ContentHash(dataset.KindPair, content))
	item := dataset.Item{ID: "item-1", DatasetID: ds.ID, Kind: dataset.KindPair, Content: content, Source: dataset.SourceImportJSONL, ContentHash: hash, CreatedAt: time.Unix(2000, 0)}
	core.MustCast[dataset.Item](store.ItemAppend(item))
	core.MustCast[dataset.Review](store.ReviewAppend(dataset.Review{ItemID: item.ID, Status: dataset.StatusApproved, Reviewer: "snider", CreatedAt: time.Unix(2500, 0)}))

	medium := coreio.NewMemoryMedium()
	r := dataset.ExportDataset(store, medium, dataset.ExportRequest{
		DatasetID: ds.ID, Format: dataset.FormatPairsJSONL, OutputPath: "train.jsonl",
	})
	export := r.Value.(dataset.Export)
	core.Println(export.ItemCount, export.FilterDescription)
	// Output:
	// 1 status=approved
}
