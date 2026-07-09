// SPDX-Licence-Identifier: EUPL-1.2

package artifact

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/kv"
	memvid "dappco.re/go/inference/model/state"
)

func TestExport_Good(t *testing.T) {
	store := memvid.NewInMemoryStore(nil)
	path := core.PathJoin(t.TempDir(), "state.kvbin")

	record, err := Export(context.Background(), testSnapshot(), Options{
		Model:  "lem-gemma",
		Prompt: "trace me",
		KVPath: path,
		Store:  store,
		URI:    "mlx://session/lem-gemma/trace",
		Title:  "LEM Gemma trace",
		Tags:   map[string]string{"arch": "gemma4_text"},
	})

	if err != nil {
		t.Fatalf("Export() error = %v", err)
	}
	if record.KVPath != path {
		t.Fatalf("KVPath = %q, want %q", record.KVPath, path)
	}
	if record.ChunkRef.Codec != memvid.CodecMemory || record.ChunkRef.ChunkID == 0 {
		t.Fatalf("ChunkRef = %#v, want memory chunk", record.ChunkRef)
	}
	if record.SAMI.Model != "lem-gemma" || len(record.Features) != len(kv.FeatureLabels()) {
		t.Fatalf("record = %+v", record)
	}
	if _, err := kv.Load(path); err != nil {
		t.Fatalf("kv.Load() error = %v", err)
	}
	chunk, err := store.Resolve(context.Background(), record.ChunkRef.ChunkID)
	if err != nil {
		t.Fatalf("Resolve() error = %v", err)
	}
	if !core.Contains(chunk.Text, `"sami"`) || !core.Contains(chunk.Text, `"feature_labels"`) {
		t.Fatalf("artifact chunk text = %q", chunk.Text)
	}
}

func TestExport_Bad(t *testing.T) {
	_, err := Export(context.Background(), nil, Options{})

	if err == nil {
		t.Fatal("expected nil snapshot error")
	}
}

func TestExport_Ugly(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := Export(ctx, testSnapshot(), Options{})

	if !core.Is(err, context.Canceled) {
		t.Fatalf("Export() error = %v, want context.Canceled", err)
	}
}

func testSnapshot() *kv.Snapshot {
	return &kv.Snapshot{
		Version:       kv.SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{1, 2},
		NumLayers:     2,
		NumHeads:      1,
		SeqLen:        2,
		HeadDim:       2,
		NumQueryHeads: 8,
		Layers: []kv.LayerSnapshot{
			{
				Layer:      0,
				CacheIndex: 0,
				Heads: []kv.HeadSnapshot{{
					Key:   []float32{1, 0, 0, 1},
					Value: []float32{0, 1, 1, 0},
				}},
			},
			{
				Layer:      1,
				CacheIndex: 1,
				Heads: []kv.HeadSnapshot{{
					Key:   []float32{1, 1, 0, 0},
					Value: []float32{0, 0, 1, 1},
				}},
			},
		},
	}
}
