// SPDX-Licence-Identifier: EUPL-1.2

package session

import (
	"context"
	"iter"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/kv"
	mlxbundle "dappco.re/go/inference/model/bundle"
	"dappco.re/go/inference/model/spine"
	memvid "dappco.re/go/inference/model/state"
	"dappco.re/go/inference/model/state/agent"
	"dappco.re/go/inference/model/state/session/internal/sessionfake"
	"dappco.re/go/inference/decode/parser"
	"dappco.re/go/inference/eval/probe"
)

// Local inference.GenerateOption builders — the WithX functional options are
// root mlx API (which this package cannot import); tests set the same
// fields directly.
func optMaxTokens(n int) inference.GenerateOption {
	return func(c *inference.GenerateConfig) { c.MaxTokens = n }
}

func optTemperature(t float32) inference.GenerateOption {
	return func(c *inference.GenerateConfig) { c.Temperature = t }
}

func optMinP(p float32) inference.GenerateOption {
	return func(c *inference.GenerateConfig) { c.MinP = p }
}

func optTopK(k int) inference.GenerateOption {
	return func(c *inference.GenerateConfig) { c.TopK = k }
}

func optProbeSink(sink probe.Sink) inference.GenerateOption {
	return func(c *inference.GenerateConfig) { c.ProbeSink = sink }
}

func optHideThinking() inference.GenerateOption {
	return func(c *inference.GenerateConfig) { c.Thinking.Mode = parser.Hide }
}

func seqStrings(values ...string) iter.Seq[string] {
	return func(yield func(string) bool) {
		for _, v := range values {
			if !yield(v) {
				return
			}
		}
	}
}

func TestSessionPrefillAndGenerate_Good(t *testing.T) {
	nativeSession := &sessionfake.Handle{
		Tokens: []inference.Token{{ID: 1, Text: "A"}, {ID: 2, Text: "B"}},
	}
	session := &Session{session: nativeSession}

	if err := session.Prefill("stable context"); err != nil {
		t.Fatalf("Prefill() error = %v", err)
	}
	got, err := session.Generate(optMaxTokens(12), optTemperature(0.2), optMinP(0.05))

	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if got != "AB" {
		t.Fatalf("Generate() = %q, want AB", got)
	}
	if nativeSession.PrefillPrompt != "stable context" {
		t.Fatalf("prefill prompt = %q, want stable context", nativeSession.PrefillPrompt)
	}
	if nativeSession.Cfg.MaxTokens != 12 || nativeSession.Cfg.Temperature != 0.2 || nativeSession.Cfg.MinP != 0.05 {
		t.Fatalf("Generate config = %+v", nativeSession.Cfg)
	}
}

func TestSessionPrefillChunks_Good(t *testing.T) {
	nativeSession := &sessionfake.Handle{}
	session := &Session{session: nativeSession}

	if err := session.PrefillChunks(context.Background(), seqStrings("stable ", "context")); err != nil {
		t.Fatalf("PrefillChunks() error = %v", err)
	}

	if got := core.Join("", nativeSession.PrefillChunksSeen...); got != "stable context" {
		t.Fatalf("prefill chunks = %#v, joined %q", nativeSession.PrefillChunksSeen, got)
	}
}

func TestSessionPrefillTokens_Good(t *testing.T) {
	nativeSession := &sessionfake.Handle{}
	session := &Session{session: nativeSession}
	tokens := []int32{11, 12}

	if err := session.PrefillTokens(context.Background(), tokens); err != nil {
		t.Fatalf("PrefillTokens() error = %v", err)
	}
	tokens[0] = 99

	if got := nativeSession.PrefillTokensSeen; len(got) != 2 || got[0] != 11 || got[1] != 12 {
		t.Fatalf("prefill tokens = %v, want copied 11/12", got)
	}
}

func TestSessionAppendPrompt_Good(t *testing.T) {
	nativeSession := &sessionfake.Handle{}
	session := &Session{session: nativeSession}

	if err := session.AppendPrompt("\n\nQuestion: who?\nAnswer:"); err != nil {
		t.Fatalf("AppendPrompt() error = %v", err)
	}

	if nativeSession.AppendPromptSeen != "\n\nQuestion: who?\nAnswer:" {
		t.Fatalf("append prompt = %q", nativeSession.AppendPromptSeen)
	}
}

func TestSessionAppendTokens_Good(t *testing.T) {
	nativeSession := &sessionfake.Handle{}
	session := &Session{session: nativeSession}
	tokens := []int32{21, 22}

	if err := session.AppendTokens(context.Background(), tokens); err != nil {
		t.Fatalf("AppendTokens() error = %v", err)
	}
	tokens[0] = 99

	if got := nativeSession.AppendTokensSeen; len(got) != 2 || got[0] != 21 || got[1] != 22 {
		t.Fatalf("append tokens = %v, want copied 21/22", got)
	}
}

func TestSessionAppendPromptChunks_Good(t *testing.T) {
	nativeSession := &sessionfake.Handle{}
	session := &Session{session: nativeSession}

	if err := session.AppendPromptChunks(context.Background(), seqStrings("\n\nQuestion: ", "who?\nAnswer:")); err != nil {
		t.Fatalf("AppendPromptChunks() error = %v", err)
	}

	if got := core.Join("", nativeSession.AppendChunksSeen...); got != "\n\nQuestion: who?\nAnswer:" {
		t.Fatalf("append chunks = %#v, joined %q", nativeSession.AppendChunksSeen, got)
	}
}

func TestSessionNilGuards_Bad(t *testing.T) {
	var session *Session
	if err := session.AppendPrompt("x"); err == nil {
		t.Fatal("expected nil append prompt error")
	}
	if err := session.AppendPromptChunks(context.Background(), seqStrings("x")); err == nil {
		t.Fatal("expected nil append prompt chunks error")
	}
	if err := session.PrefillChunks(context.Background(), seqStrings("x")); err == nil {
		t.Fatal("expected nil prefill chunks error")
	}
	if err := session.AppendTokens(context.Background(), []int32{1}); err == nil {
		t.Fatal("expected nil append tokens error")
	}
	if err := session.PrefillTokens(context.Background(), []int32{1}); err == nil {
		t.Fatal("expected nil prefill tokens error")
	}
	if text, err := session.Generate(); err == nil || text != "" {
		t.Fatalf("Generate(nil) = %q/%v, want error", text, err)
	}
	if err := session.RestoreKV(nil); err == nil {
		t.Fatal("expected nil session restore error")
	}
	if err := (&Session{}).RestoreKV(nil); err == nil {
		t.Fatal("expected empty session restore error")
	}
	if err := (&Session{session: &sessionfake.Handle{}}).RestoreKV(nil); err == nil {
		t.Fatal("expected nil KV snapshot error")
	}
	if _, err := session.SaveKVToMemvid(nil, memvid.NewInMemoryStore(nil), kv.MemvidOptions{}); err == nil {
		t.Fatal("expected nil session save-to-memvid error")
	}
	if _, err := session.SaveKVBlocksToMemvid(nil, memvid.NewInMemoryStore(nil), kv.MemvidBlockOptions{}); err == nil {
		t.Fatal("expected nil session save-blocks error")
	}
	if err := session.LoadKVBlocksFromMemvid(nil, memvid.NewInMemoryStore(nil), &kv.MemvidBlockBundle{}); err == nil {
		t.Fatal("expected invalid memvid block load error")
	}
	if err := session.RestoreBundle(nil); err == nil {
		t.Fatal("expected nil bundle restore error")
	}
	if err := session.RestoreBundleFromMemvid(nil, nil, memvid.NewInMemoryStore(nil)); err == nil {
		t.Fatal("expected nil memvid bundle restore error")
	}
	if err := session.LoadBundle(core.PathJoin(t.TempDir(), "missing.bundle.json")); err == nil {
		t.Fatal("expected missing bundle load error")
	}
	session.Reset()
	if err := session.Close(); err != nil {
		t.Fatalf("Close(nil) = %v, want nil", err)
	}
	if err := session.Err(); err != nil {
		t.Fatalf("Err(nil) = %v, want nil", err)
	}
}

func TestSessionGenerate_ForwardsProbeSink_Good(t *testing.T) {
	recorder := probe.NewRecorder()
	nativeSession := &sessionfake.Handle{
		ProbeEvents: []probe.Event{{
			Kind:  probe.KindEntropy,
			Phase: probe.PhaseDecode,
			Step:  1,
			Entropy: &probe.Entropy{
				Value: 0.42,
			},
		}},
	}
	session := &Session{session: nativeSession}

	if _, err := session.Generate(optProbeSink(recorder)); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}

	if nativeSession.Cfg.ProbeSink == nil {
		t.Fatal("native probe.Sink = nil, want configured")
	}
	events := recorder.Events()
	if len(events) != 1 {
		t.Fatalf("probe events len = %d, want 1", len(events))
	}
	if events[0].Kind != probe.KindEntropy || events[0].Entropy == nil || events[0].Entropy.Value != 0.42 {
		t.Fatalf("probe event = %+v", events[0])
	}
}

func TestModelSessionMemvidKV_Good_SaveAndLoad(t *testing.T) {
	store := memvid.NewInMemoryStore(nil)
	nativeSession := &sessionfake.Handle{
		KV: &kv.Snapshot{
			Version:       kv.SnapshotVersion,
			Architecture:  "gemma4_text",
			Tokens:        []int32{10, 20},
			Generated:     []int32{30},
			TokenOffset:   2,
			NumLayers:     1,
			NumHeads:      1,
			SeqLen:        2,
			HeadDim:       2,
			NumQueryHeads: 1,
			LogitShape:    []int32{1, 1, 2},
			Logits:        []float32{0.25, 0.75},
			Layers: []kv.LayerSnapshot{{
				Layer:      0,
				CacheIndex: 0,
				Heads: []kv.HeadSnapshot{{
					Key:   []float32{1, 2, 3, 4},
					Value: []float32{5, 6, 7, 8},
				}},
			}},
		},
	}
	session := &Session{session: nativeSession}

	ref, err := session.SaveKVToMemvid(context.Background(), store, kv.MemvidOptions{URI: "mlx://session/demo"})
	if err != nil {
		t.Fatalf("SaveKVToMemvid() error = %v", err)
	}
	restoredNative := &sessionfake.Handle{}
	restored := &Session{session: restoredNative}
	if err := restored.LoadKVFromMemvid(context.Background(), store, ref); err != nil {
		t.Fatalf("LoadKVFromMemvid() error = %v", err)
	}

	if restoredNative.RestoredKV == nil || restoredNative.RestoredKV.Tokens[1] != 20 || restoredNative.RestoredKV.Generated[0] != 30 {
		t.Fatalf("restored KV = %+v", restoredNative.RestoredKV)
	}
	if restoredNative.RestoredKV.Logits[1] != 0.75 {
		t.Fatalf("restored logits = %+v", restoredNative.RestoredKV.Logits)
	}
}

func TestModelSessionMemvidBundle_Good_Restore(t *testing.T) {
	store := memvid.NewInMemoryStore(nil)
	snapshot := sessionTestRootSnapshot()
	ref, err := snapshot.SaveMemvid(context.Background(), store, kv.MemvidOptions{})
	if err != nil {
		t.Fatalf("SaveMemvid() error = %v", err)
	}
	hash, err := kv.HashSnapshot(snapshot)
	if err != nil {
		t.Fatalf("kv.HashSnapshot() error = %v", err)
	}
	nativeSession := &sessionfake.Handle{}
	session := &Session{
		session: nativeSession,
		info:    spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1},
	}
	b := &mlxbundle.Bundle{
		Version: mlxbundle.Version,
		Kind:    mlxbundle.Kind,
		Model:   mlxbundle.Model{Architecture: "gemma4_text", NumLayers: 1},
		KVHash:  hash,
		Refs: []mlxbundle.Ref{{
			Kind:   mlxbundle.RefMemvid,
			URI:    mlxbundle.MemvidURI(ref),
			Memvid: ref,
		}},
	}

	if err := session.RestoreBundleFromMemvid(context.Background(), b, store); err != nil {
		t.Fatalf("RestoreBundleFromMemvid() error = %v", err)
	}
	if nativeSession.RestoredKV == nil || nativeSession.RestoredKV.Tokens[0] != 1 {
		t.Fatalf("restored KV = %+v", nativeSession.RestoredKV)
	}
}

func TestModelSessionMemvidKVBlocks_Good_SaveAndLoad(t *testing.T) {
	store := memvid.NewInMemoryStore(nil)
	nativeSession := &sessionfake.Handle{
		CaptureErr: core.NewError("full snapshot capture should not be used"),
		KVBlocks: []kv.Block{
			{
				Index:      0,
				TokenStart: 0,
				TokenCount: 2,
				Snapshot:   testNativeKVBlock([]int32{10, 20}, 2, []float32{1, 2, 3, 4}, []float32{9, 10, 11, 12}, nil, nil),
			},
			{
				Index:      1,
				TokenStart: 2,
				TokenCount: 2,
				Snapshot:   testNativeKVBlock([]int32{30, 40}, 4, []float32{5, 6, 7, 8}, []float32{13, 14, 15, 16}, []float32{0.25, 0.75}, []int32{40}),
			},
		},
	}
	session := &Session{session: nativeSession}

	bundle, err := session.SaveKVBlocksToMemvid(context.Background(), store, kv.MemvidBlockOptions{BlockSize: 2})
	if err != nil {
		t.Fatalf("SaveKVBlocksToMemvid() error = %v", err)
	}
	if len(bundle.Blocks) != 2 {
		t.Fatalf("bundle blocks = %+v, want 2", bundle.Blocks)
	}
	restoredNative := &sessionfake.Handle{}
	restored := &Session{session: restoredNative}
	if err := restored.LoadKVBlocksFromMemvid(context.Background(), store, bundle); err != nil {
		t.Fatalf("LoadKVBlocksFromMemvid() error = %v", err)
	}

	if len(restoredNative.RestoredBlocks) != 2 {
		t.Fatalf("restored blocks = %+v, want 2", restoredNative.RestoredBlocks)
	}
	last := restoredNative.RestoredBlocks[1].Snapshot
	if last == nil || last.Tokens[1] != 40 || last.Generated[0] != 40 {
		t.Fatalf("restored final block KV = %+v", last)
	}
	if last.Layers[0].Heads[0].Value[3] != 16 {
		t.Fatalf("restored final block values = %+v", last.Layers[0].Heads[0].Value)
	}
}

func TestModelSessionMemvidKVBlocks_Good_LoadPrefixStreamsOnlyNeededBlocks(t *testing.T) {
	store := memvid.NewInMemoryStore(nil)
	nativeSession := &sessionfake.Handle{
		KVBlocks: []kv.Block{
			{
				Index:      0,
				TokenStart: 0,
				TokenCount: 2,
				Snapshot:   testNativeKVBlock([]int32{10, 20}, 2, []float32{1, 2, 3, 4}, []float32{9, 10, 11, 12}, nil, nil),
			},
			{
				Index:      1,
				TokenStart: 2,
				TokenCount: 2,
				Snapshot:   testNativeKVBlock([]int32{30, 40}, 4, []float32{5, 6, 7, 8}, []float32{13, 14, 15, 16}, nil, nil),
			},
		},
	}
	session := &Session{session: nativeSession}
	bundle, err := session.SaveKVBlocksToMemvid(context.Background(), store, kv.MemvidBlockOptions{BlockSize: 2})
	if err != nil {
		t.Fatalf("SaveKVBlocksToMemvid() error = %v", err)
	}

	restoredNative := &sessionfake.Handle{}
	restored := &Session{session: restoredNative}
	if err := restored.LoadKVPrefixBlocksFromMemvid(context.Background(), store, bundle, 2); err != nil {
		t.Fatalf("LoadKVPrefixBlocksFromMemvid() error = %v", err)
	}
	if len(restoredNative.RestoredBlocks) != 1 {
		t.Fatalf("restored blocks = %+v, want one streamed prefix block", restoredNative.RestoredBlocks)
	}
	if got := restoredNative.RestoredBlocks[0].Snapshot.Tokens; len(got) != 2 || got[0] != 10 || got[1] != 20 {
		t.Fatalf("restored prefix tokens = %+v, want [10 20]", got)
	}
}

func TestSessionPrefill_Bad(t *testing.T) {
	var session *Session

	if err := session.Prefill("prompt"); err == nil {
		t.Fatal("expected nil session error")
	}
}

func TestSessionGenerate_Ugly(t *testing.T) {
	wantErr := core.NewError("decode failed")
	nativeSession := &sessionfake.Handle{
		Tokens:   []inference.Token{{ID: 1, Text: "partial"}},
		ErrValue: wantErr,
	}
	session := &Session{session: nativeSession}

	_, err := session.Generate()

	if !core.Is(err, wantErr) {
		t.Fatalf("Generate() error = %v, want %v", err, wantErr)
	}
}

func TestSessionGenerateStream_Good(t *testing.T) {
	session := &Session{session: &sessionfake.Handle{
		Tokens: []inference.Token{{ID: 7, Text: "x"}, {ID: 8, Text: "y"}},
	}}

	ch := session.GenerateStream(context.Background(), optTopK(4))
	var got []spine.Token
	timeout := time.After(2 * time.Second)
	for {
		select {
		case tok, ok := <-ch:
			if !ok {
				if len(got) != 2 || got[0].Text != "x" || got[1].Value != "y" {
					t.Fatalf("stream tokens = %+v", got)
				}
				return
			}
			got = append(got, tok)
		case <-timeout:
			t.Fatal("timed out waiting for stream")
		}
	}
}

func TestSessionGenerateStream_HideGemma4Thinking_Good(t *testing.T) {
	session := &Session{
		info: spine.ModelInfo{Architecture: "gemma4_text"},
		session: &sessionfake.Handle{
			Tokens: []inference.Token{
				{ID: 7, Text: "<|channel>thought\nprivate plan"},
				{ID: 8, Text: "<channel|>Chapter 2"},
			},
		},
	}

	ch := session.GenerateStream(context.Background(), optHideThinking())
	got := core.NewBuilder()
	timeout := time.After(2 * time.Second)
	for {
		select {
		case tok, ok := <-ch:
			if !ok {
				if got.String() != "Chapter 2" {
					t.Fatalf("stream text = %q, want Chapter 2", got.String())
				}
				return
			}
			got.WriteString(tok.Text)
		case <-timeout:
			t.Fatal("timed out waiting for stream")
		}
	}
}

func TestSessionParserTokenText_PreservesDecodedContent_Good(t *testing.T) {
	tok := spine.NewTokenizer(fakeRawTokenizer{raw: "Plain"})

	got := sessionParserTokenText(tok, inference.Token{ID: 7, Text: " Plain"})

	if got != " Plain" {
		t.Fatalf("parser token text = %q, want decoded stream text", got)
	}
}

func TestSessionParserTokenText_PreservesControlToken_Good(t *testing.T) {
	tok := spine.NewTokenizer(fakeRawTokenizer{raw: "<|channel>thought\n"})

	got := sessionParserTokenText(tok, inference.Token{ID: 7, Text: ""})

	if got != "<|channel>thought\n" {
		t.Fatalf("parser token text = %q, want raw control token", got)
	}
}

func TestSessionGenerateStream_Bad(t *testing.T) {
	var session *Session

	ch := session.GenerateStream(context.Background())

	if tok, ok := <-ch; ok {
		t.Fatalf("stream yielded %+v, want closed", tok)
	}
}

func TestSessionGenerateStream_Ugly(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	session := &Session{session: &sessionfake.Handle{
		Tokens: []inference.Token{{ID: 7, Text: "x"}},
	}}

	ch := session.GenerateStream(ctx)

	if tok, ok := <-ch; ok {
		t.Fatalf("stream yielded %+v after cancellation", tok)
	}
}

func TestSessionCaptureKVAnalyzeAndSave_Good(t *testing.T) {
	native := &sessionfake.Handle{
		KV: &kv.Snapshot{
			Version:       kv.SnapshotVersion,
			Architecture:  "gemma4_text",
			Tokens:        []int32{1, 2},
			NumLayers:     1,
			NumHeads:      1,
			SeqLen:        2,
			HeadDim:       2,
			NumQueryHeads: 8,
			Layers: []kv.LayerSnapshot{{
				Layer:      0,
				CacheIndex: 0,
				Heads: []kv.HeadSnapshot{{
					Key:   []float32{1, 0, 0, 1},
					Value: []float32{0, 1, 1, 0},
				}},
			}},
		},
	}
	session := &Session{session: native}

	snapshot, err := session.CaptureKV()

	if err != nil {
		t.Fatalf("CaptureKV() error = %v", err)
	}
	if snapshot.Architecture != "gemma4_text" || snapshot.NumQueryHeads != 8 {
		t.Fatalf("CaptureKV() = %+v", snapshot)
	}
	snapshot.Tokens[0] = 99
	if native.KV.Tokens[0] != 1 {
		t.Fatal("CaptureKV() returned aliased token data")
	}
	analysis, err := session.AnalyzeKV()
	if err != nil {
		t.Fatalf("kv.Analyze() error = %v", err)
	}
	if analysis == nil || len(kv.Features(analysis)) != 7 {
		t.Fatalf("kv.Analyze() = %+v", analysis)
	}
	path := core.PathJoin(t.TempDir(), "session.kvbin")
	if err := session.SaveKV(path); err != nil {
		t.Fatalf("SaveKV() error = %v", err)
	}
	loaded, err := kv.Load(path)
	if err != nil {
		t.Fatalf("kv.Load() error = %v", err)
	}
	if loaded.Architecture != "gemma4_text" || loaded.SeqLen != 2 {
		t.Fatalf("loaded snapshot = %+v", loaded)
	}
}

func TestSessionRestoreAndLoadKV_Good(t *testing.T) {
	native := &sessionfake.Handle{}
	session := &Session{session: native}
	snapshot := &kv.Snapshot{
		Version:       kv.SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{1, 2},
		Generated:     []int32{2},
		TokenOffset:   2,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        2,
		HeadDim:       1,
		NumQueryHeads: 8,
		LogitShape:    []int32{1, 1, 3},
		Logits:        []float32{0.1, 0.2, 0.7},
		Layers: []kv.LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []kv.HeadSnapshot{{
				Key:   []float32{1, 2},
				Value: []float32{3, 4},
			}},
		}},
	}

	if err := session.RestoreKV(snapshot); err != nil {
		t.Fatalf("RestoreKV() error = %v", err)
	}
	if native.RestoredKV == nil || native.RestoredKV.Generated[0] != 2 {
		t.Fatalf("restored KV = %+v", native.RestoredKV)
	}
	native.RestoredKV = nil
	path := core.PathJoin(t.TempDir(), "restore.kvbin")
	if err := snapshot.Save(path); err != nil {
		t.Fatalf("Save() error = %v", err)
	}
	if err := session.LoadKV(path); err != nil {
		t.Fatalf("LoadKV() error = %v", err)
	}
	if native.RestoredKV == nil || native.RestoredKV.TokenOffset != 2 {
		t.Fatalf("loaded KV restore = %+v", native.RestoredKV)
	}
}

func TestSessionExportBundle_Good(t *testing.T) {
	native := &sessionfake.Handle{
		KV: &kv.Snapshot{
			Version:       kv.SnapshotVersion,
			Architecture:  "gemma4_text",
			Tokens:        []int32{1, 2},
			Generated:     []int32{2},
			TokenOffset:   2,
			NumLayers:     1,
			NumHeads:      1,
			SeqLen:        2,
			HeadDim:       2,
			NumQueryHeads: 8,
			LogitShape:    []int32{1, 1, 3},
			Logits:        []float32{0.1, 0.2, 0.7},
			Layers: []kv.LayerSnapshot{{
				Layer:      0,
				CacheIndex: 0,
				Heads: []kv.HeadSnapshot{{
					Key:   []float32{1, 0, 0, 1},
					Value: []float32{0, 1, 1, 0},
				}},
			}},
		},
	}
	session := &Session{session: native}

	snapshot, err := session.CaptureKV()
	if err != nil {
		t.Fatalf("CaptureKV() error = %v", err)
	}
	b, err := mlxbundle.New(snapshot, mlxbundle.Options{
		Model:  "gemma4-e4b",
		Prompt: "stable context",
		Runtime: mlxbundle.Runtime{
			Version: "test",
		},
	})

	if err != nil {
		t.Fatalf("ExportBundle() error = %v", err)
	}
	if b == nil || b.Model.Name != "gemma4-e4b" || b.Runtime.Name != "go-mlx" {
		t.Fatalf("ExportBundle() = %+v", b)
	}
	if b.KV == nil || b.KV.Generated[0] != 2 || b.SAMI == nil {
		t.Fatalf("ExportBundle() KV/SAMI = %+v/%+v", b.KV, b.SAMI)
	}
}

func TestSessionCaptureKV_Bad(t *testing.T) {
	var session *Session

	snapshot, err := session.CaptureKV()

	if err == nil {
		t.Fatal("expected nil session error")
	}
	if snapshot != nil {
		t.Fatalf("snapshot = %v, want nil", snapshot)
	}
}

func TestSessionCaptureKV_Ugly(t *testing.T) {
	wantErr := core.NewError("capture failed")
	session := &Session{session: &sessionfake.Handle{CaptureErr: wantErr}}

	_, err := session.CaptureKV()

	if !core.Is(err, wantErr) {
		t.Fatalf("CaptureKV() error = %v, want %v", err, wantErr)
	}
}

func TestSessionForkResetClose_Good(t *testing.T) {
	forkedNative := &sessionfake.Handle{}
	native := &sessionfake.Handle{Forked: forkedNative}
	session := &Session{session: native}

	forked, err := session.Fork()

	if err != nil {
		t.Fatalf("Fork() error = %v", err)
	}
	if forked == nil || forked.session != forkedNative {
		t.Fatalf("Fork() = %#v, want wrapped fork", forked)
	}
	session.Reset()
	if native.ResetCalls != 1 {
		t.Fatalf("reset calls = %d, want 1", native.ResetCalls)
	}
	if err := session.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if native.CloseCalls != 1 {
		t.Fatalf("close calls = %d, want 1", native.CloseCalls)
	}
}

func TestSessionFork_Bad(t *testing.T) {
	var session *Session

	forked, err := session.Fork()

	if err == nil {
		t.Fatal("expected nil session error")
	}
	if forked != nil {
		t.Fatalf("forked = %v, want nil", forked)
	}
}

func TestSessionClose_Ugly(t *testing.T) {
	wantErr := core.NewError("close failed")
	session := &Session{session: &sessionfake.Handle{CloseErr: wantErr}}

	err := session.Close()

	if !core.Is(err, wantErr) {
		t.Fatalf("Close() error = %v, want %v", err, wantErr)
	}
}

// snapshotRestoreHandle is a deliberately narrow inference.SessionHandle: it
// implements the base interface plus RestoreKV (nativeSessionRestorer) but
// NOT RestoreKVBlocks. The session block-restore paths probe for
// nativeSessionKVBlockRestorer first; with that assertion failing they fall
// back to assembling a CPU-side snapshot and calling RestoreKV — the branch
// the all-capable sessionfake.Handle can never reach.
type snapshotRestoreHandle struct {
	restored *kv.Snapshot
}

func (h *snapshotRestoreHandle) Prefill(context.Context, string) error      { return nil }
func (h *snapshotRestoreHandle) AppendPrompt(context.Context, string) error { return nil }
func (h *snapshotRestoreHandle) Generate(context.Context, inference.GenerateConfig) iter.Seq[inference.Token] {
	return func(func(inference.Token) bool) {}
}
func (h *snapshotRestoreHandle) CaptureKV(context.Context) (*kv.Snapshot, error) {
	return nil, nil
}
func (h *snapshotRestoreHandle) RangeKVBlocks(context.Context, int, kv.CaptureOptions, func(kv.Block) (bool, error)) error {
	return nil
}
func (h *snapshotRestoreHandle) Fork(context.Context) (inference.SessionHandle, error) {
	return nil, nil
}
func (h *snapshotRestoreHandle) Reset()       {}
func (h *snapshotRestoreHandle) Close() error { return nil }
func (h *snapshotRestoreHandle) Err() error   { return nil }
func (h *snapshotRestoreHandle) RestoreKV(_ context.Context, snapshot *kv.Snapshot) error {
	h.restored = snapshot
	return nil
}

// TestSessionLoadKVPrefixBlocks_SnapshotFallback_Good drives the non-native
// block-restore fallback: blocks are written with the all-capable fake, then
// loaded into a session whose handle lacks RestoreKVBlocks, so the prefix is
// assembled into a CPU snapshot and pushed through RestoreKV.
func TestSessionLoadKVPrefixBlocks_SnapshotFallback_Good(t *testing.T) {
	store := memvid.NewInMemoryStore(nil)
	writer := &Session{session: &sessionfake.Handle{
		KVBlocks: []kv.Block{
			{Index: 0, TokenStart: 0, TokenCount: 2, Snapshot: testNativeKVBlock([]int32{10, 20}, 2, []float32{1, 2, 3, 4}, []float32{9, 10, 11, 12}, nil, nil)},
			{Index: 1, TokenStart: 2, TokenCount: 2, Snapshot: testNativeKVBlock([]int32{30, 40}, 4, []float32{5, 6, 7, 8}, []float32{13, 14, 15, 16}, []float32{0.25, 0.75}, []int32{40})},
		},
	}}
	bundle, err := writer.SaveKVBlocksToState(context.Background(), store, kv.StateBlockOptions{BlockSize: 2})
	if err != nil {
		t.Fatalf("SaveKVBlocksToState() error = %v", err)
	}

	native := &snapshotRestoreHandle{}
	reader := &Session{session: native}
	if err := reader.LoadKVPrefixBlocksFromState(context.Background(), store, bundle, 4); err != nil {
		t.Fatalf("LoadKVPrefixBlocksFromState() error = %v", err)
	}
	if native.restored == nil {
		t.Fatal("snapshot fallback did not call RestoreKV")
	}
	// Both blocks cover the 4-token prefix, so the assembled snapshot holds
	// the full token span.
	if got := native.restored.Tokens; len(got) != 4 || got[0] != 10 || got[3] != 40 {
		t.Fatalf("assembled snapshot tokens = %+v, want [10 20 30 40]", got)
	}
}

// TestSessionWakeAgentMemory_SnapshotFallback_Good drives WakeAgentMemory's
// third branch — with the block-restorer capability absent, the wake loads a
// CPU snapshot and restores it, reporting the "snapshot" strategy rather than
// "kv-blocks".
func TestSessionWakeAgentMemory_SnapshotFallback_Good(t *testing.T) {
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	info := spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8}
	source := &Session{session: &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}, info: info}
	sleep, err := source.SleepAgentMemory(ctx, store, agent.SleepOptions{
		EntryURI:     "mlx://agent/snapshot",
		BlockOptions: kv.StateBlockOptions{BlockSize: 1},
	})
	if err != nil {
		t.Fatalf("SleepAgentMemory() error = %v", err)
	}

	native := &snapshotRestoreHandle{}
	awake := &Session{session: native, info: info}
	wake, err := awake.WakeAgentMemory(ctx, store, agent.WakeOptions{
		IndexURI: sleep.IndexURI,
		EntryURI: sleep.EntryURI,
	})
	if err != nil {
		t.Fatalf("WakeAgentMemory() error = %v", err)
	}
	if wake.RestoreStrategy != "snapshot" {
		t.Fatalf("RestoreStrategy = %q, want snapshot", wake.RestoreStrategy)
	}
	if native.restored == nil || len(native.restored.Tokens) != 2 {
		t.Fatalf("restored snapshot = %+v, want two-token state", native.restored)
	}
}

// TestSessionRestoreBundleInMemory_Good covers RestoreBundle's in-memory
// path (b.Snapshot()) — distinct from the State-backed
// RestoreBundleFromState path the memvid bundle test exercises. The bundle
// model identity must match the session ModelInfo or CheckCompatibility
// rejects it.
func TestSessionRestoreBundleInMemory_Good(t *testing.T) {
	native := &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}
	source := &Session{session: native, info: spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1}}

	snapshot, err := source.CaptureKV()
	if err != nil {
		t.Fatalf("CaptureKV() error = %v", err)
	}
	b, err := mlxbundle.New(snapshot, mlxbundle.Options{Model: "gemma4-e4b", Prompt: "stable context"})
	if err != nil {
		t.Fatalf("mlxbundle.New() error = %v", err)
	}

	target := &sessionfake.Handle{}
	session := &Session{session: target, info: spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1}}
	if err := session.RestoreBundle(b); err != nil {
		t.Fatalf("RestoreBundle() error = %v", err)
	}
	if target.RestoredKV == nil || target.RestoredKV.Tokens[0] != 1 {
		t.Fatalf("restored KV = %+v, want two-token state", target.RestoredKV)
	}
}

// TestSessionRestoreBundleInMemory_Ugly — a bundle whose model identity does
// not match the session is rejected by CheckCompatibility before any restore.
func TestSessionRestoreBundleInMemory_Ugly(t *testing.T) {
	native := &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}
	source := &Session{session: native, info: spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1}}
	snapshot, err := source.CaptureKV()
	if err != nil {
		t.Fatalf("CaptureKV() error = %v", err)
	}
	b, err := mlxbundle.New(snapshot, mlxbundle.Options{Model: "gemma4-e4b"})
	if err != nil {
		t.Fatalf("mlxbundle.New() error = %v", err)
	}

	target := &sessionfake.Handle{}
	mismatch := &Session{session: target, info: spine.ModelInfo{Architecture: "llama", NumLayers: 99}}
	if err := mismatch.RestoreBundle(b); err == nil {
		t.Fatal("RestoreBundle() with mismatched model = nil error, want incompatibility")
	}
	if target.RestoredKV != nil {
		t.Fatalf("RestoreBundle() restored despite mismatch = %+v", target.RestoredKV)
	}
}

func testNativeKVBlock(tokens []int32, tokenOffset int, key, value, logits []float32, generated []int32) *kv.Snapshot {
	snapshot := &kv.Snapshot{
		Version:       kv.SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        append([]int32(nil), tokens...),
		Generated:     append([]int32(nil), generated...),
		TokenOffset:   tokenOffset,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        len(tokens),
		HeadDim:       2,
		NumQueryHeads: 1,
		Layers: []kv.LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []kv.HeadSnapshot{{
				Key:   append([]float32(nil), key...),
				Value: append([]float32(nil), value...),
			}},
		}},
	}
	if len(logits) > 0 {
		snapshot.LogitShape = []int32{1, 1, int32(len(logits))}
		snapshot.Logits = append([]float32(nil), logits...)
	}
	return snapshot
}

// sessionTestRootSnapshot mirrors the root tests' stateBundleTestSnapshot
// fixture — the canonical two-token gemma4 root-form KV snapshot.
func sessionTestRootSnapshot() *kv.Snapshot {
	return &kv.Snapshot{
		Version:       kv.SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{1, 2},
		Generated:     []int32{2},
		TokenOffset:   2,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        2,
		HeadDim:       2,
		NumQueryHeads: 8,
		LogitShape:    []int32{1, 1, 3},
		Logits:        []float32{0.1, 0.2, 0.7},
		Layers: []kv.LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []kv.HeadSnapshot{{
				Key:   []float32{1, 0, 0, 1},
				Value: []float32{0, 1, 1, 0},
			}},
		}},
	}
}

// fakeRawTokenizer mirrors the root tokenizer_test fixture: IDToken
// returns the raw token form, DecodeOne the empty string, so the parser
// helper exercises its raw-form fallback branch.
type fakeRawTokenizer struct {
	raw string
}

func (f fakeRawTokenizer) Encode(string) []int32        { return nil }
func (f fakeRawTokenizer) Decode([]int32) string        { return "" }
func (f fakeRawTokenizer) DecodeOne(int32) string       { return "" }
func (f fakeRawTokenizer) TokenID(string) (int32, bool) { return 0, false }
func (f fakeRawTokenizer) IDToken(int32) string         { return f.raw }
func (f fakeRawTokenizer) BOS() int32                   { return 0 }
func (f fakeRawTokenizer) EOS() int32                   { return 0 }
func (f fakeRawTokenizer) HasBOSToken() bool            { return false }
