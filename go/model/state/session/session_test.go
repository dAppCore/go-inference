// SPDX-Licence-Identifier: EUPL-1.2

package session

import (
	"context"
	"iter"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/decode/parser"
	"dappco.re/go/inference/eval/probe"
	"dappco.re/go/inference/kv"
	mlxbundle "dappco.re/go/inference/model/bundle"
	"dappco.re/go/inference/model/spine"
	memvid "dappco.re/go/inference/model/state"
	"dappco.re/go/inference/model/state/agent"
	"dappco.re/go/inference/model/state/session/internal/sessionfake"
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

func TestSession_PrefillChunks_Good(t *testing.T) {
	nativeSession := &sessionfake.Handle{}
	session := &Session{session: nativeSession}

	if err := session.PrefillChunks(context.Background(), seqStrings("stable ", "context")); err != nil {
		t.Fatalf("PrefillChunks() error = %v", err)
	}

	if got := core.Join("", nativeSession.PrefillChunksSeen...); got != "stable context" {
		t.Fatalf("prefill chunks = %#v, joined %q", nativeSession.PrefillChunksSeen, got)
	}
}

func TestSession_PrefillTokens_Good(t *testing.T) {
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

func TestSession_AppendPrompt_Good(t *testing.T) {
	nativeSession := &sessionfake.Handle{}
	session := &Session{session: nativeSession}

	if err := session.AppendPrompt("\n\nQuestion: who?\nAnswer:"); err != nil {
		t.Fatalf("AppendPrompt() error = %v", err)
	}

	if nativeSession.AppendPromptSeen != "\n\nQuestion: who?\nAnswer:" {
		t.Fatalf("append prompt = %q", nativeSession.AppendPromptSeen)
	}
}

func TestSession_AppendTokens_Good(t *testing.T) {
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

func TestSession_AppendPromptChunks_Good(t *testing.T) {
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

func TestSession_Generate_ForwardsProbeSink_Good(t *testing.T) {
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

func TestSession_Prefill_Bad(t *testing.T) {
	var session *Session

	if err := session.Prefill("prompt"); err == nil {
		t.Fatal("expected nil session error")
	}
}

func TestSession_Generate_Ugly(t *testing.T) {
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

func TestSession_GenerateStream_Good(t *testing.T) {
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

func TestSession_GenerateStream_HideGemma4Thinking_Good(t *testing.T) {
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

func TestSession_GenerateStream_Bad(t *testing.T) {
	var session *Session

	ch := session.GenerateStream(context.Background())

	if tok, ok := <-ch; ok {
		t.Fatalf("stream yielded %+v, want closed", tok)
	}
}

func TestSession_GenerateStream_Ugly(t *testing.T) {
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

func TestSession_CaptureKV_Bad(t *testing.T) {
	var session *Session

	snapshot, err := session.CaptureKV()

	if err == nil {
		t.Fatal("expected nil session error")
	}
	if snapshot != nil {
		t.Fatalf("snapshot = %v, want nil", snapshot)
	}
}

func TestSession_CaptureKV_Ugly(t *testing.T) {
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

func TestSession_Fork_Bad(t *testing.T) {
	var session *Session

	forked, err := session.Fork()

	if err == nil {
		t.Fatal("expected nil session error")
	}
	if forked != nil {
		t.Fatalf("forked = %v, want nil", forked)
	}
}

func TestSession_Close_Ugly(t *testing.T) {
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

// TestSession_LoadKVPrefixBlocksFromState_SnapshotFallback_Good drives the non-native
// block-restore fallback: blocks are written with the all-capable fake, then
// loaded into a session whose handle lacks RestoreKVBlocks, so the prefix is
// assembled into a CPU snapshot and pushed through RestoreKV.
func TestSession_LoadKVPrefixBlocksFromState_SnapshotFallback_Good(t *testing.T) {
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

// TestSession_WakeAgentMemory_SnapshotFallback_Good drives WakeAgentMemory's
// third branch — with the block-restorer capability absent, the wake loads a
// CPU snapshot and restores it, reporting the "snapshot" strategy rather than
// "kv-blocks".
func TestSession_WakeAgentMemory_SnapshotFallback_Good(t *testing.T) {
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

// ---------------------------------------------------------------------------
// House test-standard triplets — one clean Test<File>_<Symbol>_{Good,Bad,Ugly}
// per session.go public symbol, added alongside the richer scenario-named
// tests above (which stay in place; this block fills the exact-name gap the
// ax7-gaps audit checks for).
// ---------------------------------------------------------------------------

// TestSession_New_Good wraps a handle, model info, and tokenizer — the
// fields land untouched, ready for the wrapper methods to use.
func TestSession_New_Good(t *testing.T) {
	handle := &sessionfake.Handle{}
	tok := spine.NewTokenizer(nil)
	info := spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 4}

	sess := New(handle, info, tok)

	if sess.session != handle {
		t.Fatalf("New() session = %v, want handle", sess.session)
	}
	if sess.info.Architecture != "gemma4_text" || sess.info.NumLayers != 4 {
		t.Fatalf("New() info = %+v, want preserved fields", sess.info)
	}
	if sess.tok != tok {
		t.Fatalf("New() tok = %v, want %v", sess.tok, tok)
	}
}

// TestSession_New_Bad — a nil handle is accepted (no validation at
// construction) but leaves the session in the "no live handle" state Valid()
// reports on.
func TestSession_New_Bad(t *testing.T) {
	sess := New(nil, spine.ModelInfo{}, nil)

	if sess == nil {
		t.Fatal("New(nil handle) = nil, want a non-nil Session")
	}
	if sess.Valid() {
		t.Fatal("New(nil handle).Valid() = true, want false")
	}
}

// TestSession_New_Ugly — a zero-value ModelInfo and nil tokenizer still
// produce a session usable for handle-only operations, a legitimate
// incremental-construction shape callers rely on.
func TestSession_New_Ugly(t *testing.T) {
	handle := &sessionfake.Handle{}
	sess := New(handle, spine.ModelInfo{}, nil)

	if !sess.Valid() {
		t.Fatal("New(zero info, nil tok).Valid() = false, want true")
	}
	if sess.Native() != handle {
		t.Fatalf("Native() = %v, want %v", sess.Native(), handle)
	}
}

// TestSession_Prefill_Good forwards the prompt to the native handle verbatim.
func TestSession_Prefill_Good(t *testing.T) {
	native := &sessionfake.Handle{}
	session := &Session{session: native}

	if err := session.Prefill("stable context"); err != nil {
		t.Fatalf("Prefill() error = %v", err)
	}
	if native.PrefillPrompt != "stable context" {
		t.Fatalf("PrefillPrompt = %q, want %q", native.PrefillPrompt, "stable context")
	}
}

// TestSession_Prefill_Ugly — the native handle's error is propagated
// verbatim (distinct from the nil-session guard the Bad case exercises).
func TestSession_Prefill_Ugly(t *testing.T) {
	wantErr := core.NewError("native prefill failed")
	session := &Session{session: &sessionfake.Handle{PrefillErr: wantErr}}

	if err := session.Prefill("prompt"); !core.Is(err, wantErr) {
		t.Fatalf("Prefill() error = %v, want %v", err, wantErr)
	}
}

// TestSession_PrefillChunks_Bad — a nil session returns the sentinel before
// any capability probe.
func TestSession_PrefillChunks_Bad(t *testing.T) {
	var session *Session

	if err := session.PrefillChunks(context.Background(), seqStrings("a")); !core.Is(err, errModelSessionNil) {
		t.Fatalf("PrefillChunks(nil session) error = %v, want %v", err, errModelSessionNil)
	}
}

// TestSession_PrefillChunks_Ugly — the native chunk-prefiller's error is
// propagated verbatim.
func TestSession_PrefillChunks_Ugly(t *testing.T) {
	wantErr := core.NewError("native chunk prefill failed")
	session := &Session{session: &sessionfake.Handle{PrefillErr: wantErr}}

	if err := session.PrefillChunks(context.Background(), seqStrings("a", "b")); !core.Is(err, wantErr) {
		t.Fatalf("PrefillChunks() error = %v, want %v", err, wantErr)
	}
}

// TestSession_PrefillTokens_Bad — a nil session returns the sentinel before
// any capability probe.
func TestSession_PrefillTokens_Bad(t *testing.T) {
	var session *Session

	if err := session.PrefillTokens(context.Background(), []int32{1}); !core.Is(err, errModelSessionNil) {
		t.Fatalf("PrefillTokens(nil session) error = %v, want %v", err, errModelSessionNil)
	}
}

// TestSession_PrefillTokens_Ugly — the native token-prefiller's error is
// propagated verbatim (distinct from the no-native-support Bad case in the
// scenario tests above).
func TestSession_PrefillTokens_Ugly(t *testing.T) {
	wantErr := core.NewError("native token prefill failed")
	session := &Session{session: &sessionfake.Handle{PrefillErr: wantErr}}

	if err := session.PrefillTokens(context.Background(), []int32{1, 2}); !core.Is(err, wantErr) {
		t.Fatalf("PrefillTokens() error = %v, want %v", err, wantErr)
	}
}

// TestSession_AppendPrompt_Bad — a nil session returns the sentinel.
func TestSession_AppendPrompt_Bad(t *testing.T) {
	var session *Session

	if err := session.AppendPrompt("more"); !core.Is(err, errModelSessionNil) {
		t.Fatalf("AppendPrompt(nil session) error = %v, want %v", err, errModelSessionNil)
	}
}

// TestSession_AppendPrompt_Ugly — the native handle's error is propagated.
func TestSession_AppendPrompt_Ugly(t *testing.T) {
	wantErr := core.NewError("native append failed")
	session := &Session{session: &sessionfake.Handle{AppendErr: wantErr}}

	if err := session.AppendPrompt("more"); !core.Is(err, wantErr) {
		t.Fatalf("AppendPrompt() error = %v, want %v", err, wantErr)
	}
}

// TestSession_AppendPromptChunks_Bad — a nil session returns the sentinel.
func TestSession_AppendPromptChunks_Bad(t *testing.T) {
	var session *Session

	if err := session.AppendPromptChunks(context.Background(), seqStrings("a")); !core.Is(err, errModelSessionNil) {
		t.Fatalf("AppendPromptChunks(nil session) error = %v, want %v", err, errModelSessionNil)
	}
}

// TestSession_AppendPromptChunks_Ugly — the native chunk-appender's error is
// propagated verbatim.
func TestSession_AppendPromptChunks_Ugly(t *testing.T) {
	wantErr := core.NewError("native chunk append failed")
	session := &Session{session: &sessionfake.Handle{AppendErr: wantErr}}

	if err := session.AppendPromptChunks(context.Background(), seqStrings("a", "b")); !core.Is(err, wantErr) {
		t.Fatalf("AppendPromptChunks() error = %v, want %v", err, wantErr)
	}
}

// TestSession_AppendTokens_Bad — a nil session returns the sentinel.
func TestSession_AppendTokens_Bad(t *testing.T) {
	var session *Session

	if err := session.AppendTokens(context.Background(), []int32{1}); !core.Is(err, errModelSessionNil) {
		t.Fatalf("AppendTokens(nil session) error = %v, want %v", err, errModelSessionNil)
	}
}

// TestSession_AppendTokens_Ugly — the native token-appender's error is
// propagated verbatim.
func TestSession_AppendTokens_Ugly(t *testing.T) {
	wantErr := core.NewError("native token append failed")
	session := &Session{session: &sessionfake.Handle{AppendErr: wantErr}}

	if err := session.AppendTokens(context.Background(), []int32{1, 2}); !core.Is(err, wantErr) {
		t.Fatalf("AppendTokens() error = %v, want %v", err, wantErr)
	}
}

// TestSession_Generate_Good drains the native token stream into a buffered
// string.
func TestSession_Generate_Good(t *testing.T) {
	session := &Session{session: &sessionfake.Handle{
		Tokens: []inference.Token{{ID: 1, Text: "Hi"}, {ID: 2, Text: " there"}},
	}}

	reply, err := session.Generate()

	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if reply != "Hi there" {
		t.Fatalf("Generate() = %q, want %q", reply, "Hi there")
	}
}

// TestSession_Generate_Bad — a nil session returns the sentinel before any
// generation is attempted.
func TestSession_Generate_Bad(t *testing.T) {
	var session *Session

	if _, err := session.Generate(); !core.Is(err, errModelSessionNil) {
		t.Fatalf("Generate(nil session) error = %v, want %v", err, errModelSessionNil)
	}
}

// TestSession_CaptureKV_Good forwards to CaptureKVWithOptions with the zero
// options value and returns an owned (cloned) snapshot.
func TestSession_CaptureKV_Good(t *testing.T) {
	seed := sessionfake.TestKVSnapshot()
	session := &Session{session: &sessionfake.Handle{KV: seed}}

	snapshot, err := session.CaptureKV()

	if err != nil {
		t.Fatalf("CaptureKV() error = %v", err)
	}
	if snapshot == seed {
		t.Fatal("CaptureKV() returned the native snapshot by reference, want a clone")
	}
	if snapshot.Architecture != "gemma4_text" || len(snapshot.Tokens) != 2 {
		t.Fatalf("CaptureKV() = %+v, want the seeded two-token snapshot", snapshot)
	}
}

// TestSession_CaptureKVWithOptions_Good — a handle without the with-options
// capability falls back to plain CaptureKV.
func TestSession_CaptureKVWithOptions_Good(t *testing.T) {
	session := &Session{session: &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}}

	snapshot, err := session.CaptureKVWithOptions(kv.CaptureOptions{})

	if err != nil {
		t.Fatalf("CaptureKVWithOptions() error = %v", err)
	}
	if snapshot == nil || snapshot.Architecture != "gemma4_text" {
		t.Fatalf("CaptureKVWithOptions() = %+v, want the seeded snapshot", snapshot)
	}
}

// TestSession_CaptureKVWithOptions_Bad — a nil session returns the sentinel
// before any capability probe.
func TestSession_CaptureKVWithOptions_Bad(t *testing.T) {
	var session *Session

	if _, err := session.CaptureKVWithOptions(kv.CaptureOptions{}); !core.Is(err, errModelSessionNil) {
		t.Fatalf("CaptureKVWithOptions(nil session) error = %v, want %v", err, errModelSessionNil)
	}
}

// TestSession_CaptureKVWithOptions_Ugly — the fallback plain-CaptureKV error
// is propagated verbatim.
func TestSession_CaptureKVWithOptions_Ugly(t *testing.T) {
	wantErr := core.NewError("native capture failed")
	session := &Session{session: &sessionfake.Handle{CaptureErr: wantErr}}

	if _, err := session.CaptureKVWithOptions(kv.CaptureOptions{}); !core.Is(err, wantErr) {
		t.Fatalf("CaptureKVWithOptions() error = %v, want %v", err, wantErr)
	}
}

// TestSession_AnalyzeKV_Good captures and analyses the retained KV state.
func TestSession_AnalyzeKV_Good(t *testing.T) {
	session := &Session{session: &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}}

	analysis, err := session.AnalyzeKV()

	if err != nil {
		t.Fatalf("AnalyzeKV() error = %v", err)
	}
	if analysis == nil {
		t.Fatal("AnalyzeKV() = nil, want an analysis of the captured snapshot")
	}
}

// TestSession_AnalyzeKV_Bad — a nil session surfaces CaptureKV's sentinel
// before any analysis is attempted.
func TestSession_AnalyzeKV_Bad(t *testing.T) {
	var session *Session

	if _, err := session.AnalyzeKV(); !core.Is(err, errModelSessionNil) {
		t.Fatalf("AnalyzeKV(nil session) error = %v, want %v", err, errModelSessionNil)
	}
}

// TestSession_AnalyzeKV_Ugly — a live handle whose capture fails surfaces the
// capture error, distinct from the nil-session Bad case.
func TestSession_AnalyzeKV_Ugly(t *testing.T) {
	wantErr := core.NewError("analyze capture failed")
	session := &Session{session: &sessionfake.Handle{CaptureErr: wantErr}}

	if _, err := session.AnalyzeKV(); !core.Is(err, wantErr) {
		t.Fatalf("AnalyzeKV() error = %v, want %v", err, wantErr)
	}
}

// TestSession_SaveKV_Good captures and writes the retained KV state to disk.
func TestSession_SaveKV_Good(t *testing.T) {
	session := &Session{session: &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}}
	path := core.PathJoin(t.TempDir(), "save.kvbin")

	if err := session.SaveKV(path); err != nil {
		t.Fatalf("SaveKV() error = %v", err)
	}
	loaded, err := kv.Load(path)
	if err != nil {
		t.Fatalf("kv.Load() error = %v", err)
	}
	if loaded.Architecture != "gemma4_text" {
		t.Fatalf("saved snapshot = %+v, want gemma4_text", loaded)
	}
}

// TestSession_SaveKV_Bad — a nil session surfaces CaptureKV's sentinel
// before any file write.
func TestSession_SaveKV_Bad(t *testing.T) {
	var session *Session

	if err := session.SaveKV(core.PathJoin(t.TempDir(), "x.kvbin")); !core.Is(err, errModelSessionNil) {
		t.Fatalf("SaveKV(nil session) error = %v, want %v", err, errModelSessionNil)
	}
}

// TestSession_SaveKV_Ugly — a live handle whose capture fails surfaces the
// capture error before any file write.
func TestSession_SaveKV_Ugly(t *testing.T) {
	wantErr := core.NewError("save capture failed")
	session := &Session{session: &sessionfake.Handle{CaptureErr: wantErr}}

	if err := session.SaveKV(core.PathJoin(t.TempDir(), "x.kvbin")); !core.Is(err, wantErr) {
		t.Fatalf("SaveKV() error = %v, want %v", err, wantErr)
	}
}

// TestSession_RestoreKV_Good replaces the retained state and clears any
// stale agent-memory cache.
func TestSession_RestoreKV_Good(t *testing.T) {
	native := &sessionfake.Handle{}
	session := &Session{session: native, agentMemory: &agent.WakeReport{EntryURI: "stale"}}
	snapshot := sessionfake.TestKVSnapshot()

	if err := session.RestoreKV(snapshot); err != nil {
		t.Fatalf("RestoreKV() error = %v", err)
	}
	if native.RestoredKV != snapshot {
		t.Fatalf("RestoredKV = %p, want %p", native.RestoredKV, snapshot)
	}
	if session.agentMemory != nil {
		t.Fatal("RestoreKV() left a stale agentMemory cache")
	}
}

// TestSession_RestoreKV_Bad — a nil snapshot returns the sentinel before any
// native restore.
func TestSession_RestoreKV_Bad(t *testing.T) {
	session := &Session{session: &sessionfake.Handle{}}

	if err := session.RestoreKV(nil); !core.Is(err, errKVSnapshotNil) {
		t.Fatalf("RestoreKV(nil snapshot) error = %v, want %v", err, errKVSnapshotNil)
	}
}

// TestSession_RestoreKV_Ugly — a nil session returns the sentinel before the
// snapshot is even inspected.
func TestSession_RestoreKV_Ugly(t *testing.T) {
	var session *Session

	if err := session.RestoreKV(sessionfake.TestKVSnapshot()); !core.Is(err, errModelSessionNil) {
		t.Fatalf("RestoreKV(nil session) error = %v, want %v", err, errModelSessionNil)
	}
}

// TestSession_LoadKV_Good round-trips a snapshot through disk and back into
// the retained session state.
func TestSession_LoadKV_Good(t *testing.T) {
	native := &sessionfake.Handle{}
	session := &Session{session: native}
	path := core.PathJoin(t.TempDir(), "load.kvbin")
	if err := sessionfake.TestKVSnapshot().Save(path); err != nil {
		t.Fatalf("Save() error = %v", err)
	}

	if err := session.LoadKV(path); err != nil {
		t.Fatalf("LoadKV() error = %v", err)
	}
	if native.RestoredKV == nil || len(native.RestoredKV.Tokens) != 2 {
		t.Fatalf("restored KV = %+v, want two-token state", native.RestoredKV)
	}
}

// TestSession_LoadKV_Bad — a missing file surfaces the load failure before
// any restore is attempted.
func TestSession_LoadKV_Bad(t *testing.T) {
	session := &Session{session: &sessionfake.Handle{}}

	if err := session.LoadKV(core.PathJoin(t.TempDir(), "missing.kvbin")); err == nil {
		t.Fatal("LoadKV(missing) error = nil, want load failure")
	}
}

// TestSession_LoadKV_Ugly — the file loads fine but the native handle cannot
// restore, surfacing RestoreKV's error through the nested call.
func TestSession_LoadKV_Ugly(t *testing.T) {
	session := &Session{session: &sessionfake.Handle{}}
	path := core.PathJoin(t.TempDir(), "load.kvbin")
	if err := sessionfake.TestKVSnapshot().Save(path); err != nil {
		t.Fatalf("Save() error = %v", err)
	}
	wantErr := core.NewError("native restore failed")
	session.session = &sessionfake.Handle{RestoreErr: wantErr}

	if err := session.LoadKV(path); !core.Is(err, wantErr) {
		t.Fatalf("LoadKV() error = %v, want %v", err, wantErr)
	}
}

// TestSession_SaveKVToState_Good captures and writes the retained KV state
// as a State chunk.
func TestSession_SaveKVToState_Good(t *testing.T) {
	store := memvid.NewInMemoryStore(nil)
	session := &Session{session: &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}}

	ref, err := session.SaveKVToState(context.Background(), store, kv.StateOptions{URI: "mlx://session/state"})

	if err != nil {
		t.Fatalf("SaveKVToState() error = %v", err)
	}
	if ref.ChunkID == 0 {
		t.Fatalf("SaveKVToState() ref = %+v, want a stored chunk", ref)
	}
}

// TestSession_SaveKVToState_Bad — a nil session surfaces CaptureKV's
// sentinel before any store write.
func TestSession_SaveKVToState_Bad(t *testing.T) {
	var session *Session
	store := memvid.NewInMemoryStore(nil)

	if _, err := session.SaveKVToState(context.Background(), store, kv.StateOptions{}); !core.Is(err, errModelSessionNil) {
		t.Fatalf("SaveKVToState(nil session) error = %v, want %v", err, errModelSessionNil)
	}
}

// TestSession_SaveKVToState_Ugly — a live handle whose capture fails
// surfaces the capture error before any store write.
func TestSession_SaveKVToState_Ugly(t *testing.T) {
	wantErr := core.NewError("save-to-state capture failed")
	session := &Session{session: &sessionfake.Handle{CaptureErr: wantErr}}
	store := memvid.NewInMemoryStore(nil)

	if _, err := session.SaveKVToState(context.Background(), store, kv.StateOptions{}); !core.Is(err, wantErr) {
		t.Fatalf("SaveKVToState() error = %v, want %v", err, wantErr)
	}
}

// TestSession_SaveKVToMemvid_Good — the deprecated alias forwards to
// SaveKVToState unchanged (kv.MemvidOptions is a type alias for
// kv.StateOptions, so the same literal serves both calls).
func TestSession_SaveKVToMemvid_Good(t *testing.T) {
	store := memvid.NewInMemoryStore(nil)
	session := &Session{session: &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}}

	ref, err := session.SaveKVToMemvid(context.Background(), store, kv.MemvidOptions{URI: "mlx://session/memvid"})

	if err != nil {
		t.Fatalf("SaveKVToMemvid() error = %v", err)
	}
	if ref.ChunkID == 0 {
		t.Fatalf("SaveKVToMemvid() ref = %+v, want a stored chunk", ref)
	}
}

// TestSession_SaveKVToMemvid_Bad — a nil session surfaces the same sentinel
// as the SaveKVToState path it forwards to.
func TestSession_SaveKVToMemvid_Bad(t *testing.T) {
	var session *Session
	store := memvid.NewInMemoryStore(nil)

	if _, err := session.SaveKVToMemvid(context.Background(), store, kv.MemvidOptions{}); !core.Is(err, errModelSessionNil) {
		t.Fatalf("SaveKVToMemvid(nil session) error = %v, want %v", err, errModelSessionNil)
	}
}

// TestSession_SaveKVToMemvid_Ugly — a live handle whose capture fails
// surfaces the capture error through the forwarded call.
func TestSession_SaveKVToMemvid_Ugly(t *testing.T) {
	wantErr := core.NewError("save-to-memvid capture failed")
	session := &Session{session: &sessionfake.Handle{CaptureErr: wantErr}}
	store := memvid.NewInMemoryStore(nil)

	if _, err := session.SaveKVToMemvid(context.Background(), store, kv.MemvidOptions{}); !core.Is(err, wantErr) {
		t.Fatalf("SaveKVToMemvid() error = %v, want %v", err, wantErr)
	}
}

// TestSession_LoadKVFromState_Good round-trips a KV state chunk back into
// the retained session state.
func TestSession_LoadKVFromState_Good(t *testing.T) {
	store := memvid.NewInMemoryStore(nil)
	writer := &Session{session: &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}}
	ref, err := writer.SaveKVToState(context.Background(), store, kv.StateOptions{URI: "mlx://session/roundtrip"})
	if err != nil {
		t.Fatalf("SaveKVToState() error = %v", err)
	}

	reader := &sessionfake.Handle{}
	session := &Session{session: reader}
	if err := session.LoadKVFromState(context.Background(), store, ref); err != nil {
		t.Fatalf("LoadKVFromState() error = %v", err)
	}
	if reader.RestoredKV == nil || len(reader.RestoredKV.Tokens) != 2 {
		t.Fatalf("restored KV = %+v, want two-token state", reader.RestoredKV)
	}
}

// TestSession_LoadKVFromState_Bad — a bogus chunk ref surfaces the load
// failure before any restore.
func TestSession_LoadKVFromState_Bad(t *testing.T) {
	store := memvid.NewInMemoryStore(nil)
	session := &Session{session: &sessionfake.Handle{}}

	if err := session.LoadKVFromState(context.Background(), store, memvid.ChunkRef{ChunkID: 999}); err == nil {
		t.Fatal("LoadKVFromState(bad ref) error = nil, want load failure")
	}
}

// TestSession_LoadKVFromState_Ugly — a nil *Session forwards through the
// load to RestoreKV's guard, surfacing the sentinel rather than panicking.
func TestSession_LoadKVFromState_Ugly(t *testing.T) {
	store := memvid.NewInMemoryStore(nil)
	writer := &Session{session: &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}}
	ref, err := writer.SaveKVToState(context.Background(), store, kv.StateOptions{URI: "mlx://session/nilrecv"})
	if err != nil {
		t.Fatalf("SaveKVToState() error = %v", err)
	}

	var session *Session
	if err := session.LoadKVFromState(context.Background(), store, ref); !core.Is(err, errModelSessionNil) {
		t.Fatalf("LoadKVFromState(nil session) error = %v, want %v", err, errModelSessionNil)
	}
}

// TestSession_LoadKVFromMemvid_Good — the deprecated alias forwards to
// LoadKVFromState unchanged.
func TestSession_LoadKVFromMemvid_Good(t *testing.T) {
	store := memvid.NewInMemoryStore(nil)
	writer := &Session{session: &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}}
	ref, err := writer.SaveKVToState(context.Background(), store, kv.StateOptions{URI: "mlx://session/memvid-roundtrip"})
	if err != nil {
		t.Fatalf("SaveKVToState() error = %v", err)
	}

	reader := &sessionfake.Handle{}
	session := &Session{session: reader}
	if err := session.LoadKVFromMemvid(context.Background(), store, ref); err != nil {
		t.Fatalf("LoadKVFromMemvid() error = %v", err)
	}
	if reader.RestoredKV == nil || len(reader.RestoredKV.Tokens) != 2 {
		t.Fatalf("restored KV = %+v, want two-token state", reader.RestoredKV)
	}
}

// TestSession_LoadKVFromMemvid_Bad — a bogus chunk ref surfaces the load
// failure, same as the LoadKVFromState path it forwards to.
func TestSession_LoadKVFromMemvid_Bad(t *testing.T) {
	store := memvid.NewInMemoryStore(nil)
	session := &Session{session: &sessionfake.Handle{}}

	if err := session.LoadKVFromMemvid(context.Background(), store, memvid.ChunkRef{ChunkID: 999}); err == nil {
		t.Fatal("LoadKVFromMemvid(bad ref) error = nil, want load failure")
	}
}

// TestSession_LoadKVFromMemvid_Ugly — a nil *Session surfaces
// errModelSessionNil through the forwarded RestoreKV guard.
func TestSession_LoadKVFromMemvid_Ugly(t *testing.T) {
	store := memvid.NewInMemoryStore(nil)
	writer := &Session{session: &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}}
	ref, err := writer.SaveKVToState(context.Background(), store, kv.StateOptions{URI: "mlx://session/memvid-nilrecv"})
	if err != nil {
		t.Fatalf("SaveKVToState() error = %v", err)
	}

	var session *Session
	if err := session.LoadKVFromMemvid(context.Background(), store, ref); !core.Is(err, errModelSessionNil) {
		t.Fatalf("LoadKVFromMemvid(nil session) error = %v, want %v", err, errModelSessionNil)
	}
}

// TestSession_SaveKVBlocksToState_Good streams the native KV blocks into
// per-block State chunks.
func TestSession_SaveKVBlocksToState_Good(t *testing.T) {
	store := memvid.NewInMemoryStore(nil)
	session := &Session{session: &sessionfake.Handle{
		KVBlocks: []kv.Block{
			{Index: 0, TokenStart: 0, TokenCount: 2, Snapshot: testNativeKVBlock([]int32{10, 20}, 2, []float32{1, 2, 3, 4}, []float32{9, 10, 11, 12}, nil, nil)},
		},
	}}

	bundle, err := session.SaveKVBlocksToState(context.Background(), store, kv.StateBlockOptions{BlockSize: 2})

	if err != nil {
		t.Fatalf("SaveKVBlocksToState() error = %v", err)
	}
	if bundle == nil || bundle.BlockSize != 2 || len(bundle.Blocks) != 1 {
		t.Fatalf("SaveKVBlocksToState() bundle = %+v, want one 2-token block", bundle)
	}
}

// TestSession_SaveKVBlocksToState_Bad — a nil session returns the sentinel
// before any block stream is attempted.
func TestSession_SaveKVBlocksToState_Bad(t *testing.T) {
	var session *Session
	store := memvid.NewInMemoryStore(nil)

	if _, err := session.SaveKVBlocksToState(context.Background(), store, kv.StateBlockOptions{}); !core.Is(err, errModelSessionNil) {
		t.Fatalf("SaveKVBlocksToState(nil session) error = %v, want %v", err, errModelSessionNil)
	}
}

// TestSession_SaveKVBlocksToState_Ugly — the native RangeKVBlocks error is
// propagated verbatim.
func TestSession_SaveKVBlocksToState_Ugly(t *testing.T) {
	wantErr := core.NewError("range blocks failed")
	session := &Session{session: &rangeErrHandle{err: wantErr}}
	store := memvid.NewInMemoryStore(nil)

	if _, err := session.SaveKVBlocksToState(context.Background(), store, kv.StateBlockOptions{}); !core.Is(err, wantErr) {
		t.Fatalf("SaveKVBlocksToState() error = %v, want %v", err, wantErr)
	}
}

// TestSession_SaveKVBlocksToMemvid_Good — the deprecated alias forwards to
// SaveKVBlocksToState unchanged (kv.MemvidBlockOptions aliases
// kv.StateBlockOptions).
func TestSession_SaveKVBlocksToMemvid_Good(t *testing.T) {
	store := memvid.NewInMemoryStore(nil)
	session := &Session{session: &sessionfake.Handle{
		KVBlocks: []kv.Block{
			{Index: 0, TokenStart: 0, TokenCount: 2, Snapshot: testNativeKVBlock([]int32{10, 20}, 2, []float32{1, 2, 3, 4}, []float32{9, 10, 11, 12}, nil, nil)},
		},
	}}

	bundle, err := session.SaveKVBlocksToMemvid(context.Background(), store, kv.MemvidBlockOptions{BlockSize: 2})

	if err != nil {
		t.Fatalf("SaveKVBlocksToMemvid() error = %v", err)
	}
	if bundle == nil || bundle.BlockSize != 2 || len(bundle.Blocks) != 1 {
		t.Fatalf("SaveKVBlocksToMemvid() bundle = %+v, want one 2-token block", bundle)
	}
}

// TestSession_SaveKVBlocksToMemvid_Bad — a nil session returns the sentinel,
// same as the SaveKVBlocksToState path it forwards to.
func TestSession_SaveKVBlocksToMemvid_Bad(t *testing.T) {
	var session *Session
	store := memvid.NewInMemoryStore(nil)

	if _, err := session.SaveKVBlocksToMemvid(context.Background(), store, kv.MemvidBlockOptions{}); !core.Is(err, errModelSessionNil) {
		t.Fatalf("SaveKVBlocksToMemvid(nil session) error = %v, want %v", err, errModelSessionNil)
	}
}

// TestSession_SaveKVBlocksToMemvid_Ugly — the native RangeKVBlocks error is
// propagated through the forwarded call.
func TestSession_SaveKVBlocksToMemvid_Ugly(t *testing.T) {
	wantErr := core.NewError("range blocks failed")
	session := &Session{session: &rangeErrHandle{err: wantErr}}
	store := memvid.NewInMemoryStore(nil)

	if _, err := session.SaveKVBlocksToMemvid(context.Background(), store, kv.MemvidBlockOptions{}); !core.Is(err, wantErr) {
		t.Fatalf("SaveKVBlocksToMemvid() error = %v, want %v", err, wantErr)
	}
}

// TestSession_LoadKVBlocksFromState_Good delegates to
// LoadKVPrefixBlocksFromState with prefixTokens 0, restoring every block.
func TestSession_LoadKVBlocksFromState_Good(t *testing.T) {
	store := memvid.NewInMemoryStore(nil)
	writer := &Session{session: &sessionfake.Handle{
		KVBlocks: []kv.Block{
			{Index: 0, TokenStart: 0, TokenCount: 2, Snapshot: testNativeKVBlock([]int32{10, 20}, 2, []float32{1, 2, 3, 4}, []float32{9, 10, 11, 12}, nil, nil)},
		},
	}}
	bundle, err := writer.SaveKVBlocksToState(context.Background(), store, kv.StateBlockOptions{BlockSize: 2})
	if err != nil {
		t.Fatalf("SaveKVBlocksToState() error = %v", err)
	}

	reader := &sessionfake.Handle{}
	session := &Session{session: reader}
	if err := session.LoadKVBlocksFromState(context.Background(), store, bundle); err != nil {
		t.Fatalf("LoadKVBlocksFromState() error = %v", err)
	}
	if len(reader.RestoredBlocks) != 1 {
		t.Fatalf("RestoredBlocks = %v, want 1 block", reader.RestoredBlocks)
	}
}

// TestSession_LoadKVBlocksFromState_Bad — a nil bundle surfaces the
// block-bundle-nil sentinel.
func TestSession_LoadKVBlocksFromState_Bad(t *testing.T) {
	session := &Session{session: &sessionfake.Handle{}}
	store := memvid.NewInMemoryStore(nil)

	if err := session.LoadKVBlocksFromState(context.Background(), store, nil); !core.Is(err, errStateKVBlockBundleNil) {
		t.Fatalf("LoadKVBlocksFromState(nil bundle) error = %v, want %v", err, errStateKVBlockBundleNil)
	}
}

// TestSession_LoadKVBlocksFromState_Ugly — a nil session returns the
// sentinel before any bundle is inspected.
func TestSession_LoadKVBlocksFromState_Ugly(t *testing.T) {
	var session *Session
	store := memvid.NewInMemoryStore(nil)

	if err := session.LoadKVBlocksFromState(context.Background(), store, &kv.StateBlockBundle{}); !core.Is(err, errModelSessionNil) {
		t.Fatalf("LoadKVBlocksFromState(nil session) error = %v, want %v", err, errModelSessionNil)
	}
}

// TestSession_LoadKVBlocksFromMemvid_Good — the deprecated alias forwards to
// LoadKVBlocksFromState unchanged (kv.MemvidBlockBundle aliases
// kv.StateBlockBundle).
func TestSession_LoadKVBlocksFromMemvid_Good(t *testing.T) {
	store := memvid.NewInMemoryStore(nil)
	writer := &Session{session: &sessionfake.Handle{
		KVBlocks: []kv.Block{
			{Index: 0, TokenStart: 0, TokenCount: 2, Snapshot: testNativeKVBlock([]int32{10, 20}, 2, []float32{1, 2, 3, 4}, []float32{9, 10, 11, 12}, nil, nil)},
		},
	}}
	bundle, err := writer.SaveKVBlocksToState(context.Background(), store, kv.StateBlockOptions{BlockSize: 2})
	if err != nil {
		t.Fatalf("SaveKVBlocksToState() error = %v", err)
	}

	reader := &sessionfake.Handle{}
	session := &Session{session: reader}
	if err := session.LoadKVBlocksFromMemvid(context.Background(), store, bundle); err != nil {
		t.Fatalf("LoadKVBlocksFromMemvid() error = %v", err)
	}
	if len(reader.RestoredBlocks) != 1 {
		t.Fatalf("RestoredBlocks = %v, want 1 block", reader.RestoredBlocks)
	}
}

// TestSession_LoadKVBlocksFromMemvid_Bad — a nil bundle surfaces the same
// sentinel as the LoadKVBlocksFromState path it forwards to.
func TestSession_LoadKVBlocksFromMemvid_Bad(t *testing.T) {
	session := &Session{session: &sessionfake.Handle{}}
	store := memvid.NewInMemoryStore(nil)

	if err := session.LoadKVBlocksFromMemvid(context.Background(), store, nil); !core.Is(err, errStateKVBlockBundleNil) {
		t.Fatalf("LoadKVBlocksFromMemvid(nil bundle) error = %v, want %v", err, errStateKVBlockBundleNil)
	}
}

// TestSession_LoadKVBlocksFromMemvid_Ugly — a nil session returns the
// sentinel through the forwarded call.
func TestSession_LoadKVBlocksFromMemvid_Ugly(t *testing.T) {
	var session *Session
	store := memvid.NewInMemoryStore(nil)

	if err := session.LoadKVBlocksFromMemvid(context.Background(), store, &kv.MemvidBlockBundle{}); !core.Is(err, errModelSessionNil) {
		t.Fatalf("LoadKVBlocksFromMemvid(nil session) error = %v, want %v", err, errModelSessionNil)
	}
}

// TestSession_LoadKVPrefixBlocksFromState_Good streams only the blocks
// needed to cover prefixTokens through the native block restorer.
func TestSession_LoadKVPrefixBlocksFromState_Good(t *testing.T) {
	store := memvid.NewInMemoryStore(nil)
	writer := &Session{session: &sessionfake.Handle{
		KVBlocks: []kv.Block{
			{Index: 0, TokenStart: 0, TokenCount: 2, Snapshot: testNativeKVBlock([]int32{10, 20}, 2, []float32{1, 2, 3, 4}, []float32{9, 10, 11, 12}, nil, nil)},
			{Index: 1, TokenStart: 2, TokenCount: 2, Snapshot: testNativeKVBlock([]int32{30, 40}, 4, []float32{5, 6, 7, 8}, []float32{13, 14, 15, 16}, nil, nil)},
		},
	}}
	bundle, err := writer.SaveKVBlocksToState(context.Background(), store, kv.StateBlockOptions{BlockSize: 2})
	if err != nil {
		t.Fatalf("SaveKVBlocksToState() error = %v", err)
	}

	reader := &sessionfake.Handle{}
	session := &Session{session: reader}
	if err := session.LoadKVPrefixBlocksFromState(context.Background(), store, bundle, 2); err != nil {
		t.Fatalf("LoadKVPrefixBlocksFromState() error = %v", err)
	}
	if len(reader.RestoredBlocks) != 1 {
		t.Fatalf("RestoredBlocks = %v, want 1 block (only the 2-token prefix)", reader.RestoredBlocks)
	}
}

// TestSession_LoadKVPrefixBlocksFromState_Bad — a nil bundle returns the
// sentinel before any store read.
func TestSession_LoadKVPrefixBlocksFromState_Bad(t *testing.T) {
	session := &Session{session: &sessionfake.Handle{}}
	store := memvid.NewInMemoryStore(nil)

	if err := session.LoadKVPrefixBlocksFromState(context.Background(), store, nil, 0); !core.Is(err, errStateKVBlockBundleNil) {
		t.Fatalf("LoadKVPrefixBlocksFromState(nil bundle) error = %v, want %v", err, errStateKVBlockBundleNil)
	}
}

// TestSession_LoadKVPrefixBlocksFromState_Ugly — a nil session returns the
// sentinel before the bundle is even inspected.
func TestSession_LoadKVPrefixBlocksFromState_Ugly(t *testing.T) {
	var session *Session
	store := memvid.NewInMemoryStore(nil)

	if err := session.LoadKVPrefixBlocksFromState(context.Background(), store, &kv.StateBlockBundle{}, 0); !core.Is(err, errModelSessionNil) {
		t.Fatalf("LoadKVPrefixBlocksFromState(nil session) error = %v, want %v", err, errModelSessionNil)
	}
}

// TestSession_LoadKVPrefixBlocksFromMemvid_Good — the deprecated alias
// forwards to LoadKVPrefixBlocksFromState unchanged.
func TestSession_LoadKVPrefixBlocksFromMemvid_Good(t *testing.T) {
	store := memvid.NewInMemoryStore(nil)
	writer := &Session{session: &sessionfake.Handle{
		KVBlocks: []kv.Block{
			{Index: 0, TokenStart: 0, TokenCount: 2, Snapshot: testNativeKVBlock([]int32{10, 20}, 2, []float32{1, 2, 3, 4}, []float32{9, 10, 11, 12}, nil, nil)},
		},
	}}
	bundle, err := writer.SaveKVBlocksToState(context.Background(), store, kv.StateBlockOptions{BlockSize: 2})
	if err != nil {
		t.Fatalf("SaveKVBlocksToState() error = %v", err)
	}

	reader := &sessionfake.Handle{}
	session := &Session{session: reader}
	if err := session.LoadKVPrefixBlocksFromMemvid(context.Background(), store, bundle, 2); err != nil {
		t.Fatalf("LoadKVPrefixBlocksFromMemvid() error = %v", err)
	}
	if len(reader.RestoredBlocks) != 1 {
		t.Fatalf("RestoredBlocks = %v, want 1 block", reader.RestoredBlocks)
	}
}

// TestSession_LoadKVPrefixBlocksFromMemvid_Bad — a nil bundle surfaces the
// same sentinel as the LoadKVPrefixBlocksFromState path it forwards to.
func TestSession_LoadKVPrefixBlocksFromMemvid_Bad(t *testing.T) {
	session := &Session{session: &sessionfake.Handle{}}
	store := memvid.NewInMemoryStore(nil)

	if err := session.LoadKVPrefixBlocksFromMemvid(context.Background(), store, nil, 0); !core.Is(err, errStateKVBlockBundleNil) {
		t.Fatalf("LoadKVPrefixBlocksFromMemvid(nil bundle) error = %v, want %v", err, errStateKVBlockBundleNil)
	}
}

// TestSession_LoadKVPrefixBlocksFromMemvid_Ugly — a nil session returns the
// sentinel through the forwarded call.
func TestSession_LoadKVPrefixBlocksFromMemvid_Ugly(t *testing.T) {
	var session *Session
	store := memvid.NewInMemoryStore(nil)

	if err := session.LoadKVPrefixBlocksFromMemvid(context.Background(), store, &kv.MemvidBlockBundle{}, 0); !core.Is(err, errModelSessionNil) {
		t.Fatalf("LoadKVPrefixBlocksFromMemvid(nil session) error = %v, want %v", err, errModelSessionNil)
	}
}

// TestSession_RestoreBundle_Good restores the session from an in-memory
// bundle whose model identity matches.
func TestSession_RestoreBundle_Good(t *testing.T) {
	source := &Session{session: &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}, info: spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1}}
	snapshot, err := source.CaptureKV()
	if err != nil {
		t.Fatalf("CaptureKV() error = %v", err)
	}
	b, err := mlxbundle.New(snapshot, mlxbundle.Options{Model: "gemma4-e4b"})
	if err != nil {
		t.Fatalf("mlxbundle.New() error = %v", err)
	}
	target := &sessionfake.Handle{}
	session := &Session{session: target, info: spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1}}

	if err := session.RestoreBundle(b); err != nil {
		t.Fatalf("RestoreBundle() error = %v", err)
	}
	if target.RestoredKV == nil {
		t.Fatal("RestoreBundle() did not restore the KV state")
	}
}

// TestSession_RestoreBundle_Bad — a nil bundle returns the sentinel before
// any compatibility check.
func TestSession_RestoreBundle_Bad(t *testing.T) {
	session := &Session{session: &sessionfake.Handle{}}

	if err := session.RestoreBundle(nil); !core.Is(err, errStateBundleNil) {
		t.Fatalf("RestoreBundle(nil) error = %v, want %v", err, errStateBundleNil)
	}
}

// TestSession_RestoreBundle_Ugly — a bundle whose model identity does not
// match the session is rejected by CheckCompatibility before any restore.
func TestSession_RestoreBundle_Ugly(t *testing.T) {
	target := &sessionfake.Handle{}
	session := &Session{session: target, info: spine.ModelInfo{Architecture: "llama", NumLayers: 99}}
	b := &mlxbundle.Bundle{
		Version: mlxbundle.Version,
		Kind:    mlxbundle.Kind,
		Model:   mlxbundle.Model{Architecture: "gemma4_text", NumLayers: 1},
		KVHash:  "deadbeef",
	}

	if err := session.RestoreBundle(b); err == nil {
		t.Fatal("RestoreBundle(mismatch) error = nil, want incompatibility")
	}
	if target.RestoredKV != nil {
		t.Fatalf("RestoreBundle(mismatch) restored despite mismatch = %+v", target.RestoredKV)
	}
}

// TestSession_RestoreBundleFromState_Good restores the session from a
// State-backed bundle reference.
func TestSession_RestoreBundleFromState_Good(t *testing.T) {
	store := memvid.NewInMemoryStore(nil)
	snapshot := sessionTestRootSnapshot()
	ref, err := snapshot.SaveState(context.Background(), store, kv.StateOptions{})
	if err != nil {
		t.Fatalf("SaveState() error = %v", err)
	}
	hash, err := kv.HashSnapshot(snapshot)
	if err != nil {
		t.Fatalf("kv.HashSnapshot() error = %v", err)
	}
	target := &sessionfake.Handle{}
	session := &Session{session: target, info: spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1}}
	b := &mlxbundle.Bundle{
		Version: mlxbundle.Version,
		Kind:    mlxbundle.Kind,
		Model:   mlxbundle.Model{Architecture: "gemma4_text", NumLayers: 1},
		KVHash:  hash,
		Refs:    []mlxbundle.Ref{{Kind: mlxbundle.RefState, URI: mlxbundle.StateURI(ref), State: ref}},
	}

	if err := session.RestoreBundleFromState(context.Background(), b, store); err != nil {
		t.Fatalf("RestoreBundleFromState() error = %v", err)
	}
	if target.RestoredKV == nil || target.RestoredKV.Tokens[0] != 1 {
		t.Fatalf("restored KV = %+v, want two-token state", target.RestoredKV)
	}
}

// TestSession_RestoreBundleFromState_Bad — a nil bundle returns the
// sentinel before any State read.
func TestSession_RestoreBundleFromState_Bad(t *testing.T) {
	session := &Session{session: &sessionfake.Handle{}}
	store := memvid.NewInMemoryStore(nil)

	if err := session.RestoreBundleFromState(context.Background(), nil, store); !core.Is(err, errStateBundleNil) {
		t.Fatalf("RestoreBundleFromState(nil bundle) error = %v, want %v", err, errStateBundleNil)
	}
}

// TestSession_RestoreBundleFromState_Ugly — a mismatched model identity is
// rejected by CheckCompatibility before any State read.
func TestSession_RestoreBundleFromState_Ugly(t *testing.T) {
	store := memvid.NewInMemoryStore(nil)
	session := &Session{session: &sessionfake.Handle{}, info: spine.ModelInfo{Architecture: "llama", NumLayers: 99}}
	b := &mlxbundle.Bundle{
		Version: mlxbundle.Version,
		Kind:    mlxbundle.Kind,
		Model:   mlxbundle.Model{Architecture: "gemma4_text", NumLayers: 1},
		KVHash:  "deadbeef",
	}

	if err := session.RestoreBundleFromState(context.Background(), b, store); err == nil {
		t.Fatal("RestoreBundleFromState(mismatch) error = nil, want incompatibility")
	}
}

// TestSession_RestoreBundleFromMemvid_Good — the deprecated alias forwards
// to RestoreBundleFromState unchanged.
func TestSession_RestoreBundleFromMemvid_Good(t *testing.T) {
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
	target := &sessionfake.Handle{}
	session := &Session{session: target, info: spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1}}
	b := &mlxbundle.Bundle{
		Version: mlxbundle.Version,
		Kind:    mlxbundle.Kind,
		Model:   mlxbundle.Model{Architecture: "gemma4_text", NumLayers: 1},
		KVHash:  hash,
		Refs:    []mlxbundle.Ref{{Kind: mlxbundle.RefMemvid, URI: mlxbundle.MemvidURI(ref), Memvid: ref}},
	}

	if err := session.RestoreBundleFromMemvid(context.Background(), b, store); err != nil {
		t.Fatalf("RestoreBundleFromMemvid() error = %v", err)
	}
	if target.RestoredKV == nil || target.RestoredKV.Tokens[0] != 1 {
		t.Fatalf("restored KV = %+v, want two-token state", target.RestoredKV)
	}
}

// TestSession_RestoreBundleFromMemvid_Bad — a nil bundle surfaces the same
// sentinel as the RestoreBundleFromState path it forwards to.
func TestSession_RestoreBundleFromMemvid_Bad(t *testing.T) {
	session := &Session{session: &sessionfake.Handle{}}
	store := memvid.NewInMemoryStore(nil)

	if err := session.RestoreBundleFromMemvid(context.Background(), nil, store); !core.Is(err, errStateBundleNil) {
		t.Fatalf("RestoreBundleFromMemvid(nil bundle) error = %v, want %v", err, errStateBundleNil)
	}
}

// TestSession_RestoreBundleFromMemvid_Ugly — a mismatched model identity is
// rejected through the forwarded call.
func TestSession_RestoreBundleFromMemvid_Ugly(t *testing.T) {
	store := memvid.NewInMemoryStore(nil)
	session := &Session{session: &sessionfake.Handle{}, info: spine.ModelInfo{Architecture: "llama", NumLayers: 99}}
	b := &mlxbundle.Bundle{
		Version: mlxbundle.Version,
		Kind:    mlxbundle.Kind,
		Model:   mlxbundle.Model{Architecture: "gemma4_text", NumLayers: 1},
		KVHash:  "deadbeef",
	}

	if err := session.RestoreBundleFromMemvid(context.Background(), b, store); err == nil {
		t.Fatal("RestoreBundleFromMemvid(mismatch) error = nil, want incompatibility")
	}
}

// TestSession_LoadBundle_Good — a bundle written to disk is loaded and
// restored, covering LoadBundle's success-then-RestoreBundle tail.
func TestSession_LoadBundle_Good(t *testing.T) {
	native := &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}
	source := &Session{session: native, info: spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1}}
	snapshot, err := source.CaptureKV()
	if err != nil {
		t.Fatalf("CaptureKV() error = %v", err)
	}
	b, err := mlxbundle.New(snapshot, mlxbundle.Options{Model: "gemma4-e4b", Prompt: "ctx"})
	if err != nil {
		t.Fatalf("mlxbundle.New() error = %v", err)
	}
	path := core.PathJoin(t.TempDir(), "session.bundle.json")
	if err := b.Save(path); err != nil {
		t.Fatalf("bundle.Save() error = %v", err)
	}

	target := &sessionfake.Handle{}
	session := &Session{session: target, info: spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1}}
	if err := session.LoadBundle(path); err != nil {
		t.Fatalf("LoadBundle() error = %v", err)
	}
	if target.RestoredKV == nil || target.RestoredKV.Tokens[0] != 1 {
		t.Fatalf("LoadBundle() restored KV = %+v, want two-token state", target.RestoredKV)
	}
}

// TestSession_LoadBundle_Bad — a missing bundle file surfaces the load
// failure before any restore is attempted.
func TestSession_LoadBundle_Bad(t *testing.T) {
	session := &Session{session: &sessionfake.Handle{}}

	if err := session.LoadBundle(core.PathJoin(t.TempDir(), "missing.bundle.json")); err == nil {
		t.Fatal("LoadBundle(missing) error = nil, want load failure")
	}
}

// TestSession_LoadBundle_Ugly — the bundle file loads fine but its model
// identity does not match the session, so RestoreBundle rejects it.
func TestSession_LoadBundle_Ugly(t *testing.T) {
	source := &Session{session: &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}, info: spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1}}
	snapshot, err := source.CaptureKV()
	if err != nil {
		t.Fatalf("CaptureKV() error = %v", err)
	}
	b, err := mlxbundle.New(snapshot, mlxbundle.Options{Model: "gemma4-e4b"})
	if err != nil {
		t.Fatalf("mlxbundle.New() error = %v", err)
	}
	path := core.PathJoin(t.TempDir(), "mismatch.bundle.json")
	if err := b.Save(path); err != nil {
		t.Fatalf("bundle.Save() error = %v", err)
	}

	session := &Session{session: &sessionfake.Handle{}, info: spine.ModelInfo{Architecture: "llama", NumLayers: 99}}
	if err := session.LoadBundle(path); err == nil {
		t.Fatal("LoadBundle(mismatch) error = nil, want incompatibility")
	}
}

// TestSession_Fork_Good creates an independent session over the seeded fork
// handle, carrying a cloned agent-memory report forward.
func TestSession_Fork_Good(t *testing.T) {
	forkedNative := &sessionfake.Handle{}
	native := &sessionfake.Handle{Forked: forkedNative}
	session := &Session{session: native, agentMemory: &agent.WakeReport{EntryURI: "parent"}}

	forked, err := session.Fork()

	if err != nil {
		t.Fatalf("Fork() error = %v", err)
	}
	if forked == nil || forked.session != forkedNative {
		t.Fatalf("Fork() = %+v, want wrapped fork handle", forked)
	}
	if forked.agentMemory == nil || forked.agentMemory.EntryURI != "parent" {
		t.Fatalf("Fork() agentMemory = %+v, want cloned parent report", forked.agentMemory)
	}
}

// TestSession_Fork_Ugly — a native Fork that returns a nil handle without an
// error surfaces the nil-session-fork sentinel, distinct from the
// error-propagation Ugly case in the scenario tests above.
func TestSession_Fork_Ugly(t *testing.T) {
	session := &Session{session: &sessionfake.Handle{}}

	forked, err := session.Fork()

	if !core.Is(err, errNativeNilSessionFork) {
		t.Fatalf("Fork() error = %v, want %v", err, errNativeNilSessionFork)
	}
	if forked != nil {
		t.Fatalf("Fork() = %+v, want nil", forked)
	}
}

// TestSession_Reset_Good releases retained state and clears the agent-memory
// cache.
func TestSession_Reset_Good(t *testing.T) {
	native := &sessionfake.Handle{}
	session := &Session{session: native, agentMemory: &agent.WakeReport{EntryURI: "stale"}}

	session.Reset()

	if native.ResetCalls != 1 {
		t.Fatalf("ResetCalls = %d, want 1", native.ResetCalls)
	}
	if session.agentMemory != nil {
		t.Fatal("Reset() left a stale agentMemory cache")
	}
}

// TestSession_Reset_Bad — a nil session is a no-op, not a panic.
func TestSession_Reset_Bad(t *testing.T) {
	var session *Session
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("Reset(nil session) panicked: %v", r)
		}
	}()

	session.Reset()
}

// TestSession_Reset_Ugly — a non-nil Session with a nil native handle is
// also a no-op, distinct from the nil-*Session Bad case.
func TestSession_Reset_Ugly(t *testing.T) {
	session := &Session{agentMemory: &agent.WakeReport{EntryURI: "stale"}}

	session.Reset()

	if session.agentMemory == nil {
		t.Fatal("Reset() on a nil-handle session cleared agentMemory, want the guard to skip the whole body")
	}
}

// TestSession_Close_Good releases the native handle and clears the session
// pointer so a second Close is a no-op.
func TestSession_Close_Good(t *testing.T) {
	native := &sessionfake.Handle{}
	session := &Session{session: native}

	if err := session.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if native.CloseCalls != 1 {
		t.Fatalf("CloseCalls = %d, want 1", native.CloseCalls)
	}
	if session.session != nil {
		t.Fatal("Close() did not clear the native handle")
	}
}

// TestSession_Close_Bad — a nil session returns nil rather than panicking.
func TestSession_Close_Bad(t *testing.T) {
	var session *Session

	if err := session.Close(); err != nil {
		t.Fatalf("Close(nil session) error = %v, want nil", err)
	}
}

// TestSession_Native_Good returns the underlying native handle unchanged.
func TestSession_Native_Good(t *testing.T) {
	native := &sessionfake.Handle{}
	session := &Session{session: native}

	if session.Native() != native {
		t.Fatalf("Native() = %v, want %v", session.Native(), native)
	}
}

// TestSession_Native_Bad — a nil session returns a nil handle.
func TestSession_Native_Bad(t *testing.T) {
	var session *Session

	if session.Native() != nil {
		t.Fatal("Native(nil session) != nil, want nil")
	}
}

// TestSession_Native_Ugly — a non-nil Session with no native handle also
// returns nil, distinct from the nil-*Session Bad case.
func TestSession_Native_Ugly(t *testing.T) {
	session := &Session{}

	if session.Native() != nil {
		t.Fatal("Native(no handle) != nil, want nil")
	}
}

// TestSession_Valid_Good reports true for a session holding a live handle.
func TestSession_Valid_Good(t *testing.T) {
	session := &Session{session: &sessionfake.Handle{}}

	if !session.Valid() {
		t.Fatal("Valid() = false, want true")
	}
}

// TestSession_Valid_Bad — a nil session reports false.
func TestSession_Valid_Bad(t *testing.T) {
	var session *Session

	if session.Valid() {
		t.Fatal("Valid(nil session) = true, want false")
	}
}

// TestSession_Valid_Ugly — a non-nil Session with no native handle also
// reports false, distinct from the nil-*Session Bad case.
func TestSession_Valid_Ugly(t *testing.T) {
	session := &Session{}

	if session.Valid() {
		t.Fatal("Valid(no handle) = true, want false")
	}
}

// TestSession_Err_Good returns nil when the native handle carries no error.
func TestSession_Err_Good(t *testing.T) {
	session := &Session{session: &sessionfake.Handle{}}

	if err := session.Err(); err != nil {
		t.Fatalf("Err() = %v, want nil", err)
	}
}

// TestSession_Err_Bad — a nil session returns nil rather than panicking.
func TestSession_Err_Bad(t *testing.T) {
	var session *Session

	if err := session.Err(); err != nil {
		t.Fatalf("Err(nil session) = %v, want nil", err)
	}
}

// TestSession_Err_Ugly — a live handle's seeded error is forwarded verbatim.
func TestSession_Err_Ugly(t *testing.T) {
	wantErr := core.NewError("native err value")
	session := &Session{session: &sessionfake.Handle{ErrValue: wantErr}}

	if err := session.Err(); !core.Is(err, wantErr) {
		t.Fatalf("Err() = %v, want %v", err, wantErr)
	}
}
