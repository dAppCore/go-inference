// SPDX-Licence-Identifier: EUPL-1.2

package engine

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/kv"
)

func ExampleNewSessionHandle() {
	tok, cleanup := mustExampleTokenizer()
	defer cleanup()
	m := NewTextModel(&fakeTokenModel{}, tok, "gemma-test", inference.ModelInfo{}, 4096)
	handle := NewSessionHandle(m, &fakeSession{})
	defer func() { _ = handle.Close() }()
	if err := handle.Prefill(context.Background(), "hi"); err != nil {
		panic(err)
	}
	core.Println(handle.Err() == nil)
	// Output: true
}

func ExampleSessionHandle_Prefill() {
	tok, cleanup := mustExampleTokenizer()
	defer cleanup()
	m := NewTextModel(&fakeTokenModel{}, tok, "gemma-test", inference.ModelInfo{}, 4096)
	handle := NewSessionHandle(m, &fakeSession{})
	defer func() { _ = handle.Close() }()
	err := handle.Prefill(context.Background(), "You are a helpful assistant.")
	core.Println(err == nil)
	// Output: true
}

func ExampleSessionHandle_AppendPrompt() {
	tok, cleanup := mustExampleTokenizer()
	defer cleanup()
	m := NewTextModel(&fakeTokenModel{}, tok, "gemma-test", inference.ModelInfo{}, 4096)
	handle := NewSessionHandle(m, &fakeSession{})
	defer func() { _ = handle.Close() }()
	if err := handle.Prefill(context.Background(), "first turn"); err != nil {
		panic(err)
	}
	err := handle.AppendPrompt(context.Background(), " second turn")
	core.Println(err == nil)
	// Output: true
}

func ExampleSessionHandle_Generate() {
	tok, cleanup := mustExampleTokenizer()
	defer cleanup()
	m := NewTextModel(&fakeTokenModel{}, tok, "gemma-test", inference.ModelInfo{}, 4096)
	handle := NewSessionHandle(m, &fakeSession{pos: 1, genIDs: []int32{10, 11}})
	defer func() { _ = handle.Close() }()
	count := 0
	for range handle.Generate(context.Background(), inference.GenerateConfig{MaxTokens: 2}) {
		count++
	}
	core.Println(count)
	// Output: 2
}

func ExampleSessionHandle_CaptureKV() {
	tok, cleanup := mustExampleTokenizer()
	defer cleanup()
	m := NewTextModel(&fakeTokenModel{}, tok, "gemma-test", inference.ModelInfo{}, 4096)
	handle := NewSessionHandle(m, &fakeSession{pos: 3})
	defer func() { _ = handle.Close() }()
	snap, err := handle.CaptureKV(context.Background())
	if err != nil {
		panic(err)
	}
	core.Println(snap.SeqLen)
	// Output: 3
}

func ExampleSessionHandle_RangeKVBlocks() {
	tok, cleanup := mustExampleTokenizer()
	defer cleanup()
	m := NewTextModel(&fakeTokenModel{}, tok, "gemma-test", inference.ModelInfo{}, 4096)
	handle := NewSessionHandle(m, &fakeSession{pos: 4})
	defer func() { _ = handle.Close() }()
	blocks := 0
	err := handle.RangeKVBlocks(context.Background(), 16, kv.CaptureOptions{}, func(kv.Block) (bool, error) {
		blocks++
		return true, nil
	})
	if err != nil {
		panic(err)
	}
	core.Println(blocks)
	// Output: 1
}

func ExampleSessionHandle_RestoreKV() {
	handle := NewSessionHandle(&TextModel{}, &fakeSession{})
	defer func() { _ = handle.Close() }()
	err := handle.RestoreKV(context.Background(), &kv.Snapshot{Tokens: []int32{1, 2, 3}})
	core.Println(err == nil)
	// Output: true
}

func ExampleSessionHandle_RestoreFromKV() {
	handle := NewSessionHandle(&TextModel{}, &fakeSession{})
	defer func() { _ = handle.Close() }()
	err := handle.RestoreFromKV(context.Background(), &kv.Snapshot{Tokens: []int32{1, 2, 3}})
	core.Println(err == nil)
	// Output: true
}

func ExampleSessionHandle_Fork() {
	tok, cleanup := mustExampleTokenizer()
	defer cleanup()
	m := NewTextModel(&fakeTokenModel{genIDs: defaultFakeGenIDs}, tok, "gemma-test", inference.ModelInfo{}, 4096)
	handle := NewSessionHandle(m, &fakeSession{pos: 1, genIDs: defaultFakeGenIDs})
	defer func() { _ = handle.Close() }()
	fork, err := handle.Fork(context.Background())
	if err != nil {
		panic(err)
	}
	defer func() { _ = fork.Close() }()
	core.Println(fork != nil)
	// Output: true
}

func ExampleSessionHandle_Reset() {
	tok, cleanup := mustExampleTokenizer()
	defer cleanup()
	m := NewTextModel(&fakeTokenModel{}, tok, "gemma-test", inference.ModelInfo{}, 4096)
	handle := NewSessionHandle(m, &fakeSession{})
	defer func() { _ = handle.Close() }()
	if err := handle.Prefill(context.Background(), "before reset"); err != nil {
		panic(err)
	}
	handle.Reset()
	err := handle.Prefill(context.Background(), "after reset")
	core.Println(err == nil)
	// Output: true
}

func ExampleSessionHandle_Close() {
	handle := NewSessionHandle(&TextModel{}, &fakeSession{})
	err := handle.Close()
	core.Println(err == nil)
	// Output: true
}

func ExampleSessionHandle_Err() {
	m := NewTextModel(&fakeTokenModel{}, nil, "gemma-test", inference.ModelInfo{}, 4096)
	handle := NewSessionHandle(m, &fakeSession{})
	defer func() { _ = handle.Close() }()
	_ = handle.AppendPrompt(context.Background(), "no prefix yet")
	core.Println(handle.Err() != nil)
	// Output: true
}
