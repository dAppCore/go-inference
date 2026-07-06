// SPDX-Licence-Identifier: EUPL-1.2

package engine

import (
	"context"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/decode/tokenizer"
)

// mustExampleTokenizer builds the same fixtureTokenizerJSON as the *_test.go
// suite, but without a *testing.T (Example functions take none) — mirrors
// decode/tokenizer's own mustExampleTokenizer fixture pattern: MkdirTemp +
// WriteFile + LoadTokenizer, panicking on setup failure since an Example has
// no t.Fatalf to fall back on.
func mustExampleTokenizer() (*tokenizer.Tokenizer, func()) {
	dirResult := core.MkdirTemp("", "go-inference-engine-example-*")
	if !dirResult.OK {
		panic(dirResult.Value)
	}
	dir := dirResult.Value.(string)
	path := core.PathJoin(dir, "tokenizer.json")
	if result := core.WriteFile(path, []byte(fixtureTokenizerJSON), 0o644); !result.OK {
		core.RemoveAll(dir)
		panic(result.Value)
	}
	tok, err := tokenizer.LoadTokenizer(path)
	if err != nil {
		core.RemoveAll(dir)
		panic(err)
	}
	return tok, func() { core.RemoveAll(dir) }
}

func ExampleNewTextModel() {
	tok, cleanup := mustExampleTokenizer()
	defer cleanup()
	m := NewTextModel(&fakeTokenModel{}, tok, "gemma-test", inference.ModelInfo{Architecture: "gemma-test"}, 4096)
	core.Println(m.ModelType())
	// Output: gemma-test
}

func ExampleTextModel_OpenTrainer() {
	tok, cleanup := mustExampleTokenizer()
	defer cleanup()
	m := NewTextModel(&fakeTrainerModel{trainer: &fakeTrainer{}}, tok, "gemma-test", inference.ModelInfo{}, 4096)
	tr, err := m.OpenTrainer(inference.TrainingConfig{Epochs: 1})
	if err != nil {
		panic(err)
	}
	defer func() { _ = tr.Close() }()
	loss, err := tr.Step(inference.Batch{TokenIDs: [][]int32{{1, 2, 3}}})
	if err != nil {
		panic(err)
	}
	core.Println(loss)
	// Output: 0
}

func ExampleTextModel_Generate() {
	tok, cleanup := mustExampleTokenizer()
	defer cleanup()
	m := NewTextModel(&fakeTokenModel{genIDs: []int32{10, 11}}, tok, "gemma-test", inference.ModelInfo{}, 4096)
	count := 0
	for range m.Generate(context.Background(), "hi", inference.WithMaxTokens(2)) {
		count++
	}
	core.Println(count)
	// Output: 2
}

func ExampleTextModel_Chat() {
	tok, cleanup := mustExampleTokenizer()
	defer cleanup()
	m := NewTextModel(&fakeTokenModel{genIDs: []int32{10, 11}}, tok, "gemma-test", inference.ModelInfo{}, 4096)
	count := 0
	for range m.Chat(context.Background(), []inference.Message{{Role: "user", Content: "hi"}}, inference.WithMaxTokens(2)) {
		count++
	}
	core.Println(count)
	// Output: 2
}

func ExampleTextModel_FormatChatPrompt() {
	m := &TextModel{}
	core.Println(m.FormatChatPrompt([]inference.Message{{Role: "user", Content: "hi"}}))
	// Output: <start_of_turn>user
	// hi<end_of_turn>
	// <start_of_turn>model
}

func ExampleTextModel_FormatChatContinuation() {
	m := &TextModel{}
	core.Println(m.FormatChatContinuation([]inference.Message{{Role: "user", Content: "and now?"}}))
	// Output: <end_of_turn>
	// <start_of_turn>user
	// and now?<end_of_turn>
	// <start_of_turn>model
}

func ExampleTextModel_Classify() {
	tok, cleanup := mustExampleTokenizer()
	defer cleanup()
	m := NewTextModel(&fakeTokenModel{genIDs: []int32{7}}, tok, "gemma-test", inference.ModelInfo{}, 4096)
	r := m.Classify(context.Background(), []string{"positive", "negative"})
	if !r.OK {
		panic(r.Error())
	}
	results := r.Value.([]inference.ClassifyResult)
	core.Println(len(results), results[0].Token.ID)
	// Output: 2 7
}

func ExampleTextModel_BatchGenerate() {
	tok, cleanup := mustExampleTokenizer()
	defer cleanup()
	m := NewTextModel(&fakeTokenModel{genIDs: []int32{1, 2}}, tok, "gemma-test", inference.ModelInfo{}, 4096)
	r := m.BatchGenerate(context.Background(), []string{"a", "b"}, inference.WithMaxTokens(2))
	if !r.OK {
		panic(r.Error())
	}
	results := r.Value.([]inference.BatchResult)
	core.Println(len(results), len(results[0].Tokens))
	// Output: 2 2
}

func ExampleTextModel_NewSession() {
	tok, cleanup := mustExampleTokenizer()
	defer cleanup()
	m := NewTextModel(&fakeTokenModel{}, tok, "gemma-test", inference.ModelInfo{}, 4096)
	sess := m.NewSession()
	defer func() { _ = sess.Close() }()
	core.Println(sess != nil)
	// Output: true
}

func ExampleTextModel_ModelType() {
	m := &TextModel{modelType: "gemma4"}
	core.Println(m.ModelType())
	// Output: gemma4
}

func ExampleTextModel_Info() {
	m := &TextModel{info: inference.ModelInfo{Architecture: "gemma3", VocabSize: 262144}}
	info := m.Info()
	core.Println(info.Architecture, info.VocabSize)
	// Output: gemma3 262144
}

func ExampleTextModel_Capabilities() {
	m := &TextModel{tm: cacheModeTokenModel{modes: []string{"native"}}}
	report := m.Capabilities()
	core.Println(report.CacheModes)
	// Output: [native]
}

func ExampleTextModel_Metrics() {
	m := &TextModel{}
	m.setMetrics(5, 3, 2*time.Millisecond, time.Now())
	got := m.Metrics()
	core.Println(got.PromptTokens, got.GeneratedTokens)
	// Output: 5 3
}

func ExampleTextModel_Err() {
	m := &TextModel{}
	m.setOK()
	core.Println(m.Err().OK)
	// Output: true
}

func ExampleTextModel_Close() {
	tm := &fakeTokenModel{}
	m := NewTextModel(tm, nil, "gemma-test", inference.ModelInfo{}, 4096)
	r := m.Close()
	core.Println(r.OK)
	// Output: true
}
