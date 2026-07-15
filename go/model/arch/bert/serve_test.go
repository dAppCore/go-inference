// SPDX-Licence-Identifier: EUPL-1.2

package bert_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/inference/model/arch/bert"
	"dappco.re/go/inference/serving/compat"
	"dappco.re/go/inference/serving/provider/openai"
)

// TestServeModel_MuxEmbeddingsAndRerank drives the host encoder end-to-end
// through the assembled OpenAI-compatible mux — the same path a live `curl` to
// /v1/embeddings and /v1/rerank exercises. Opt-in on the bge-small snapshot.
func TestServeModel_MuxEmbeddingsAndRerank(t *testing.T) {
	snapshot := bgeSmallSnapshot(t)
	if snapshot == "" {
		t.Skip("bge-small-en-v1.5 snapshot not found; set BERT_PARITY_SNAPSHOT for the serve receipt")
	}
	model, err := bert.Load(snapshot)
	if err != nil {
		t.Fatalf("bert.Load: %v", err)
	}
	resolver := openai.NewStaticResolver(map[string]inference.TextModel{
		"bge-small-en-v1.5": bert.NewServeModel(model),
	})
	server := httptest.NewServer(compat.NewMux(resolver))
	t.Cleanup(server.Close)

	t.Run("embeddings_float", func(t *testing.T) {
		body := post(t, server.URL+openai.DefaultEmbeddingsPath,
			`{"model":"bge-small-en-v1.5","input":["hello world","password reset"]}`)
		var resp struct {
			Data []struct {
				Embedding []float32 `json:"embedding"`
			} `json:"data"`
			Usage inference.EmbeddingUsage `json:"usage"`
		}
		if err := json.Unmarshal(body, &resp); err != nil {
			t.Fatalf("decode: %v (body=%s)", err, body)
		}
		if len(resp.Data) != 2 {
			t.Fatalf("data len = %d, want 2", len(resp.Data))
		}
		if len(resp.Data[0].Embedding) != model.Config().HiddenSize {
			t.Fatalf("vector dim = %d, want %d", len(resp.Data[0].Embedding), model.Config().HiddenSize)
		}
		t.Logf("embeddings: 2 vectors of dim %d, prompt_tokens=%d", len(resp.Data[0].Embedding), resp.Usage.PromptTokens)
	})

	t.Run("embeddings_base64", func(t *testing.T) {
		body := post(t, server.URL+openai.DefaultEmbeddingsPath,
			`{"model":"bge-small-en-v1.5","input":"hello world","encoding_format":"base64"}`)
		var resp struct {
			Data []struct {
				Embedding string `json:"embedding"`
			} `json:"data"`
		}
		if err := json.Unmarshal(body, &resp); err != nil {
			t.Fatalf("decode: %v (body=%s)", err, body)
		}
		if len(resp.Data) != 1 || resp.Data[0].Embedding == "" {
			t.Fatalf("expected one base64 embedding string, got %s", body)
		}
	})

	t.Run("rerank", func(t *testing.T) {
		body := post(t, server.URL+openai.DefaultRerankPath, `{
			"model":"bge-small-en-v1.5",
			"query":"How do I reset my password?",
			"documents":[
				"The quick brown fox jumps over the lazy dog.",
				"Open account settings and choose reset password.",
				"Vector search retrieves semantically similar documents."
			],
			"top_n":2
		}`)
		var resp struct {
			Results []inference.RerankScore `json:"results"`
		}
		if err := json.Unmarshal(body, &resp); err != nil {
			t.Fatalf("decode: %v (body=%s)", err, body)
		}
		if len(resp.Results) != 2 {
			t.Fatalf("results len = %d, want 2 (top_n)", len(resp.Results))
		}
		if resp.Results[0].Index != 1 {
			t.Errorf("top result index = %d, want 1 (the reset-password doc)", resp.Results[0].Index)
		}
		t.Logf("rerank top: index=%d score=%.4f", resp.Results[0].Index, resp.Results[0].Score)
	})
}

func post(t *testing.T, url, body string) []byte {
	t.Helper()
	resp, err := http.Post(url, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST %s: %v", url, err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("POST %s status = %d, body=%s", url, resp.StatusCode, data)
	}
	return data
}
