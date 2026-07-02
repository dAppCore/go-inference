// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	"encoding/json"
	"testing"

	"dappco.re/go/inference"
)

// TestRerankResponse_AppendRoundTrip locks the hand-rolled rerank
// encoder shape against encoding/json. The rerank wire is a
// single-shape contract (object/model/results) so the test exercises
// every RerankScore branch (with/without text/labels/zero-score).
func TestRerankResponse_AppendRoundTrip(t *testing.T) {
	cases := []struct {
		name string
		in   RerankResponse
	}{
		{"empty-results", RerankResponse{Object: "list", Model: "qwen3-rerank"}},
		{"basic-results", RerankResponse{
			Object: "list", Model: "qwen3-rerank",
			Results: []inference.RerankScore{
				{Index: 0, Score: 0.91, Text: "alpha"},
				{Index: 1, Score: 0.82, Text: "beta"},
				{Index: 2, Score: 0.74, Text: "gamma"},
			},
		}},
		{"with-labels", RerankResponse{
			Object: "list", Model: "qwen3-rerank",
			Results: []inference.RerankScore{{
				Index: 0, Score: 0.95, Text: "x",
				Labels: map[string]string{"locale": "en"},
			}},
		}},
		{"with-multiple-labels", RerankResponse{
			Object: "list", Model: "qwen3-rerank",
			Results: []inference.RerankScore{{
				Index: 0, Score: 0.95, Text: "x",
				// 2+ labels exercises the comma-separator branch
				// between label entries (appendRerankScore) and the
				// per-label size-accumulation loop (rerankResponseSize).
				Labels: map[string]string{"locale": "en", "tier": "hot"},
			}},
		}},
		{"zero-score", RerankResponse{
			Object: "list", Model: "qwen3-rerank",
			Results: []inference.RerankScore{{Index: 0, Text: "match"}},
		}},
		{"escapes", RerankResponse{
			Object: "list", Model: "qwen3-rerank",
			Results: []inference.RerankScore{{Index: 0, Score: 0.5, Text: "quote \" tab\t"}},
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			encoded := appendRerankResponse(nil, tc.in)
			var back RerankResponse
			if err := json.Unmarshal(encoded, &back); err != nil {
				t.Fatalf("json.Unmarshal(%s) error = %v", encoded, err)
			}
			if back.Object != tc.in.Object || back.Model != tc.in.Model {
				t.Fatalf("identity: got %+v, want %+v", back, tc.in)
			}
			if len(back.Results) != len(tc.in.Results) {
				t.Fatalf("results len = %d, want %d", len(back.Results), len(tc.in.Results))
			}
			for i := range tc.in.Results {
				if back.Results[i].Index != tc.in.Results[i].Index ||
					back.Results[i].Score != tc.in.Results[i].Score ||
					back.Results[i].Text != tc.in.Results[i].Text {
					t.Fatalf("results[%d] = %+v, want %+v", i, back.Results[i], tc.in.Results[i])
				}
				if len(back.Results[i].Labels) != len(tc.in.Results[i].Labels) {
					t.Fatalf("results[%d].labels len = %d, want %d", i, len(back.Results[i].Labels), len(tc.in.Results[i].Labels))
				}
				for k, v := range tc.in.Results[i].Labels {
					if back.Results[i].Labels[k] != v {
						t.Fatalf("results[%d].labels[%q] = %q, want %q", i, k, back.Results[i].Labels[k], v)
					}
				}
			}
		})
	}
}

// TestRerankResponseSize_Good pins the size estimator against the
// actual encoded length for a fully-populated result set (multiple
// results, one with labels) — the single-allocation contract every
// *Size estimator in this package exists to guarantee.
func TestRerankResponseSize_Good(t *testing.T) {
	resp := RerankResponse{
		Object: "list", Model: "qwen3-rerank",
		Results: []inference.RerankScore{
			{Index: 0, Score: 0.91, Text: "alpha"},
			{Index: 1, Score: 0.82, Text: "beta", Labels: map[string]string{"locale": "en", "tier": "hot"}},
		},
	}
	buf := make([]byte, 0, rerankResponseSize(resp))
	got := appendRerankResponse(buf, resp)
	if len(got) > cap(buf) {
		t.Fatalf("rerankResponseSize(%+v) = %d, actual encoded length %d exceeds capacity", resp, rerankResponseSize(resp), len(got))
	}
}

// TestRerankResponseSize_Bad covers the empty-results floor.
func TestRerankResponseSize_Bad(t *testing.T) {
	resp := RerankResponse{Object: "list", Model: "qwen3-rerank"}
	buf := make([]byte, 0, rerankResponseSize(resp))
	got := appendRerankResponse(buf, resp)
	if len(got) > cap(buf) {
		t.Fatalf("rerankResponseSize(empty) = %d, actual encoded length %d exceeds capacity", rerankResponseSize(resp), len(got))
	}
}

// TestRerankResponseSize_Ugly drives a zero-score, no-labels result
// through the estimator — Score==0 and empty Labels both skip their
// respective size-accumulation branches, the complement of the Good
// case above.
func TestRerankResponseSize_Ugly(t *testing.T) {
	resp := RerankResponse{
		Object: "list", Model: "qwen3-rerank",
		Results: []inference.RerankScore{{Index: 0, Text: "match"}},
	}
	buf := make([]byte, 0, rerankResponseSize(resp))
	got := appendRerankResponse(buf, resp)
	if len(got) > cap(buf) {
		t.Fatalf("rerankResponseSize(zero-score) = %d, actual encoded length %d exceeds capacity", rerankResponseSize(resp), len(got))
	}
}
