// SPDX-Licence-Identifier: EUPL-1.2

// Hand-rolled encoders for the OpenAI service-endpoint wire shapes
// (rerank). Embeddings is encoded in chunkenc.go alongside the
// chat-completion shapes; rerank lives here because it walks the
// inference.RerankScore contract type, owned by the contract layer.

package openai

import "dappco.re/go/inference"

// appendRerankScore walks one inference.RerankScore into buf. The
// contract carries Index / Score / Text / Labels with omitempty on
// every field — emit only the fields that carry a non-zero value.
// Field ordering matches the struct declaration so wire output is
// byte-compatible with encoding/json's reflect walk.
func appendRerankScore(buf []byte, score inference.RerankScore) []byte {
	buf = append(buf, '{')
	leading := false
	if score.Index != 0 {
		buf = appendIntField(buf, "index", score.Index, false)
		leading = true
	}
	if score.Score != 0 {
		if leading {
			buf = append(buf, ',')
		}
		buf = append(buf, '"', 's', 'c', 'o', 'r', 'e', '"', ':')
		buf = appendFloat64(buf, score.Score)
		leading = true
	}
	if score.Text != "" {
		buf = appendStringField(buf, "text", score.Text, leading)
		leading = true
	}
	if len(score.Labels) > 0 {
		if leading {
			buf = append(buf, ',')
		}
		buf = append(buf, '"', 'l', 'a', 'b', 'e', 'l', 's', '"', ':', '{')
		labelFirst := true
		for k, v := range score.Labels {
			if !labelFirst {
				buf = append(buf, ',')
			}
			labelFirst = false
			buf = appendJSONString(buf, k)
			buf = append(buf, ':')
			buf = appendJSONString(buf, v)
		}
		buf = append(buf, '}')
	}
	return append(buf, '}')
}

// appendRerankResponse walks the RerankResponse shape into buf.
// The Results slice scales with documents: walking inference.RerankScore
// inline skips the per-element reflect cost encoding/json pays.
func appendRerankResponse(buf []byte, resp RerankResponse) []byte {
	buf = append(buf, '{')
	buf = appendStringField(buf, "object", resp.Object, false)
	buf = appendStringField(buf, "model", resp.Model, true)
	buf = append(buf, ',', '"', 'r', 'e', 's', 'u', 'l', 't', 's', '"', ':', '[')
	for i, score := range resp.Results {
		if i > 0 {
			buf = append(buf, ',')
		}
		buf = appendRerankScore(buf, score)
	}
	return append(buf, ']', '}')
}

// rerankResponseSize estimates the backing-buffer size for one
// RerankResponse so the encoder allocates once.
func rerankResponseSize(resp RerankResponse) int {
	size := 4 // braces + slack
	size += 11 + len(resp.Object)
	size += 10 + len(resp.Model)
	size += 12 // "results":[]
	for _, score := range resp.Results {
		// {"index":N,"score":0.xx,"text":"..."} — score typically
		// in 0..1, 4-6 ASCII chars; text is the dominant variable.
		size += 12 + len(score.Text)
		if score.Index != 0 {
			size += 9 + 12 // "index":N
		}
		if score.Score != 0 {
			size += 9 + 12 // "score":0.xx
		}
		if len(score.Labels) > 0 {
			size += 12
			for k, v := range score.Labels {
				size += 6 + len(k) + len(v)
			}
		}
	}
	return size
}
