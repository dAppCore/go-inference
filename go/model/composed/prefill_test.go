// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"bytes"
	"testing"

	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/mamba2"
	"dappco.re/go/inference/model/qwen3"
	"dappco.re/go/inference/model/rwkv7"
)

// prefill_test.go proves model.BatchPrefillStepper's core invariant across EVERY mixer kind composed
// serves — the property composed's own token_model.go's composedStepper.PrefillBatch must hold regardless
// of which mixer(s) a checkpoint configures: batching an N-token prefill into ONE forwardEmb call must
// leave every layer's recurrent/KV state — and therefore every subsequent decode step and every generated
// token — byte-identical to the pre-existing per-token walk. Each mixer kind gets its own synthetic
// ComposedModel fixture (mirroring mkComposedModel/TestHybridDecodeEqualsPrefill's existing pattern) so a
// mixer-specific regression (e.g. attention's KV cache position, mamba2/rwkv7's conv+state threading)
// cannot hide behind another kind's coincidental correctness.

// mkMixerOnlyComposedModel builds an nLayers-deep ComposedModel where every layer's mixer comes from mk
// (called once per layer index so each layer's weights are distinctly seeded) — the single-mixer-kind
// counterpart to mkComposedModel's gated-delta-only construction, parametrised over the mixer family so
// every kind composed supports can run the same parity assertions below.
func mkMixerOnlyComposedModel(nLayers, D, vocab, FF int, mk func(li int) Mixer) *ComposedModel {
	layers := make([]Layer, nLayers)
	for li := range layers {
		layers[li] = Layer{
			InputNorm:    syn(D, li*13+1),
			Mixer:        mk(li),
			PostAttnNorm: syn(D, li*13+2),
			MLP:          &MLP{Gate: syn(FF*D, li*13+3), Up: syn(FF*D, li*13+4), Down: syn(D*FF, li*13+5), FF: FF},
		}
	}
	return &ComposedModel{
		Embed: syn(vocab*D, 100), Layers: layers, NormF: syn(D, 101), Output: nil,
		D: D, Vocab: vocab, Eps: 1e-5,
	}
}

// mkAttnOnlyComposedModel builds an all-full_attention ComposedModel (the "pure-attention" fixture).
func mkAttnOnlyComposedModel(nLayers, D, vocab, FF int) *ComposedModel {
	cfg := AttnConfig{Heads: 4, KVHeads: 2, HeadDim: 8, RotaryDim: 4, RopeTheta: 1e6, NormEps: 1e-6}
	return mkMixerOnlyComposedModel(nLayers, D, vocab, FF, func(li int) Mixer { return mkAttnMixer(cfg, D, li*13+20) })
}

// mkMamba2OnlyComposedModel builds an all-mamba2 ComposedModel.
func mkMamba2OnlyComposedModel(nLayers, D, vocab, FF int) *ComposedModel {
	cfg := mamba2.BlockConfig{NumHeads: 4, HeadDim: 4, StateDim: 4, NumGroups: 2, ConvKernel: 3, Eps: 1e-5}
	return mkMixerOnlyComposedModel(nLayers, D, vocab, FF, func(li int) Mixer { return mkMamba2Mixer(cfg, D, li*13+20) })
}

// mkRWKV7OnlyComposedModel builds an all-rwkv7 ComposedModel.
func mkRWKV7OnlyComposedModel(nLayers, D, vocab, FF int) *ComposedModel {
	cfg := rwkv7.BlockConfig{NumHeads: 2, KeyDim: 4, ValueDim: 4}
	return mkMixerOnlyComposedModel(nLayers, D, vocab, FF, func(li int) Mixer { return mkRWKV7Mixer(cfg, D, li*13+20) })
}

// mkHybridComposedModel interleaves gated-delta and full-attention layers (the Qwen 3.6 schedule shape) —
// the same construction TestHybridDecodeEqualsPrefill (attention_test.go) uses.
func mkHybridComposedModel(D, vocab, FF int) *ComposedModel {
	gdCfg := qwen3.GatedDeltaConfig{KeyHeads: 2, ValueHeads: 4, HeadDim: 8, ConvKernel: 4, Eps: 1e-5}
	atCfg := AttnConfig{Heads: 4, KVHeads: 2, HeadDim: 8, RotaryDim: 4, RopeTheta: 1e6, NormEps: 1e-6}
	mk := func(li int, mx Mixer) Layer {
		return Layer{
			InputNorm:    syn(D, li*13+1),
			Mixer:        mx,
			PostAttnNorm: syn(D, li*13+2),
			MLP:          &MLP{Gate: syn(FF*D, li*13+3), Up: syn(FF*D, li*13+4), Down: syn(D*FF, li*13+5), FF: FF},
		}
	}
	return &ComposedModel{
		Embed: syn(vocab*D, 100), NormF: syn(D, 101), D: D, Vocab: vocab, Eps: 1e-5,
		Layers: []Layer{
			mk(0, mkGatedDeltaMixer(gdCfg, D, 20)), // linear_attention
			mk(1, mkAttnMixer(atCfg, D, 40)),       // full_attention
			mk(2, mkGatedDeltaMixer(gdCfg, D, 60)), // linear_attention
			mk(3, mkAttnMixer(atCfg, D, 80)),       // full_attention
		},
	}
}

// prefillParityCase names one mixer-kind fixture both parity tests below run against.
type prefillParityCase struct {
	name string
	m    *ComposedModel
}

// prefillParityCases is the shared per-mixer-kind fixture table — one entry per mixer family composed
// dispatches (see composed.go's Mixer/projMixer), plus the interleaved hybrid schedule the orchestration
// exists to serve.
func prefillParityCases() []prefillParityCase {
	const D, vocab, FF = 8, 32, 16
	return []prefillParityCase{
		{"pure-attention", mkAttnOnlyComposedModel(3, D, vocab, FF)},
		{"gated-delta", mkComposedModel(3, D, vocab, FF)},
		{"mamba2", mkMamba2OnlyComposedModel(3, D, vocab, FF)},
		{"rwkv7", mkRWKV7OnlyComposedModel(3, D, vocab, FF)},
		{"hybrid", mkHybridComposedModel(D, vocab, FF)},
	}
}

// TestPrefillBatchStateEqualsPerTokenWalk is the byte-for-bit state-equality receipt: for every mixer
// kind, a session that prefills the WHOLE prompt via ONE PrefillBatch call must reach the identical last
// hidden AND leave every layer's recurrent/KV state such that subsequent decode steps — attention's cache
// position, gated-delta's conv+delta, mamba2's conv+ssm, rwkv7's [H,K,V] recurrence all feed the very next
// Step's math — produce byte-identical hiddens to a session that prefilled the SAME prompt one Step call
// per token (the pre-existing path). Any state divergence, however small, propagates into the next decode
// step's output, so matching decode outputs after the prefill boundary IS the state-equality proof.
func TestPrefillBatchStateEqualsPerTokenWalk(t *testing.T) {
	prompt := []int32{1, 5, 9, 2, 7, 3}
	decodeTail := []int32{4, 6, 8, 0}

	for _, tc := range prefillParityCases() {
		t.Run(tc.name, func(t *testing.T) {
			tm := NewTokenModel(tc.m)

			sessWalk, err := tm.OpenSession()
			if err != nil {
				t.Fatalf("OpenSession (per-token walk): %v", err)
			}
			var lastWalk []byte
			for _, id := range prompt {
				emb, err := tm.Embed(id)
				if err != nil {
					t.Fatalf("Embed: %v", err)
				}
				if lastWalk, err = sessWalk.Step(emb); err != nil {
					t.Fatalf("Step: %v", err)
				}
			}

			sessBatch, err := tm.OpenSession()
			if err != nil {
				t.Fatalf("OpenSession (batch): %v", err)
			}
			bp, ok := sessBatch.(model.BatchPrefillStepper)
			if !ok {
				t.Fatalf("%s: composedStepper does not implement model.BatchPrefillStepper", tc.name)
			}
			embs := make([][]byte, len(prompt))
			for i, id := range prompt {
				e, err := tm.Embed(id)
				if err != nil {
					t.Fatalf("Embed: %v", err)
				}
				embs[i] = e
			}
			lastBatch, err := bp.PrefillBatch(embs)
			if err != nil {
				t.Fatalf("PrefillBatch: %v", err)
			}

			if !bytes.Equal(lastWalk, lastBatch) {
				t.Fatalf("%s: last hidden after batch prefill != per-token walk", tc.name)
			}

			for _, id := range decodeTail {
				emb, err := tm.Embed(id)
				if err != nil {
					t.Fatalf("Embed: %v", err)
				}
				hWalk, err := sessWalk.Step(emb)
				if err != nil {
					t.Fatalf("sessWalk.Step: %v", err)
				}
				hBatch, err := sessBatch.Step(emb)
				if err != nil {
					t.Fatalf("sessBatch.Step: %v", err)
				}
				if !bytes.Equal(hWalk, hBatch) {
					t.Fatalf("%s: decode hidden diverged after batch prefill vs per-token walk (state not equal)", tc.name)
				}
			}
			t.Logf("%s: batch-prefill state == per-token-walk state, bit-exact through %d prefill + %d decode tokens", tc.name, len(prompt), len(decodeTail))
		})
	}
}

// noBatchTokenModel wraps a *ComposedTokenModel, stripping BatchPrefillStepper from the stepper
// OpenSession returns — forcing model.Generate's shared loop onto the per-token prefill walk that ran
// before this capability existed, so a test can compare that baseline against the batched-prefill path
// over the SAME model.
type noBatchTokenModel struct{ *ComposedTokenModel }

func (tm noBatchTokenModel) OpenSession() (model.DecodeStepper, error) {
	sess, err := tm.ComposedTokenModel.OpenSession()
	if err != nil {
		return nil, err
	}
	return noBatchStepper{sess.(*composedStepper)}, nil
}

// noBatchStepper forwards Step/Head but does NOT embed composedStepper directly (a struct embed would
// promote PrefillBatch too), so a model.BatchPrefillStepper type assertion on it fails and
// generateStepwiseWithSession falls back to one Step call per prompt token.
type noBatchStepper struct{ inner *composedStepper }

func (s noBatchStepper) Step(emb []byte) ([]byte, error)    { return s.inner.Step(emb) }
func (s noBatchStepper) Head(hidden []byte) ([]byte, error) { return s.inner.Head(hidden) }

var (
	_ model.SessionModel  = noBatchTokenModel{}
	_ model.DecodeStepper = noBatchStepper{}
	_ model.LMHead        = noBatchStepper{}
)

// TestPrefillBatchGreedyTokensUnchanged is the generation-level receipt: for every mixer kind, N greedy
// tokens from model.Generate over the batched-prefill path (the real composedStepper, which now
// implements model.BatchPrefillStepper) must be byte-identical to the SAME model's tokens from the forced
// per-token-walk path (noBatchTokenModel) — batching the prefill changes HOW the prompt reaches the
// cache, never WHAT gets generated.
func TestPrefillBatchGreedyTokensUnchanged(t *testing.T) {
	prompt := []int32{1, 5, 9, 2}
	const maxNew = 6

	for _, tc := range prefillParityCases() {
		t.Run(tc.name, func(t *testing.T) {
			gotBatch, err := model.Generate(NewTokenModel(tc.m), prompt, maxNew, -1)
			if err != nil {
				t.Fatalf("Generate (batched prefill): %v", err)
			}
			gotWalk, err := model.Generate(noBatchTokenModel{NewTokenModel(tc.m)}, prompt, maxNew, -1)
			if err != nil {
				t.Fatalf("Generate (per-token walk): %v", err)
			}
			if len(gotBatch) != len(gotWalk) {
				t.Fatalf("%s: token count %d != %d", tc.name, len(gotBatch), len(gotWalk))
			}
			for i := range gotBatch {
				if gotBatch[i] != gotWalk[i] {
					t.Fatalf("%s: token %d = %d, want %d (batched prefill changed generation)", tc.name, i, gotBatch[i], gotWalk[i])
				}
			}
			t.Logf("%s: %d greedy tokens identical whether prefill is batched or per-token: %v", tc.name, maxNew, gotBatch)
		})
	}
}
