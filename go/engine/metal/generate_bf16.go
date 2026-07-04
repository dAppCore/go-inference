// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// GenerateBF16 is the autoregressive token loop on an assembled bf16 model — the
// whole chain end to end: embed the ids → DecodeForward (the norm-faithful arch decode,
// behind model.Backend) → LM head on the last hidden state → greedy argmax → append,
// until maxNew tokens or eosID (eosID < 0 disables early stop). Returns the generated ids
// (excluding the prompt).
//
// Whole-sequence today: each step re-decodes the full running sequence over a fresh cache
// (correct, but O(N²) — incremental single-token decode with a persistent cache is the
// efficiency follow-up the model.Backend doc flags). Greedy/deterministic — the right shape
// for a tok/s bench; a sampled variant can layer model.Sampler on the same logits. Every
// model-specific number arrives DECLARED on the arch (embed scale, SDPA scale, eps,
// soft-cap) — the loop assumes nothing about which model it is running.
func GenerateBF16(g *BF16Model, arch model.Arch, promptIDs []int32, maxNew, maxLen, eosID int) ([]int32, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if g == nil || len(g.Layers) != len(arch.Layer) {
		return nil, core.NewError("native.GenerateBF16: weights/arch layer count mismatch")
	}
	if len(promptIDs) == 0 {
		return nil, core.NewError("native.GenerateBF16: empty prompt")
	}
	if maxNew <= 0 {
		return nil, core.NewError("native.GenerateBF16: maxNew must be > 0")
	}
	if len(promptIDs)+maxNew > maxLen {
		return nil, core.NewError("native.GenerateBF16: prompt + maxNew exceeds maxLen cache rows")
	}
	embedScale := embedScaleOf(arch) // the model-declared embedding multiplier (gemma-family √hidden)
	attnScale := attnScaleOf(arch)   // the model-declared SDPA scale (gemma4 1.0, not 1/√headDim)

	gen := make([]int32, 0, maxNew)
	var genErr error
	withAutoreleasePool(func() {
		// build the resident decode state ONCE; the KV caches persist across stepToken
		// calls within this pool, so each token costs one step (O(1)), not a re-decode.
		lb, moeWeights, berr := buildBF16ArchLayerBufs(g.Layers, arch.Layer, arch.Hidden, arch.Heads, arch.KVHeads, arch.HeadDim, arch.FF, maxLen, arch.SlidingWindow, nil) // in-memory weights → upload copies
		if berr != nil {
			genErr = berr
			return
		}
		state := newArchDecodeState(arch.Layer, lb, moeWeights, arch.Hidden, arch.Heads, arch.KVHeads, arch.HeadDim, arch.FF, arch.SlidingWindow, arch.RotaryDim, arch.RotaryDimLocal, arch.RopeBase, arch.RopeLocalBase, attnScale, arch.Eps, arch.ValueNorm, maxLen)
		defer state.Close()

		// step one token id at pos (embed is a pure-host gather; stepToken is the device step).
		emb := make([]byte, arch.Hidden*bf16Size)
		step := func(id int32, pos int) ([]byte, error) {
			_, err := embedTokenBF16Into(emb, g.Embed, id, arch.Vocab, arch.Hidden, embedScale)
			if err != nil {
				return nil, err
			}
			return state.stepToken(emb, pos)
		}

		// prefill the prompt over the growing cache; keep the last token's hidden state.
		var hidden []byte
		for p := 0; p < len(promptIDs); p++ {
			if hidden, genErr = step(promptIDs[p], p); genErr != nil {
				return
			}
		}
		// decode: head → greedy → append → step the new token at the next position.
		logits := make([]byte, arch.Vocab*bf16Size)
		for len(gen) < maxNew {
			logits, err := LMHeadBF16Into(logits, hidden, g.FinalNorm, g.LMHead, arch.Hidden, arch.Vocab, arch.Eps, arch.SoftCap)
			if err != nil {
				genErr = err
				return
			}
			next, err := model.Greedy(logits, arch.Vocab)
			if err != nil {
				genErr = err
				return
			}
			gen = append(gen, next)
			if (eosID >= 0 && int(next) == eosID) || len(gen) == maxNew {
				break
			}
			if hidden, genErr = step(next, len(promptIDs)+len(gen)-1); genErr != nil {
				return
			}
		}
	})
	return gen, genErr
}
