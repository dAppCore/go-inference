// SPDX-Licence-Identifier: EUPL-1.2

// Package dflash is the DFlash speculative drafter's contract — the block-parallel
// twin of pkg/ngram. DFlash (arXiv 2602.06036, "Block Diffusion for Flash
// Speculative Decoding") drafts with a small block-diffusion model that proposes a
// whole BLOCK of tokens in a single forward pass — conditioned on fused hidden
// states from several layers of the target ("verifier") model — which the target
// then verifies with the ordinary greedy prefix-accept rule. The real proposal
// comes from a diffusion forward the metal engine does not yet run (see
// docs/design-dflash.md for the evidenced engine gap); this package ships the
// parts that DON'T need a GPU, model-free and testable exactly as ngram is:
//
//   - the checkpoint contract — recognise a DFlash drafter and read its block/
//     verifier-layer parameters (ParseConfig);
//   - the proposer seam — BlockProposer, the interface the engine's diffusion
//     draft forward will implement, with a model-free LookupProposer stand-in so
//     the contract is exercisable and lands a real accept-rate on structured text;
//   - the verify driver — AcceptBlock / Generate, the greedy block-verification
//     whose every committed token is the target's own, so the emitted sequence is
//     byte-identical to plain autoregression WHATEVER the drafter proposes. That
//     identity is DFlash's losslessness, and it is the executable specification the
//     engine's verify path must honour.
//
// The whole flow, drafter proposes and target verifies, needing no draft model:
//
//	p := dflash.NewLookupProposer(dflash.Config{BlockSize: 8})
//	// next is the target's greedy argmax oracle (one real forward yields a block
//	// of these at once; per-call here is identical for greedy verification).
//	out, stats := dflash.Generate(prompt, 128, p, next)
//	// out == dflash.Autoregress(prompt, 128, next) — lossless by construction.
//	_ = stats.AcceptRate() // drafter tokens the target kept / tokens offered
package dflash

import core "dappco.re/go"

// Config carries the drafter-facing fields of a real DFlash speculator checkpoint
// (config.json with speculators_model_type "dflash"): the block the drafter
// proposes per parallel forward, the verifier layers whose hidden states are fused
// to condition it, and the target ("verifier") model the drafter was trained
// against. BlockSize is clamped to ≥ 1 so a zero Config is a usable single-token
// drafter rather than a dead one; the hidden-layer ids and verifier are metadata
// the loader and the serve notice reason about (the pure-Go verify driver needs
// only BlockSize).
//
//	dflash.Config{BlockSize: 8, AuxHiddenLayerIDs: []int{3, 13, 23, 32, 42}}
type Config struct {
	BlockSize         int    // γ — candidate tokens proposed per block (clamped ≥ 1); real default 8
	AuxHiddenLayerIDs []int  // verifier layer ids fused into the drafter's KV conditioning
	Verifier          string // the target model the drafter verifies against (speculators_config.verifier.name)
}

// dflashConfigJSON is the parse shape of a DFlash checkpoint's config.json — only
// the fields the drafter contract needs. speculators_model_type is the marker;
// block_size falls back to speculators_config.speculative_tokens when absent.
type dflashConfigJSON struct {
	SpeculatorsModelType string `json:"speculators_model_type"`
	BlockSize            int    `json:"block_size"`
	AuxHiddenStateLayers []int  `json:"aux_hidden_state_layer_ids"`
	SpeculatorsConfig    struct {
		SpeculativeTokens int `json:"speculative_tokens"`
		Verifier          struct {
			Name string `json:"name"`
		} `json:"verifier"`
	} `json:"speculators_config"`
}

// ParseConfig recognises a DFlash checkpoint from its config.json BYTES and reads
// the drafter contract. ok is true only when speculators_model_type == "dflash";
// any other model (or unparseable data) returns ok=false so a caller can fall
// through to the next drafter kind. It reads config only — never weights — the
// same posture as the reactive assistant/model config parsers.
//
//	if cfg, ok := dflash.ParseConfig(data); ok { /* a DFlash drafter */ }
func ParseConfig(data []byte) (Config, bool) {
	var raw dflashConfigJSON
	if r := core.JSONUnmarshal(data, &raw); !r.OK {
		return Config{}, false
	}
	if raw.SpeculatorsModelType != "dflash" {
		return Config{}, false
	}
	block := raw.BlockSize
	if block <= 0 {
		block = raw.SpeculatorsConfig.SpeculativeTokens
	}
	cfg := Config{
		BlockSize:         max(block, 1),
		AuxHiddenLayerIDs: append([]int(nil), raw.AuxHiddenStateLayers...),
		Verifier:          raw.SpeculatorsConfig.Verifier.Name,
	}
	return cfg, true
}

// BlockProposer proposes a block of candidate continuation tokens for a context.
// It is the seam the engine's block-diffusion draft forward implements: the real
// DFlash drafter denoises a masked block conditioned on fused target hidden
// states, but a proposer may use ANY rule — the verify driver's losslessness does
// not depend on the proposer being correct, only the accept-rate does.
type BlockProposer interface {
	// ProposeBlock returns up to BlockSize candidate token ids continuing context.
	// An empty return means "no proposal" — the target then decodes one token
	// itself, exactly as with an ngram miss.
	ProposeBlock(context []int) []int
}

// ProposerFunc adapts a plain function to a BlockProposer (the http.HandlerFunc
// pattern), so a test or a scripted engine can supply a proposal rule inline.
//
//	var p dflash.BlockProposer = dflash.ProposerFunc(func(ctx []int) []int { return nil })
type ProposerFunc func(context []int) []int

// ProposeBlock calls the wrapped function.
func (f ProposerFunc) ProposeBlock(context []int) []int { return f(context) }

// LookupProposer is the model-free reference drafter: it proposes a block by
// prompt-lookup over the context (the last occurrence of the trailing token, and
// the BlockSize tokens that followed it), standing in for the real diffusion
// forward so the block-verify contract is exercisable without a GPU. It lands a
// genuine accept-rate on repetitive text — the same "predict repeats for free"
// signal ngram gives — while proposing a fixed-shape BLOCK rather than a
// variable-length draft.
type LookupProposer struct {
	blockSize int
}

// NewLookupProposer builds the reference block drafter from a Config, clamping
// BlockSize up to 1 so it always proposes at least one token on a hit.
//
//	p := dflash.NewLookupProposer(dflash.Config{BlockSize: 8})
func NewLookupProposer(cfg Config) *LookupProposer {
	return &LookupProposer{blockSize: max(cfg.BlockSize, 1)}
}

// ProposeBlock proposes up to BlockSize tokens by prompt-lookup: it scans back for
// the most recent earlier occurrence of the context's trailing token and returns
// the tokens that followed it, capped at BlockSize. No earlier occurrence yields
// an empty block (the target then decodes normally). Deterministic — same
// context, same block, every time.
func (p *LookupProposer) ProposeBlock(context []int) []int {
	l := len(context)
	if l < 2 {
		return nil
	}
	last := context[l-1]
	// Most-recent earlier occurrence of the trailing token; the tokens after it
	// are the proposed continuation.
	for e := l - 2; e >= 0; e-- {
		if context[e] != last {
			continue
		}
		from := e + 1
		end := min(from+p.blockSize, l)
		if end <= from {
			return nil
		}
		out := make([]int, end-from)
		copy(out, context[from:end])
		return out
	}
	return nil
}

// NextToken is the target model's greedy next-token oracle: given the running
// prefix it returns the token the target would emit next (argmax of its logits).
// One real target forward over a K-token block yields all K of these at once; the
// per-position shape here is semantically identical for greedy verification and
// keeps the driver free of any engine.
type NextToken func(prefix []int) int

// AcceptBlock verifies one proposed block against the target and returns the tokens
// to commit plus how many DRAFT tokens were accepted. It applies the greedy
// speculative rule: walk the block while each proposed token equals the target's
// own next token; at the first divergence commit the target's correction token and
// stop; if the whole block is accepted, commit one bonus target token (the free
// token a fully-accepted verify pass yields). EVERY committed token is the
// target's own next token for the true running prefix — so the commit is
// byte-identical to plain greedy autoregression regardless of what the drafter
// proposed. That identity is DFlash's losslessness (TestAcceptBlock_Lossless...).
//
//	commit, accepted := dflash.AcceptBlock(prefix, proposed, next)
func AcceptBlock(prefix, proposed []int, next NextToken) (commit []int, acceptedDraft int) {
	seq := append([]int(nil), prefix...)
	for i := 0; i < len(proposed); i++ {
		t := next(seq)
		seq = append(seq, t)
		commit = append(commit, t)
		if proposed[i] != t {
			// Diverged: i draft tokens matched, t is the correction. Everything
			// past i in the block was conditioned on a wrong token — discard it.
			return commit, i
		}
	}
	// Whole block accepted → one free bonus token from the same verify pass.
	commit = append(commit, next(seq))
	return commit, len(proposed)
}

// Stats accumulates one speculative Generate run — the drafter's earn-its-keep
// receipt. It mirrors the vocabulary of inference.SpeculativeMetrics (proposed /
// accepted / rate) so a caller can bridge to the engine-neutral metrics surface.
type Stats struct {
	Rounds         int // speculative rounds run (blocks proposed)
	ProposedTokens int // draft tokens offered for verification across all rounds
	AcceptedTokens int // draft tokens the target accepted
	TargetCalls    int // target next-token evaluations (verify + correction + bonus)
}

// AcceptRate is accepted draft tokens over proposed draft tokens, in [0,1]; 0 when
// nothing was proposed (a run that never sped up). Feed it to specctl.Adaptive to
// size the next block by how well recent drafts landed.
//
//	c.Record(stats.ProposedTokens, stats.AcceptedTokens)
func (s Stats) AcceptRate() float64 {
	if s.ProposedTokens <= 0 {
		return 0
	}
	return float64(s.AcceptedTokens) / float64(s.ProposedTokens)
}

// Generate runs the DFlash speculative loop to at most maxTokens continuation
// tokens: each round the proposer offers a block, AcceptBlock verifies it against
// the target, and the accepted-plus-correction tokens extend the sequence. It
// returns the generated continuation (prompt excluded) and the run's Stats. The
// output equals Autoregress(prompt, maxTokens, next) token-for-token — Generate is
// lossless; the drafter changes only HOW FAST the target's own sequence is
// produced (fewer verify passes when drafts land), never WHICH tokens.
//
//	out, stats := dflash.Generate(prompt, 128, proposer, next)
func Generate(prompt []int, maxTokens int, p BlockProposer, next NextToken) ([]int, Stats) {
	var stats Stats
	if maxTokens <= 0 {
		return nil, stats
	}
	seq := append([]int(nil), prompt...)
	out := make([]int, 0, maxTokens)
	for len(out) < maxTokens {
		proposed := p.ProposeBlock(seq)
		if len(proposed) == 0 {
			// No draft — take one target token, exactly as a plain step.
			t := next(seq)
			seq = append(seq, t)
			out = append(out, t)
			stats.TargetCalls++
			continue
		}
		commit, accepted := AcceptBlock(seq, proposed, next)
		stats.Rounds++
		stats.ProposedTokens += len(proposed)
		stats.AcceptedTokens += accepted
		stats.TargetCalls += len(commit) // one next() per committed token
		for _, t := range commit {
			seq = append(seq, t)
			out = append(out, t)
			if len(out) >= maxTokens {
				break
			}
		}
	}
	if len(out) > maxTokens {
		out = out[:maxTokens]
	}
	return out, stats
}

// Autoregress is the reference plain greedy decode — the speculation-OFF baseline.
// It emits exactly maxTokens tokens, each the target's argmax for the running
// prefix. Generate must equal this token-for-token: it is the other half of the
// losslessness invariant.
//
//	base := dflash.Autoregress(prompt, 128, next)
func Autoregress(prompt []int, maxTokens int, next NextToken) []int {
	if maxTokens <= 0 {
		return nil
	}
	seq := append([]int(nil), prompt...)
	out := make([]int, 0, maxTokens)
	for i := 0; i < maxTokens; i++ {
		t := next(seq)
		seq = append(seq, t)
		out = append(out, t)
	}
	return out
}
