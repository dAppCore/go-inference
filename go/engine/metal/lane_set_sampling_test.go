// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"context"
	"os"
	"slices"
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/inference/model"
)

// laneSetSampledCases are the sampled-discipline grid the lane oracle tests
// sweep: every case engages laneSampled (temperature / min-p / repeat-penalty,
// the classic serve path's non-greedy pick) with a distinct route through the
// sampled ladder.
func laneSetSampledCases() []struct {
	name string
	cfg  inference.SamplerConfig
	seed uint64
} {
	return []struct {
		name string
		cfg  inference.SamplerConfig
		seed uint64
	}{
		{"tempTopK", inference.SamplerConfig{Temperature: 0.8, TopK: 8}, 7},
		{"tempTopP", inference.SamplerConfig{Temperature: 1.1, TopP: 0.9}, 21},
		{"minP", inference.SamplerConfig{Temperature: 0.9, MinP: 0.1}, 3},
		{"repeatPenalty", inference.SamplerConfig{Temperature: 0.7, TopK: 12, RepeatPenalty: 1.3}, 11},
	}
}

// TestLaneSetSampledMatchesGenerateSampledEach pins the per-lane sampler
// against the classic sampled generate: a one-lane set decoding with
// LaneSpec.Sampler + SampleSeed must produce token-for-token the stream
// GenerateSampledEach yields for the same prompt, params, and seed — same
// route ladder, same one-draw-per-token RNG consumption (lane_set_sampling.go).
func TestLaneSetSampledMatchesGenerateSampledEach(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set — GPU decode fixture")
	}
	m := laneSetFixtureModel(t)
	defer m.Close()
	prompt := []int32{1, 5, 3, 2}
	const maxNew = 8
	ctx := context.Background()

	for _, tc := range laneSetSampledCases() {
		t.Run(tc.name, func(t *testing.T) {
			// Oracle: the classic sampled generate on its own fresh session.
			osess, err := m.OpenSession()
			if err != nil {
				t.Fatalf("OpenSession (oracle): %v", err)
			}
			oracle, ok := osess.(*ArchSession)
			if !ok {
				t.Fatalf("OpenSession returned %T, not *ArchSession", osess)
			}
			want, err := oracle.GenerateSampledEach(prompt, maxNew, nil, model.NewSampler(tc.seed), laneSampleParams(tc.cfg), nil, nil)
			_ = oracle.Close()
			if err != nil {
				t.Fatalf("GenerateSampledEach: %v", err)
			}
			if len(want) == 0 {
				t.Fatal("oracle produced no tokens")
			}

			lsi, err := m.OpenLaneSet(inference.LaneSetConfig{MaxLanes: 1})
			if err != nil {
				t.Fatalf("OpenLaneSet: %v", err)
			}
			ls := lsi.(*laneSet)
			defer ls.Close()
			h, err := ls.Prepare(ctx, inference.LaneSpec{PromptIDs: prompt, MaxNew: maxNew, Sampler: tc.cfg, SampleSeed: tc.seed})
			if err != nil {
				t.Fatalf("Prepare: %v", err)
			}
			var got []int32
			for {
				steps, err := ls.Step(ctx)
				if err != nil {
					t.Fatalf("Step: %v", err)
				}
				if len(steps) == 0 {
					break
				}
				for _, s := range steps {
					if s.Lane.ID == h.ID && s.HasToken {
						got = append(got, s.Token)
					}
				}
				retireTerminal(t, ls, steps)
			}
			if !slices.Equal(got, want) {
				t.Fatalf("sampled lane diverges from GenerateSampledEach: lane %v vs oracle %v", got, want)
			}
		})
	}
}

// TestLaneSetSampledMixedDisciplineByteIdentity extends the fold byte-identity
// receipt to MIXED-discipline sets: greedy and sampled lanes co-resident, the
// GEMM-armed set and the replay set advanced in lockstep. Hiddens must stay
// byte-identical (phase 2 is discipline-blind) and the token vectors equal
// (each lane's sampler is its own seeded RNG stream, deterministic across the
// two phase-2 implementations).
func TestLaneSetSampledMixedDisciplineByteIdentity(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set — GPU decode fixture")
	}
	m := laneSetFixtureModel(t)
	defer m.Close()
	specs := []inference.LaneSpec{
		{PromptIDs: []int32{1, 5, 3, 2}, MaxNew: 8}, // greedy
		{PromptIDs: []int32{7, 7, 1}, MaxNew: 8, Sampler: inference.SamplerConfig{Temperature: 0.8, TopK: 8}, SampleSeed: 5},
		{PromptIDs: []int32{2, 9, 4, 6, 8, 3}, MaxNew: 8, Sampler: inference.SamplerConfig{Temperature: 0.9, MinP: 0.1}, SampleSeed: 9},
		{PromptIDs: []int32{4}, MaxNew: 8, Sampler: inference.SamplerConfig{Temperature: 0.7, TopK: 12, RepeatPenalty: 1.3}, SampleSeed: 13},
	}
	ctx := context.Background()

	open := func(mode int) (*laneSet, []inference.LaneHandle) {
		lsi, err := m.OpenLaneSet(inference.LaneSetConfig{MaxLanes: len(specs)})
		if err != nil {
			t.Fatalf("OpenLaneSet(mode %d): %v", mode, err)
		}
		ls, ok := lsi.(*laneSet)
		if !ok {
			t.Fatalf("OpenLaneSet returned %T, not *laneSet", lsi)
		}
		ls.gemmMode = mode
		handles := make([]inference.LaneHandle, len(specs))
		for i, spec := range specs {
			if handles[i], err = ls.Prepare(ctx, spec); err != nil {
				t.Fatalf("Prepare(mode %d lane %d): %v", mode, i, err)
			}
		}
		return ls, handles
	}

	lsG, hG := open(1)
	lsR, hR := open(2)
	defer lsG.Close()
	defer lsR.Close()

	for step := 0; ; step++ {
		sg, err := lsG.Step(ctx)
		if err != nil {
			t.Fatalf("step %d GEMM Step: %v", step, err)
		}
		sr, err := lsR.Step(ctx)
		if err != nil {
			t.Fatalf("step %d replay Step: %v", step, err)
		}
		if len(sg) == 0 && len(sr) == 0 {
			break
		}
		for i := range specs {
			lg, lr := lsG.lanes[hG[i].ID], lsR.lanes[hR[i].ID]
			if lg == nil || lr == nil {
				continue
			}
			if !bytes.Equal(lg.hidden, lr.hidden) {
				t.Fatalf("step %d lane %d: post-stack hidden bytes differ between GEMM and replay under mixed disciplines", step, i)
			}
		}
		tokG := stepTokens(sg, hG)
		tokR := stepTokens(sr, hR)
		if !slices.Equal(tokG, tokR) {
			t.Fatalf("step %d: token vector diverges GEMM %v vs replay %v", step, tokG, tokR)
		}
		retireTerminal(t, lsG, sg)
		retireTerminal(t, lsR, sr)
	}

	if lsG.gemmFwdCount == 0 {
		t.Fatal("gemmFwdCount is zero — the GEMM forward never fired under mixed disciplines (proof vacuous)")
	}
	if lsR.gemmFwdCount != 0 {
		t.Fatalf("replay set ran %d GEMM forwards — it should be pure ICB replay", lsR.gemmFwdCount)
	}
}
