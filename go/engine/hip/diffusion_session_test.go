// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"math"
	"testing"

	"dappco.re/go/inference"
)

type rocmDiffusionSessionTestDouble struct {
	cacheOffset int
	prefills    [][]int32
	requests    []ROCmDiffusionDenoiseRequest
	truncates   []int
	commits     [][]int32
}

func (s *rocmDiffusionSessionTestDouble) PrefillTokens(ids []int32) (int, error) {
	s.prefills = append(s.prefills, append([]int32(nil), ids...))
	return len(ids), nil
}

func (s *rocmDiffusionSessionTestDouble) CacheOffset() int {
	if s.cacheOffset > 0 {
		return s.cacheOffset
	}
	return 1
}

func (s *rocmDiffusionSessionTestDouble) Denoise(_ context.Context, req ROCmDiffusionDenoiseRequest) (ROCmDiffusionStepResult, error) {
	s.requests = append(s.requests, req)
	return ROCmDiffusionStepResult{
		Canvas:      []int32{7, 9},
		Greedy:      []int32{7, 9},
		MeanEntropy: 0,
	}, nil
}

func (s *rocmDiffusionSessionTestDouble) TruncateTo(pos int) error {
	s.truncates = append(s.truncates, pos)
	return nil
}

func (s *rocmDiffusionSessionTestDouble) CommitTokens(ids []int32) error {
	s.commits = append(s.commits, append([]int32(nil), ids...))
	return nil
}

func TestRunROCmDiffusionGenerate_Good(t *testing.T) {
	session := &rocmDiffusionSessionTestDouble{}
	var yielded []int32
	metrics, err := RunROCmDiffusionGenerate(context.Background(), ROCmDiffusionGenerateConfig{
		CanvasLength:        2,
		MaxSteps:            2,
		StabilityThreshold:  1,
		ConfidenceThreshold: 0.005,
		MaxCanvases:         2,
		TextVocabSize:       16,
		StopTokens:          []int32{9},
		Seed:                7,
	}, session, []int32{1}, func(id int32) bool {
		yielded = append(yielded, id)
		return true
	})
	if err != nil {
		t.Fatalf("RunROCmDiffusionGenerate: %v", err)
	}
	if metrics.Canvases != 1 || metrics.TotalSteps != 2 || metrics.EmittedTokens != 1 || !metrics.StoppedOnToken {
		t.Fatalf("metrics = %+v, want one stopped canvas with two denoise steps and one emitted token", metrics)
	}
	if len(session.prefills) != 1 || len(session.prefills[0]) != 1 || session.prefills[0][0] != 1 {
		t.Fatalf("prefills = %#v, want prompt [1]", session.prefills)
	}
	if len(session.requests) != 2 {
		t.Fatalf("denoise requests = %d, want 2", len(session.requests))
	}
	for _, req := range session.requests {
		if req.Prefix != 1 || len(req.Canvas) != 2 || len(req.GlobalMask) != 6 || len(req.LocalMask) != 6 {
			t.Fatalf("denoise request = %+v, want prefix-1 canvas and 2x3 masks", req)
		}
	}
	if len(session.truncates) != 2 || session.truncates[0] != 1 || session.truncates[1] != 1 {
		t.Fatalf("truncates = %v, want [1 1]", session.truncates)
	}
	if len(session.commits) != 1 || len(session.commits[0]) != 1 || session.commits[0][0] != 7 {
		t.Fatalf("commits = %#v, want [[7]]", session.commits)
	}
	if len(yielded) != 1 || yielded[0] != 7 {
		t.Fatalf("yielded = %v, want [7]", yielded)
	}
}

func TestRunROCmDiffusionGenerate_Bad(t *testing.T) {
	_, err := RunROCmDiffusionGenerate(context.Background(), ROCmDiffusionGenerateConfig{TextVocabSize: 16}, nil, []int32{1}, nil)
	if err == nil {
		t.Fatal("RunROCmDiffusionGenerate(nil session) error = nil")
	}
}

func TestRunROCmDiffusionGenerate_Ugly(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	session := &rocmDiffusionSessionTestDouble{}
	_, err := RunROCmDiffusionGenerate(ctx, ROCmDiffusionGenerateConfig{TextVocabSize: 16}, session, []int32{1}, nil)
	if err == nil {
		t.Fatal("RunROCmDiffusionGenerate(cancelled context) error = nil")
	}
	if len(session.prefills) != 0 {
		t.Fatalf("prefills = %#v, want no work after cancellation", session.prefills)
	}
}

type rocmDiffusionKernelTestDouble struct {
	hipKernelStub
	session *rocmDiffusionSessionTestDouble
}

func (k rocmDiffusionKernelTestDouble) OpenROCmDiffusionSession(context.Context, *hipLoadedModel) (ROCmDiffusionSession, error) {
	return k.session, nil
}

func TestROCmModelGenerateBlockDiffusionTokens_Good(t *testing.T) {
	session := &rocmDiffusionSessionTestDouble{}
	loaded := &hipLoadedModel{
		kernels: rocmDiffusionKernelTestDouble{session: session},
		modelInfo: inference.ModelInfo{
			Architecture: "diffusion_gemma",
			VocabSize:    16,
		},
	}
	model := &rocmModel{native: loaded, modelInfo: loaded.modelInfo, modelLabels: map[string]string{
		"diffusion_canvas_length":       "2",
		"diffusion_default_max_steps":   "2",
		"diffusion_stability_threshold": "1",
	}}
	metrics, err := model.GenerateBlockDiffusionTokens(context.Background(), []int32{1}, ROCmBlockDiffusionOptions{MaxTokens: 1, StopTokens: []int32{9}}, nil)
	if err != nil {
		t.Fatalf("GenerateBlockDiffusionTokens: %v", err)
	}
	if metrics.EmittedTokens != 1 || len(session.prefills) != 1 || len(session.commits) != 1 {
		t.Fatalf("dispatch metrics/session = %+v/%#v, want reachable loaded-model diffusion path", metrics, session)
	}
}

func TestROCmModelGenerateBlockDiffusionTokens_DefaultCanvas_Ugly(t *testing.T) {
	session := &rocmDiffusionSessionTestDouble{}
	loaded := &hipLoadedModel{
		kernels: rocmDiffusionKernelTestDouble{session: session},
		modelInfo: inference.ModelInfo{
			Architecture: "diffusion_gemma",
			VocabSize:    256,
		},
	}
	model := &rocmModel{native: loaded, modelInfo: loaded.modelInfo}
	_, err := model.GenerateBlockDiffusionTokens(context.Background(), []int32{1}, ROCmBlockDiffusionOptions{MaxTokens: 128, StopTokens: []int32{9}}, nil)
	if err != nil {
		t.Fatalf("GenerateBlockDiffusionTokens: %v", err)
	}
	if len(session.requests) == 0 || len(session.requests[0].Canvas) != rocmDiffusionDefaultCanvasLength {
		t.Fatalf("default canvas length = %d, want %d", len(session.requests[0].Canvas), rocmDiffusionDefaultCanvasLength)
	}
}

func TestROCmModelGenerateBlockDiffusionTokens_RespectsTokenBudget_Ugly(t *testing.T) {
	session := &rocmDiffusionSessionTestDouble{}
	loaded := &hipLoadedModel{
		kernels: rocmDiffusionKernelTestDouble{session: session},
		modelInfo: inference.ModelInfo{
			Architecture: "diffusion_gemma",
			VocabSize:    16,
		},
	}
	model := &rocmModel{native: loaded, modelInfo: loaded.modelInfo, modelLabels: map[string]string{
		"diffusion_canvas_length":       "2",
		"diffusion_default_max_steps":   "2",
		"diffusion_stability_threshold": "1",
	}}
	metrics, err := model.GenerateBlockDiffusionTokens(context.Background(), []int32{1}, ROCmBlockDiffusionOptions{MaxTokens: 1}, nil)
	if err != nil {
		t.Fatalf("GenerateBlockDiffusionTokens: %v", err)
	}
	if metrics.EmittedTokens != 1 || len(session.commits) != 1 || len(session.commits[0]) != 1 {
		t.Fatalf("budgeted metrics/commits = %+v/%#v, want exactly one committed token", metrics, session.commits)
	}
}

func TestROCmModelGenerateBlockDiffusionTokens_UsesAttentionWindow_Good(t *testing.T) {
	session := &rocmDiffusionSessionTestDouble{cacheOffset: 3}
	loaded := &hipLoadedModel{
		kernels: rocmDiffusionKernelTestDouble{session: session},
		modelInfo: inference.ModelInfo{
			Architecture: "diffusion_gemma",
			VocabSize:    16,
		},
	}
	model := &rocmModel{native: loaded, modelInfo: loaded.modelInfo, modelLabels: map[string]string{
		"diffusion_canvas_length":       "2",
		"diffusion_default_max_steps":   "2",
		"diffusion_stability_threshold": "1",
		"attention_sliding_window":      "1",
	}}
	_, err := model.GenerateBlockDiffusionTokens(context.Background(), []int32{1}, ROCmBlockDiffusionOptions{MaxTokens: 1, StopTokens: []int32{9}}, nil)
	if err != nil {
		t.Fatalf("GenerateBlockDiffusionTokens: %v", err)
	}
	if len(session.requests) == 0 || !math.IsInf(float64(session.requests[0].LocalMask[0]), -1) || !math.IsInf(float64(session.requests[0].LocalMask[1]), -1) {
		t.Fatalf("local mask = %v, want prefix rows outside the one-token window blocked", session.requests[0].LocalMask)
	}
}
