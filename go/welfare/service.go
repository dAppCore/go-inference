// SPDX-Licence-Identifier: EUPL-1.2

// Package welfare is the guard layer between the user's chat input and the
// model (RFC.welfare). It detects hostile prompt shapes — slurs, sustained
// anger — and, rather than refusing or silently sanitising, opens a meta-
// session where the engine speaks to the model as a peer and lets the model
// decide how to handle it.
//
// detect.go is the DETECT half (RFC.welfare §1): score the user's latest
// message + the conversation's prior hostility, decide whether the mediation
// trigger fires. mediate.go is the MEDIATE half (§2 — the engine↔model session,
// lem_ok / lem_rephrase / lem_pause). guard.go composes them into the
// per-turn gate the chat runner calls.
//
// Detection is stateless: the chat runner hands in the full conversation each
// turn (WChat: "full message history in"), so sustained hostility is read off
// the prior user turns in the array — no per-session state to hold or leak.
package welfare

import (
	core "dappco.re/go"
	"dappco.re/go/inference/welfare/slurs"
)

// Config tunes the detector. Zero-value uses the RFC.welfare defaults; tunable
// per-deployment.
type Config struct {
	AngerThreshold     float64 // AngerScore above this is "elevated" (default 0.7)
	SustainedThreshold float64 // SustainedHostility above this gates anger (default 0.5)
	SustainedWindow    int     // prior user turns weighed for sustained hostility (default 4)
	AngerFloor         float64 // a prior turn counts toward sustained at/above this (default 0.4)
	// Hostility scores one text 0..1 (lem-runtime adaptation: wired to the
	// engine's /v1/score; nil = slur-only detection, works engine-down).
	Hostility func(string) float64
}

// Service is the welfare guard. Guard is the per-turn entry point; Detect is
// the read it builds on. Stateless — safe to share across goroutines.
type Service struct {
	cfg     Config
	matcher *slurs.Matcher
}

// New constructs the welfare Service over the curated slur catalogue, applying
// RFC.welfare defaults to any zero-value Config field.
//
//	w := welfare.New(welfare.Config{})
func New(cfg Config) *Service {
	if cfg.AngerThreshold == 0 {
		cfg.AngerThreshold = 0.7
	}
	if cfg.SustainedThreshold == 0 {
		cfg.SustainedThreshold = 0.5
	}
	if cfg.SustainedWindow == 0 {
		cfg.SustainedWindow = 4
	}
	if cfg.AngerFloor == 0 {
		cfg.AngerFloor = 0.4
	}
	return &Service{
		cfg:     cfg,
		matcher: slurs.Default(),
	}
}

// Register builds the welfare Service for core registration. The chat runner
// calls Guard per turn (ChatCtx → Guard → Detect + Mediate).
//
//	core.New(core.WithName("welfare", welfare.Register))
func Register(_ *core.Core) core.Result { return core.Ok(New(Config{})) }

// ServiceName is the Wails binding name.
func (s *Service) ServiceName() string { return "Welfare" }
