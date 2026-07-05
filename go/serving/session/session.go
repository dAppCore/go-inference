// SPDX-Licence-Identifier: EUPL-1.2

// Package session is the inference-stack conversation registry behind the Responses
// API (RFC §6.10). It maps a `previous_response_id` back to the prior context so
// a caller continues a conversation WITHOUT resending the transcript (0% replay).
//
// It is NOT the KV cache. The real key/value state lives in go-mlx's Wake/Sleep
// engine (mlx RFC §7); here it is an opaque StateHandle the runtime attaches to
// a session — the inference stack routes and remembers position, go-mlx holds the weights and
// blocks. Keep this package free of model maths.
//
//	m := session.NewManager(session.NewMemoryStore())
//	s := m.Open("lemma")                                  // fresh session id
//	resp, _ := m.Append(s.ID, chat.Message{Role: chat.User, Content: []chat.ContentBlock{chat.Text("hello")}})
//	// next request carries previous_response_id = resp:
//	prior, _ := m.Continue(resp)                          // resolves s + its turns
package session

import (
	core "dappco.re/go"
	chat "dappco.re/go/inference/serving/chat"
)

// Session is one stateful conversation in the registry (RFC §6.10). Turns are
// the canonical chat messages (pkg/chat), ordered oldest→newest; StateHandle is
// the opaque reference to the go-mlx KV state for this conversation (empty until
// the runtime attaches one).
//
// A Session is a value: Manager hands back copies, so a caller never mutates the
// stored conversation by holding a reference (Turns is defensively copied).
type Session struct {
	ID          string         `json:"id"`
	Model       string         `json:"model"`
	Turns       []chat.Message `json:"turns"`
	StateHandle string         `json:"state_handle,omitempty"` // opaque go-mlx KV reference
	Created     core.Time      `json:"created"`
	Updated     core.Time      `json:"updated"`
}

// clone returns a deep copy so stored state can't be mutated through a returned
// value (the Turns slice is the only reference-typed field).
func (s Session) clone() Session {
	if s.Turns != nil {
		turns := make([]chat.Message, len(s.Turns))
		copy(turns, s.Turns)
		s.Turns = turns
	}
	return s
}
