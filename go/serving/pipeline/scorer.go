// SPDX-Licence-Identifier: EUPL-1.2

package pipeline

import (
	core "dappco.re/go"
	"dappco.re/go/inference/eval/score/lek"
	chat "dappco.re/go/inference/serving/chat"
)

// scorerAdapter maps the in-process lem-scorer (eval/score/lek) onto the Scorer
// seam (§6.6-adjacent). It scores the PAIR (latest user prompt, assistant
// response) with lek.ScorePair so the metadata carries the cross-text
// differential + authority signal — sycophancy relative to the prompt, not the
// response in isolation. Pure and non-blocking: it reads the turn and returns a
// bundle, it never touches the response text.
//
// The neural turn embedding is deliberately absent here: it rides from the
// decode return (serving.Result), not from a second model — until decode
// surfaces it, only the semantic score is recorded.
type scorerAdapter struct{}

// Score implements Scorer. It JSON-encodes the DiffResult under the "score"
// key. Returns nil (nothing to record) when both sides are empty, or when the
// encode fails — the score rides alongside, so a failure never disturbs the
// response.
func (scorerAdapter) Score(req chat.Request, resp chat.Response) map[string]string {
	var prompt string
	if m, ok := lastUser(req); ok {
		prompt = m.Text()
	}
	response := resp.Text
	if prompt == "" && response == "" {
		return nil
	}
	encoded := core.JSONMarshal(lek.ScorePair(prompt, response))
	if !encoded.OK {
		return nil
	}
	return map[string]string{"score": string(encoded.Value.([]byte))}
}
