// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func ExampleNewScheduledModel() {
	model, _ := NewScheduledModel(&schedulerFakeTextModel{tokens: []inference.Token{{Text: "ok"}}}, SchedulerConfig{QueueSize: 1})
	defer model.Close()

	_, stream, _ := model.Schedule(context.Background(), inference.ScheduledRequest{Prompt: "hello"})
	for token := range stream {
		core.Println(token.Token.Text)
	}
	// Output: ok
}
