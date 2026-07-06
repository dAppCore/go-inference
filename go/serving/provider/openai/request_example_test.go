// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	"strings"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func ExampleDecodeRequest() {
	body := strings.NewReader(`{"model":"qwen","messages":[{"role":"user","content":"hi"}]}`)

	req, err := DecodeRequest(body)
	if err != nil {
		core.Println(err)
		return
	}

	core.Println(req.Model)
	core.Println(len(req.Messages))
	// Output:
	// qwen
	// 1
}

func ExampleValidateRequest() {
	req := ChatCompletionRequest{Model: "qwen", Messages: []ChatMessage{{Role: "user", Content: "hi"}}}

	err := ValidateRequest(req)

	core.Println(err)
	// Output:
	// <nil>
}

func ExampleGenerateOptions() {
	req := ChatCompletionRequest{Model: "qwen", Messages: []ChatMessage{{Role: "user", Content: "hi"}}}

	opts, err := GenerateOptions(req)
	if err != nil {
		core.Println(err)
		return
	}

	cfg := inference.ApplyGenerateOpts(opts)
	core.Println(cfg.Temperature)
	// Output:
	// 1
}

func ExampleNormalizeStopSequences() {
	stops, err := NormalizeStopSequences(StopList{" END ", "STOP"})
	if err != nil {
		core.Println(err)
		return
	}

	core.Println(stops[0], stops[1])
	// Output:
	// END STOP
}
