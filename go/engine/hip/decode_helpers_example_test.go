// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func ExampleSpeculativeDecode() {
	target := &rocmModel{native: &fakeNativeModel{tokens: []inference.Token{{ID: 1}, {ID: 2}}}}
	draft := &rocmModel{native: &fakeNativeModel{tokens: []inference.Token{{ID: 1}, {ID: 9}}}}

	result, _ := SpeculativeDecode(context.Background(), target, draft, SpeculativeDecodeConfig{Prompt: "p", MaxTokens: 2})

	core.Println(result.Metrics.AcceptedTokens, result.Metrics.RejectedTokens)
	// Output: 1 1
}

func ExamplePromptLookupDecode() {
	target := &rocmModel{native: &fakeNativeModel{tokens: []inference.Token{{ID: 3}, {ID: 4}, {ID: 9}}}}

	result, _ := PromptLookupDecode(context.Background(), target, PromptLookupDecodeConfig{
		Prompt:       "p",
		MaxTokens:    3,
		LookupTokens: []int32{3, 4, 8},
	})

	core.Println(result.Metrics.AcceptedTokens, result.Metrics.RejectedTokens)
	// Output: 2 1
}

func ExampleAttachedDrafterDecode() {
	target := newDecodeGemma4E2BQ6Target(&fakeNativeModel{tokens: []inference.Token{{ID: 1}, {ID: 2}}})
	draft := newDecodeGemma4E2BBF16Assistant(&fakeNativeModel{tokens: []inference.Token{{ID: 1}, {ID: 9}}})

	result, _ := AttachedDrafterDecode(context.Background(), target, draft, AttachedDrafterDecodeConfig{Prompt: "p", MaxTokens: 2})

	core.Println(result.Metrics.AcceptedTokens, result.Metrics.RejectedTokens)
	// Output: 1 1
}
