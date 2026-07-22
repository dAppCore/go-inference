// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"bytes"
	"context"
	"testing"
)

// TestRunSSDCommand_Rejects covers the required-flag guard that stops a
// self-distillation run before train.RunSSDCommand loads the frozen base:
// --model and --data are both mandatory, an unknown flag fails at parse, -h
// exits 0, and --dataset without --checkpoint-dir is rejected before any
// sampling starts (the dataset tap reads the checkpoint dir's capture
// sidecar — see ingestSSDTrace in capture.go — so there is nothing it could
// honestly ingest without one). The both-present path loads a real model and
// is left to the train package's own suite; the dataset tap itself is
// covered directly by TestIngestSSDTrace_Good/_Bad in capture_test.go.
func TestRunSSDCommand_Rejects(t *testing.T) {
	for _, tc := range []struct {
		name     string
		args     []string
		wantCode int
	}{
		{"no flags", nil, 2},
		{"model without data", []string{"--model", "/some/base"}, 2},
		{"data without model", []string{"--data", "/some/prompts.jsonl"}, 2},
		{"unknown flag", []string{"--nonsense"}, 2},
		{"help", []string{"--help"}, 0},
		{"dataset without checkpoint-dir", []string{"--model", "/some/base", "--data", "/some/prompts.jsonl", "--dataset", "some-slug"}, 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			if code := runSSDCommand(context.Background(), tc.args, &stdout, &stderr); code != tc.wantCode {
				t.Fatalf("exit %d, want %d; stderr=%s", code, tc.wantCode, stderr.String())
			}
		})
	}
}
