// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"bytes"
	"context"
	"testing"
)

// TestRunSFTCommand_Rejects covers the required-flag guard that stops an SFT run
// before train.RunSFTCommand loads a model: --model and --data are both
// mandatory, so an empty either fails with exit 2, an unknown flag fails at
// parse, and -h exits 0. The both-present path loads a real model and is left
// to the train package's own suite.
func TestRunSFTCommand_Rejects(t *testing.T) {
	for _, tc := range []struct {
		name     string
		args     []string
		wantCode int
	}{
		{"no flags", nil, 2},
		{"model without data", []string{"-model", "/some/model"}, 2},
		{"data without model", []string{"-data", "/some/train.jsonl"}, 2},
		{"unknown flag", []string{"-nonsense"}, 2},
		{"help", []string{"-h"}, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			if code := runSFTCommand(context.Background(), tc.args, &stdout, &stderr); code != tc.wantCode {
				t.Fatalf("exit %d, want %d; stderr=%s", code, tc.wantCode, stderr.String())
			}
		})
	}
}
