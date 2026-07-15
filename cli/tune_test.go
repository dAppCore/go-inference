// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"bytes"
	"context"
	"testing"
)

// TestRunTuneCommand_Rejects covers tune's two distinct pre-RunTune guards: an
// unexpected positional argument (tune takes flags only) and a missing --model
// both exit 2, an unknown flag fails at parse, and -h exits 0. The model-present
// path drives tune.RunTune against a real target and is left to the tune
// package's own suite.
func TestRunTuneCommand_Rejects(t *testing.T) {
	for _, tc := range []struct {
		name     string
		args     []string
		wantCode int
	}{
		{"no flags (missing model)", nil, 2},
		{"unexpected positional", []string{"--model", "/some/model", "stray-arg"}, 2},
		{"unknown flag", []string{"--nonsense"}, 2},
		{"help", []string{"--help"}, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			if code := runTuneCommand(context.Background(), tc.args, &stdout, &stderr); code != tc.wantCode {
				t.Fatalf("exit %d, want %d; stderr=%s", code, tc.wantCode, stderr.String())
			}
		})
	}
}
