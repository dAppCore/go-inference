// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"os"
	"testing"

	"dappco.re/go/inference/model"
	_ "dappco.re/go/inference/model/arch/Qwen/qwen35" // registers qwen3_5/qwen3_5_moe — the ids moved out of composed's own spec (#18); the real-checkpoint chain tests load through the registry like the binary does
	"dappco.re/go/inference/model/composed"
)

// TestComposedChainHeadFoldDeviceVsHost is the #18 head fold's parity gate on the REAL chainable
// checkpoint (the 0.8B OptiQ fixture — its packed layers ride the whole-token chain): a chained
// forward must now set PendingHeadLogits (before the fold the chain path NEVER did — the stepper
// paid a separate headLogits command buffer + wait per token), and those logits must match
// composed.HeadLogitsHost's pure-host reference over the SAME final hidden, argmax included.
// Tolerance is the sibling head-fuse test's f32 band plus the fold's bf16 logits staging.
func TestComposedChainHeadFoldDeviceVsHost(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set — composed chain head fold")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("Metal runtime unavailable — composed chain head fold: %v", err)
	}
	dir := os.Getenv("LTHN_COMPOSED_AB_MODEL")
	if dir == "" {
		dir = composedPrefillABDefaultDir
	}
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("composed chain checkpoint absent (%s)", dir)
	}
	if composed.ComposedChainHeadDevice == nil || composed.ComposedChainTakeLogits == nil {
		t.Fatal("native init did not wire the composed chain head hooks")
	}
	tm, _, err := model.LoadComposedDir(dir)
	if err != nil {
		t.Fatalf("LoadComposedDir(%s): %v", dir, err)
	}
	ctm, ok := tm.(*composed.ComposedTokenModel)
	if !ok {
		t.Fatalf("loaded model is %T, want *composed.ComposedTokenModel", tm)
	}
	m := ctm.Model()

	prompt := []int32{16, 53, 90, 127}
	sess := composed.NewSession(m)
	y, err := sess.Forward(prompt)
	if err != nil {
		t.Fatalf("chained forward: %v", err)
	}
	devLogits := sess.PendingHeadLogits()
	if devLogits == nil {
		t.Fatal("chained forward left PendingHeadLogits nil — the head fold never engaged")
	}
	if len(devLogits) != m.Vocab {
		t.Fatalf("fold logits length: got %d want %d", len(devLogits), m.Vocab)
	}

	last := y[(len(prompt)-1)*m.D:]
	hostLogits := composed.HeadLogitsHost(m, last)
	devArg, hostArg := 0, 0
	for i := range devLogits {
		if devLogits[i] > devLogits[devArg] {
			devArg = i
		}
		if hostLogits[i] > hostLogits[hostArg] {
			hostArg = i
		}
		if math.Abs(float64(devLogits[i]-hostLogits[i])) > 2e-2*(1+math.Abs(float64(hostLogits[i]))) {
			t.Fatalf("logits[%d]: fold %v host %v (chain head fold diverged)", i, devLogits[i], hostLogits[i])
		}
	}
	if devArg != hostArg {
		t.Fatalf("argmax: fold %d host %d", devArg, hostArg)
	}
	t.Logf("chain head fold: %d logits within tolerance, argmax %d agrees with host", len(devLogits), devArg)
}
