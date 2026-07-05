// SPDX-Licence-Identifier: EUPL-1.2

package engine

import (
	"context"
	"testing"

	"dappco.re/go/inference/kv"
	"dappco.re/go/inference/model"
)

// stateRestorer mirrors model/state/session.nativeSessionRestorer — the durable
// conversation session's unexported wake seam, which is RestoreKV(ctx, snapshot)
// rather than the inference.KVRestorer name RestoreFromKV. Pinning that
// *SessionHandle satisfies this exact shape guards the generate -state wake path:
// WakeAgentMemory type-asserts its handle to this interface, so a rename or
// removal of the alias would silently reintroduce the #291 wake failure
// ("native model session does not support KV restore").
type stateRestorer interface {
	RestoreKV(context.Context, *kv.Snapshot) error
}

var _ stateRestorer = (*SessionHandle)(nil)

// restoreRecordingSession is a minimal engine.Session that records the snapshot
// handed to RestoreFromKV, so the RestoreKV alias's delegation is observable.
type restoreRecordingSession struct {
	restored *kv.Snapshot
}

func (s *restoreRecordingSession) PrefillTokens([]int32) error { return nil }
func (s *restoreRecordingSession) AppendTokens([]int32) error  { return nil }
func (s *restoreRecordingSession) Pos() int                    { return 0 }
func (s *restoreRecordingSession) GenerateFromCacheEach(int, int, func(int32) bool) ([]int32, error) {
	return nil, nil
}
func (s *restoreRecordingSession) GenerateSampledFromCacheEach(int, []int32, *model.Sampler, model.SampleParams, model.TokenTransform, func(int32) bool) ([]int32, error) {
	return nil, nil
}
func (s *restoreRecordingSession) CaptureKVWithOptions(kv.CaptureOptions) (*kv.Snapshot, error) {
	return nil, nil
}
func (s *restoreRecordingSession) RangeKVBlocks(int, kv.CaptureOptions, func(kv.Block) (bool, error)) error {
	return nil
}
func (s *restoreRecordingSession) RestoreFromKV(_ context.Context, snapshot *kv.Snapshot) error {
	s.restored = snapshot
	return nil
}
func (s *restoreRecordingSession) Close() error { return nil }

// TestSessionHandleRestoreKVDelegatesToRestoreFromKV pins the #291 fix: the
// durable session's RestoreKV seam delegates to the engine restore, so waking a
// stored KV prefix through model/state/session reaches the engine rather than the
// unsupported-restore guard.
func TestSessionHandleRestoreKVDelegatesToRestoreFromKV(t *testing.T) {
	sess := &restoreRecordingSession{}
	handle := NewSessionHandle(&TextModel{}, sess)
	snap := &kv.Snapshot{Tokens: []int32{1, 2, 3}}
	if err := handle.RestoreKV(context.Background(), snap); err != nil {
		t.Fatalf("RestoreKV: %v", err)
	}
	if sess.restored != snap {
		t.Fatal("RestoreKV did not delegate the snapshot to the engine session")
	}
}
