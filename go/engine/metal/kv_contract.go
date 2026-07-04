// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"context"
	"iter"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/kv"
)

// kv_contract.go binds pkg/native's kv.Snapshot-native capture/restore machinery
// (session_kv_snapshot.go) to the engine-neutral inference KV-state contracts
// (external/go-inference kvstate.go). The native engine speaks kv.Snapshot
// directly — there is no per-engine snapshot type and nothing for the kvconv
// converter to do on this lane. The vehicles mirror pkg/metal:
//
//   - NativeTokenModel (the loaded decode model, holds the tokenizer) satisfies
//     inference.KVSnapshotter and inference.KVChunkSnapshotter — the string-prompt
//     capture entrypoints, like metal.Model.CaptureKV.
//   - ArchSession (the incremental decode session, "the model's cache") satisfies
//     inference.KVRestorer and inference.PromptCacheClearer — restore and clear,
//     like metal.ModelSession.
//
// Gap (reported, not stubbed): the model-level, string-prompt prompt-cache
// warmers (inference.PromptCacheWarmer / PromptCacheChunkWarmer) retain a warmed
// cache across calls. NativeTokenModel is stateless — sessions are caller-owned
// via OpenSession — so a model-level warmer would need a retained-session
// lifecycle this engine deliberately does not carry (the serve layer's
// nativeTextModel.cacheSess owns it). pkg/native exposes warming at the session
// level in token-id terms: ArchSession.WarmPromptCache([]int32) + ClearPromptCache().
var (
	_ inference.KVSnapshotter      = (*NativeTokenModel)(nil)
	_ inference.KVChunkSnapshotter = (*NativeTokenModel)(nil)
	_ inference.KVRestorer         = (*ArchSession)(nil)
	_ inference.PromptCacheClearer = (*ArchSession)(nil)
)

// kvCaptureOptionsFromInference lifts the engine-neutral capture options onto the
// kv package's capture options — the two carry the same fields (the kvconv
// ToMetalKVSnapshotCaptureOptions shim for the metal lane has no native analogue
// because native already captures in kv.Snapshot terms).
func kvCaptureOptionsFromInference(opts inference.KVSnapshotCaptureOptions) kv.CaptureOptions {
	return kv.CaptureOptions{RawKVOnly: opts.RawKVOnly, BlockStartToken: opts.BlockStartToken}
}

// CaptureKV prefills a text prompt into a fresh incremental session and returns
// the resulting KV state as a portable kv.Snapshot — the native side of
// inference.KVSnapshotter, the direct mirror of pkg/metal Model.CaptureKV. It
// needs a tokenizer (AttachTokenizer); without one the decode model works in
// token-id space only, so ArchSession.CaptureKV/CaptureKVWithOptions is the
// tokenizer-free session-level capture the serve boundary drives instead.
func (m *NativeTokenModel) CaptureKV(ctx context.Context, prompt string, opts inference.KVSnapshotCaptureOptions) (*kv.Snapshot, error) {
	if m == nil {
		return nil, core.NewError("native.NativeTokenModel.CaptureKV: nil model")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if m.tok == nil {
		return nil, core.NewError("native.NativeTokenModel.CaptureKV: no tokenizer attached")
	}
	return m.captureKVTokens(ctx, m.tok.Encode(prompt), opts)
}

// CaptureKVChunks prefills ordered text chunks (each tokenised in turn) and
// returns the KV state as a kv.Snapshot — the native side of
// inference.KVChunkSnapshotter. The chunk boundary is the tokeniser boundary:
// concatenating each chunk's ids is exactly "prefill the chunks in order".
func (m *NativeTokenModel) CaptureKVChunks(ctx context.Context, chunks iter.Seq[string], opts inference.KVSnapshotCaptureOptions) (*kv.Snapshot, error) {
	if m == nil {
		return nil, core.NewError("native.NativeTokenModel.CaptureKVChunks: nil model")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if m.tok == nil {
		return nil, core.NewError("native.NativeTokenModel.CaptureKVChunks: no tokenizer attached")
	}
	if chunks == nil {
		return nil, core.NewError("native.NativeTokenModel.CaptureKVChunks: nil chunks")
	}
	var ids []int32
	for chunk := range chunks {
		ids = append(ids, m.tok.Encode(chunk)...)
	}
	return m.captureKVTokens(ctx, ids, opts)
}

// captureKVTokens opens a transient incremental session, prefills the ids, and
// captures the resident KV cache as a kv.Snapshot. The snapshot owns its bytes
// (CaptureKVWithOptions copies each layer slab and the boundary logits), so it
// outlives the session Close. Closing the transient session is safe: an
// OpenSession session references — never owns — the model's mmap'd weights, so
// Close only frees its own decode scratch (see ArchSession.Close).
func (m *NativeTokenModel) captureKVTokens(ctx context.Context, ids []int32, opts inference.KVSnapshotCaptureOptions) (*kv.Snapshot, error) {
	if len(ids) == 0 {
		return nil, core.NewError("native.NativeTokenModel.CaptureKV: empty prompt after tokenisation")
	}
	stepper, err := m.OpenSession()
	if err != nil {
		return nil, err
	}
	sess, ok := stepper.(*ArchSession)
	if !ok {
		if closer, closeOK := stepper.(interface{ Close() error }); closeOK {
			_ = closer.Close()
		}
		return nil, core.NewError("native.NativeTokenModel.CaptureKV: session does not support KV capture")
	}
	defer func() { _ = sess.Close() }()
	if err := sess.PrefillTokens(ids); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return sess.CaptureKVWithOptions(kvCaptureOptionsFromInference(opts))
}

// RestoreFromKV loads a portable kv.Snapshot into the session's resident cache so
// the next generation continues from the snapshot instead of re-prefilling — the
// native side of inference.KVRestorer, the ctx-aware wrapper over RestoreKV. The
// snapshot is consumed directly in kv.Snapshot terms, no kvconv.
func (s *ArchSession) RestoreFromKV(ctx context.Context, snapshot *kv.Snapshot) error {
	if s == nil {
		return core.NewError("native.ArchSession.RestoreFromKV: nil session")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	return s.RestoreKV(snapshot)
}
