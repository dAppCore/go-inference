// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"iter"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/kv"
)

var (
	_ inference.KVSnapshotter      = (*rocmModel)(nil)
	_ inference.KVChunkSnapshotter = (*rocmModel)(nil)
)

type hipOptionCaptureSession interface {
	CaptureKVWithOptions(context.Context, kv.CaptureOptions) (*kv.Snapshot, error)
}

func hipKVOptions(opts inference.KVSnapshotCaptureOptions) kv.CaptureOptions {
	return kv.CaptureOptions{RawKVOnly: opts.RawKVOnly, BlockStartToken: opts.BlockStartToken}
}

// CaptureKV prefills a prompt in a transient shared HIP session and returns its
// portable retained state.
func (m *rocmModel) CaptureKV(ctx context.Context, prompt string, opts inference.KVSnapshotCaptureOptions) (*kv.Snapshot, error) {
	if m == nil {
		return nil, core.NewError("rocm.CaptureKV: nil model")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	session := m.NewSession()
	if session == nil {
		return nil, core.NewError("rocm.CaptureKV: shared HIP session is unavailable")
	}
	defer func() { _ = session.Close() }()
	if err := session.Prefill(ctx, prompt); err != nil {
		return nil, core.E("rocm.CaptureKV", "prefill prompt", err)
	}
	return captureHIPSessionKV(ctx, session, opts)
}

// CaptureKVChunks prefills ordered chunks in one transient shared HIP session,
// preserving each tokenizer boundary before capturing retained state.
func (m *rocmModel) CaptureKVChunks(ctx context.Context, chunks iter.Seq[string], opts inference.KVSnapshotCaptureOptions) (*kv.Snapshot, error) {
	if m == nil {
		return nil, core.NewError("rocm.CaptureKVChunks: nil model")
	}
	if chunks == nil {
		return nil, core.NewError("rocm.CaptureKVChunks: nil chunks")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	session := m.NewSession()
	if session == nil {
		return nil, core.NewError("rocm.CaptureKVChunks: shared HIP session is unavailable")
	}
	defer func() { _ = session.Close() }()
	prefilled := false
	for chunk := range chunks {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		ids, err := m.Tokenize(chunk)
		if err != nil {
			return nil, core.E("rocm.CaptureKVChunks", "tokenize chunk", err)
		}
		if len(ids) == 0 {
			continue
		}
		if !prefilled {
			err = session.Prefill(ctx, chunk)
			prefilled = err == nil
		} else {
			err = session.AppendPrompt(ctx, chunk)
		}
		if err != nil {
			return nil, core.E("rocm.CaptureKVChunks", "prefill chunk", err)
		}
	}
	if !prefilled {
		return nil, core.NewError("rocm.CaptureKVChunks: chunks produced no prompt tokens")
	}
	return captureHIPSessionKV(ctx, session, opts)
}

func captureHIPSessionKV(ctx context.Context, session inference.SessionHandle, opts inference.KVSnapshotCaptureOptions) (*kv.Snapshot, error) {
	capture, ok := session.(hipOptionCaptureSession)
	if !ok {
		return nil, core.NewError("rocm.CaptureKV: shared HIP session does not support capture options")
	}
	snapshot, err := capture.CaptureKVWithOptions(ctx, hipKVOptions(opts))
	if err != nil {
		return nil, core.E("rocm.CaptureKV", "capture retained state", err)
	}
	return snapshot, nil
}
