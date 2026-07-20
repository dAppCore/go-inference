// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"github.com/tmc/apple/metal"
)

// session_state_tq.go — the TurboQuant KIND of the session-state BLOCK codec
// (docs/design-tq-moe-hybrid.md): a TQ owner layer's snapshot payload is its
// RAW packed code rows + γ planes + bit widths, preserved byte-exact in
// transit and validated per layer on restore. This is the mixed-kind surface:
// one SessionStateBlock carries turboquant-codes layers beside untouched
// native bf16 layers (a gemma4 dense TQ session yields exactly that mix —
// sliding ring layers native, global owners TQ), and kind boundaries are
// enforced so raw codes can never land in a bf16 cache nor bf16 rows in a
// codes cache. The monolithic bf16-shaped codecs (SerializeState, CaptureKV)
// keep their wholesale TQ decline — they cannot represent codes.
//
// The mode string is DISTINCT from the legacy cross-engine "turboquant"
// vocabulary (session_kv_snapshot.go), which names bf16-shaped payloads and
// sits in nativeKVRestorableSourceCacheMode's pass-through allowlist — this
// kind must never ride that waiver.

// nativeStateCacheModeTurboQuantCodes marks a block-codec layer whose payload
// is raw TurboQuant code rows + γ planes (NOT bf16-shaped rows).
const nativeStateCacheModeTurboQuantCodes = "turboquant-codes"

// tqSnapshotCarrierBuffers resolves the session's armed TurboQuant carrier —
// the per-layer set plus the code cache buffers — from EITHER lane: the
// recorded-ICB carrier or the state-lane carrier (tq_kv_state.go). Both
// carriers store the identical archICBKVTQ shape, so the codec addresses
// rows uniformly. nil set when no carrier is armed.
func (s *ArchSession) tqSnapshotCarrierBuffers() (set *archICBKVTQ, kCaches, vCaches []metal.MTLBuffer) {
	if s == nil {
		return nil, nil, nil
	}
	if s.state.icb.hasKVTQ() {
		return s.state.icb.kvTQ, s.state.icb.kCaches, s.state.icb.vCaches
	}
	if s.state.tqStateArmed() {
		return &s.state.kvTQState.set, s.state.kvTQState.kCaches, s.state.kvTQState.vCaches
	}
	return nil, nil, nil
}

// tqStateLayerView builds the turboquant-codes view for owner layer li when a
// TurboQuant carrier holds it: raw code planes as keyBytes/valueBytes (K and V
// at their OWN strides), the γ planes, and the bit widths. isTQ=false when the
// layer is not a TQ owner (native view assembly proceeds unchanged). TQ owners
// are GLOBAL-only, so the view is always seq-major (maxSize 0), cacheRows =
// the session maxLen — exactly the geometry both carriers allocate.
func (s *ArchSession) tqStateLayerView(li int, spec model.LayerSpec) (sessionStateLayerView, bool, error) {
	set, kCaches, vCaches := s.tqSnapshotCarrierBuffers()
	if set == nil || !set.on(li) {
		return sessionStateLayerView{}, false, nil
	}
	if li >= len(kCaches) || li >= len(vCaches) || kCaches[li] == nil || vCaches[li] == nil ||
		li >= len(set.kGammas) || li >= len(set.vGammas) || set.kGammas[li] == nil || set.vGammas[li] == nil {
		return sessionStateLayerView{}, false, core.NewError("native.sessionState: TurboQuant carrier is missing a cache plane for an enabled layer")
	}
	rows := s.maxLen
	kBytes, vBytes := rows*set.kRowBytes[li], rows*set.vRowBytes[li]
	gBytes := rows * set.gammaRowBytes[li]
	if int(bufferLengthFast(kCaches[li])) < kBytes || int(bufferLengthFast(vCaches[li])) < vBytes ||
		int(bufferLengthFast(set.kGammas[li])) < gBytes || int(bufferLengthFast(set.vGammas[li])) < gBytes {
		return sessionStateLayerView{}, false, core.NewError("native.sessionState: TurboQuant cache plane shorter than maxLen rows")
	}
	view := sessionStateLayerView{
		layer:         li,
		kvHeads:       kvHeadsOf(spec, s.arch.KVHeads),
		headDim:       headDimOf(spec, s.arch.HeadDim),
		rowBytes:      set.kRowBytes[li],
		cacheIndex:    spec.CacheIndex,
		cacheMode:     nativeStateCacheModeTurboQuantCodes,
		maxSize:       0, // global seq-major — never a ring
		cacheRows:     rows,
		keyBytes:      unsafe.Slice((*byte)(kCaches[li].Contents()), kBytes),
		valueBytes:    unsafe.Slice((*byte)(vCaches[li].Contents()), vBytes),
		vRowBytes:     set.vRowBytes[li],
		gammaRowBytes: set.gammaRowBytes[li],
		kBits:         set.kBits,
		vBits:         set.vBits,
		kGammaBytes:   unsafe.Slice((*byte)(set.kGammas[li].Contents()), gBytes),
		vGammaBytes:   unsafe.Slice((*byte)(set.vGammas[li].Contents()), gBytes),
	}
	return view, true, nil
}

// tqStateBlockLayerRange bounds-checks one plane's [start, start+tokenCount)
// row range and returns the byte slice. TQ owners are global seq-major: no
// ring wrap, no expiry — the range is a straight row slice.
func tqStateBlockLayerRange(plane []byte, start, tokenCount, rowBytes int, label string) ([]byte, error) {
	if rowBytes <= 0 {
		return nil, core.NewError("native.StateBlockSource.Load: invalid TurboQuant " + label + " stride")
	}
	off, n := start*rowBytes, tokenCount*rowBytes
	if off < 0 || n < 0 || off+n > len(plane) {
		return nil, core.NewError("native.StateBlockSource.Load: TurboQuant " + label + " block exceeds cache rows")
	}
	return plane[off : off+n], nil
}

// tqFillStateBlockLayer assembles one TurboQuant layer's block payload for the
// token range [start, start+tokenCount): raw code rows (both sides, own
// strides) + γ rows + bit widths. Zero-copy views into the live cache planes,
// exactly like the native fill.
func tqFillStateBlockLayer(view sessionStateLayerView, start, tokenCount, position int) (SessionStateLayerBlock, error) {
	if position > view.cacheRows {
		return SessionStateLayerBlock{}, core.NewError("native.StateBlockSource.Load: TurboQuant position exceeds cache rows")
	}
	keyBytes, err := tqStateBlockLayerRange(view.keyBytes, start, tokenCount, view.rowBytes, "K code")
	if err != nil {
		return SessionStateLayerBlock{}, err
	}
	valueBytes, err := tqStateBlockLayerRange(view.valueBytes, start, tokenCount, view.vRowBytes, "V code")
	if err != nil {
		return SessionStateLayerBlock{}, err
	}
	kGamma, err := tqStateBlockLayerRange(view.kGammaBytes, start, tokenCount, view.gammaRowBytes, "K gamma")
	if err != nil {
		return SessionStateLayerBlock{}, err
	}
	vGamma, err := tqStateBlockLayerRange(view.vGammaBytes, start, tokenCount, view.gammaRowBytes, "V gamma")
	if err != nil {
		return SessionStateLayerBlock{}, err
	}
	return SessionStateLayerBlock{
		Layer:           view.layer,
		CacheIndex:      view.cacheIndex,
		CacheMode:       nativeStateCacheModeTurboQuantCodes,
		MaxSize:         view.maxSize,
		KVHeads:         view.kvHeads,
		HeadDim:         view.headDim,
		RowBytes:        view.rowBytes,
		KeyBytes:        keyBytes,
		ValueBytes:      valueBytes,
		ValueRowBytes:   view.vRowBytes,
		GammaRowBytes:   view.gammaRowBytes,
		KBits:           view.kBits,
		VBits:           view.vBits,
		KeyGammaBytes:   kGamma,
		ValueGammaBytes: vGamma,
	}, nil
}

// tqRestoreStateBlockLayer validates and lands one TurboQuant layer block into
// the target view. KIND EQUALITY IS ABSOLUTE: a turboquant-codes block lands
// only on a turboquant-codes view (and vice versa), with matching bit widths,
// strides and geometry — a mismatch is a loud error, never a reinterpret, and
// never the legacy cross-engine cache-mode waiver (this function runs BEFORE
// those generic checks precisely so that allowlist can never apply).
func tqRestoreStateBlockLayer(view sessionStateLayerView, start, tokenCount, position int, layer SessionStateLayerBlock) error {
	if layer.CacheMode != nativeStateCacheModeTurboQuantCodes || view.cacheMode != nativeStateCacheModeTurboQuantCodes {
		return core.NewError("native.RestoreStateBlocks: TurboQuant cache-kind mismatch (a turboquant-codes block restores only onto a turboquant-codes layer)")
	}
	if layer.KVHeads != view.kvHeads || layer.HeadDim != view.headDim {
		return core.NewError("native.RestoreStateBlocks: TurboQuant geometry mismatch")
	}
	if layer.KBits != view.kBits || layer.VBits != view.vBits {
		return core.NewError("native.RestoreStateBlocks: TurboQuant bit-width mismatch (snapshot and session must run the same -kv-cache mode)")
	}
	if layer.RowBytes != view.rowBytes || layer.ValueRowBytes != view.vRowBytes || layer.GammaRowBytes != view.gammaRowBytes {
		return core.NewError("native.RestoreStateBlocks: TurboQuant row-stride mismatch")
	}
	if position > view.cacheRows || start+tokenCount > position {
		return core.NewError("native.RestoreStateBlocks: TurboQuant block outside position")
	}
	kDst, err := tqStateBlockLayerRange(view.keyBytes, start, tokenCount, view.rowBytes, "K code")
	if err != nil {
		return err
	}
	vDst, err := tqStateBlockLayerRange(view.valueBytes, start, tokenCount, view.vRowBytes, "V code")
	if err != nil {
		return err
	}
	kgDst, err := tqStateBlockLayerRange(view.kGammaBytes, start, tokenCount, view.gammaRowBytes, "K gamma")
	if err != nil {
		return err
	}
	vgDst, err := tqStateBlockLayerRange(view.vGammaBytes, start, tokenCount, view.gammaRowBytes, "V gamma")
	if err != nil {
		return err
	}
	if len(layer.KeyBytes) != len(kDst) || len(layer.ValueBytes) != len(vDst) ||
		len(layer.KeyGammaBytes) != len(kgDst) || len(layer.ValueGammaBytes) != len(vgDst) {
		return core.NewError("native.RestoreStateBlocks: TurboQuant payload size mismatch")
	}
	copy(kDst, layer.KeyBytes)
	copy(vDst, layer.ValueBytes)
	copy(kgDst, layer.KeyGammaBytes)
	copy(vgDst, layer.ValueGammaBytes)
	return nil
}
