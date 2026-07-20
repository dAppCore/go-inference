// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"github.com/tmc/apple/metal"
)

// cache_plan.go — the load-time cache-plan instrument (#48 follow-on to the
// #41 TurboQuant campaign): a structured, inspectable summary of the ACTUAL
// per-layer KV placement a session build resolved, independent of what
// -kv-cache asked for. This is OBSERVATION ONLY — it reports what got built
// (which carrier, how many bytes, paged or linear), it never changes what
// gets built.
//
// It exists because a #48 resume bench measured ZERO RSS delta between
// "native" and "-kv-cache turboquant" on a 31B-4bit/17K-token serve, and
// nothing in the engine printed what the session actually decided — so the
// gap between "the flag was accepted" (withKVCacheMode threads it as far as
// NewQuantTokenModel's BackendOptions, verified) and "the flag changed the
// resident bytes" was invisible. Two things this instrument makes visible
// that -kv-cache alone cannot show:
//
//   - q8 KV is ALREADY the default on the recorded-ICB lane for qualifying
//     geometry (kvQ8ICBOn — env LTHN_KV_Q8_ICB, on unless set to "0") and on
//     the paged/stepToken lane for qualifying gqa2 geometry (kvQ8Enabled —
//     env LTHN_KV_Q8, same on-unless-"0" default). A "native" comparison
//     point may already be q8-compressed, not bf16 — Q8Count/Q8Bytes says so
//     instead of leaving it a guess (see decode_forward_arch_icb_q8.go and
//     sdpa_paged.go for the two switches).
//   - the device-paged pool (archDecodeState.initDevicePagedKVWithPrealloc)
//     allocates UNCONDITIONALLY at session build, before the ICB-eligible
//     branch below even runs — it is dead weight for an ICB/state-TQ session
//     (Step/StepWithID replay s.state.icb or the state carrier, never
//     s.pagedKV) but costs real bytes the instant -paged-kv-prealloc is set.
//     PagedPoolBytes reports its LIVE resident bytes regardless of lane, so
//     a nonzero value next to lane=icb is a visible fact, not a mystery.
//
// Called once per session build — newArchSessionShardsWithHeadConfig /
// newArchQuantSessionShardsWithHeadConfig, the ONE seam both LoadDir
// (ArchSession) and the LoadTokenModelDir token-model path converge on: the
// token-model's openSession closure (token_model.go's NewBF16TokenModel /
// NewQuantTokenModel) calls straight into these same two constructors from
// OpenSession(), carrying archSessionConfig.kvCacheMode (== the backend's
// kvCacheMode field withKVCacheMode set — backend.go) through unchanged. One
// instrumentation point at the shared constructor covers both load paths.

// LayerCacheKind classifies one decode layer's resident attention-cache
// storage as the session build actually resolved it — read from the real
// allocations, never re-derived from the requested -kv-cache mode alone (a
// declined or unarmed request must show up as native here, not turboquant).
type LayerCacheKind uint8

const (
	// CacheKindRecurrent is a MixerGatedDelta layer: no KV rows exist (the
	// mixer holds recurrent state instead of an attention cache).
	CacheKindRecurrent LayerCacheKind = iota
	// CacheKindShared is a non-owner layer reading another layer's cache
	// (KVShareFrom) — no resident rows of its own.
	CacheKindShared
	// CacheKindNative is bf16 K/V rows — the engine's built-in default.
	CacheKindNative
	// CacheKindQ8 is int8 codes + f32 group scales (#367 — on by default for
	// qualifying geometry; see kvQ8ICBOn / kvQ8Enabled).
	CacheKindQ8
	// CacheKindTurboQuant is packed Lloyd-Max codes + per-row-per-head γ
	// (#41 S3, opt-in via -kv-cache turboquant[:N]).
	CacheKindTurboQuant
)

// String names the kind for the log line and test failure messages.
func (k LayerCacheKind) String() string {
	switch k {
	case CacheKindRecurrent:
		return "recurrent"
	case CacheKindShared:
		return "shared"
	case CacheKindNative:
		return "native"
	case CacheKindQ8:
		return "q8"
	case CacheKindTurboQuant:
		return "turboquant"
	default:
		return "unknown"
	}
}

// LayerCachePlan is one layer's resolved cache placement + resident byte
// cost (K+V and any scale/γ plane; 0 for a shared/recurrent layer).
type LayerCachePlan struct {
	Kind LayerCacheKind
	// Owner is this layer's KVShareFrom — == the layer's own index for an
	// owner, the owning layer's index for a shared follower.
	Owner int
	// Paged reports whether this layer's ACTUAL decode-path cache lives in
	// the device-paged pool (the stepToken lane's alternative to the linear
	// lb caches). Always false on the icb/state-tq lanes: their caches are
	// linear by construction even when a paged pool ALSO exists dead beside
	// them — see CachePlanSummary.PagedPoolBytes.
	Paged bool
	// Deferred reports a PLANNED-but-not-yet-materialised linear lb cache
	// (ensureLBKVCaches allocates lazily on first stepToken touch) — Bytes
	// is then the planned size, not a resident one.
	Deferred bool
	Bytes    int64
}

// CachePlanSummary is the whole session's ACTUAL cache plan — one value per
// session build, the log line's source and the test-inspectable return.
type CachePlanSummary struct {
	// Mode is the requested -kv-cache mode string, unparsed ("" = native).
	Mode string
	// Lane is the decode mechanism that OWNS these layers' caches: "icb" (the
	// recorded chained replay), "state-tq" (the stepToken TurboQuant state
	// carrier — MoE/hybrid stacks), or "stepToken" (native re-encode, no
	// ICB/TQ carrier armed).
	Lane string

	// Layers is the per-layer breakdown, index-aligned with model.Arch.Layer.
	Layers []LayerCachePlan

	NativeCount, Q8Count, TQCount, SharedCount, RecurrentCount int
	NativeBytes, Q8Bytes, TQBytes                              int64

	// PagedPoolBytes is the device-paged pool's ACTUAL resident bytes (summed
	// live page buffers), regardless of whether this session's lane reads
	// it. See the file doc for why a nonzero value under lane=icb matters.
	PagedPoolBytes int64
	// PagedCount/LinearCount partition the OWNING attention layers on the
	// stepToken lane between the paged pool and the deferred linear lb
	// caches; both are 0 on the icb/state-tq lanes (their owning layers are
	// always linear, whatever a dead paged pool alongside them holds).
	PagedCount, LinearCount int
}

// safeBufLen is bufferLengthFast, nil-safe — a layer that never qualified
// for a carrier (or a not-yet-materialised lb cache) leaves its buffer nil.
func safeBufLen(buf metal.MTLBuffer) int64 {
	if buf == nil {
		return 0
	}
	return int64(bufferLengthFast(buf))
}

// pagedCacheBytes sums one paged cache's LIVE page bytes (K + V + the q8
// scale pages when quantised) — pages the geometric grower has actually
// appended, never the nominal maxSize (preallocPages is the only caller that
// forces the full extent; the lazy default grows on demand, so an unused
// pool on an ICB session costs ~0 here).
func pagedCacheBytes(pc *devicePagedKVCache) int64 {
	if pc == nil {
		return 0
	}
	var total int64
	for _, b := range pc.kPages {
		total += safeBufLen(b)
	}
	for _, b := range pc.vPages {
		total += safeBufLen(b)
	}
	if pc.quantQ8 {
		for _, b := range pc.kScalePages {
			total += safeBufLen(b)
		}
		for _, b := range pc.vScalePages {
			total += safeBufLen(b)
		}
	}
	return total
}

// icbLayerPlan classifies owning layer li on the recorded-ICB lane: TQ codes
// (kvTQ), q8 codes (kvQ8), or the bf16 default — mutually exclusive per the
// allocArchICBCachesTQ/allocArchICBCaches contract (a layer is never both).
func icbLayerPlan(icb *archICBReplay, li int) LayerCachePlan {
	lp := LayerCachePlan{Kind: CacheKindNative, Owner: li, Bytes: safeBufLen(icb.kCaches[li]) + safeBufLen(icb.vCaches[li])}
	switch {
	case icb.kvTQ.on(li):
		lp.Kind = CacheKindTurboQuant
		lp.Bytes += safeBufLen(icb.kvTQ.kGammas[li]) + safeBufLen(icb.kvTQ.vGammas[li])
	case icb.kvQ8.on(li):
		lp.Kind = CacheKindQ8
		lp.Bytes += safeBufLen(icb.kvQ8.kScales[li]) + safeBufLen(icb.kvQ8.vScales[li])
	}
	return lp
}

// stateTQLayerPlan classifies owning layer li on the state-lane TurboQuant
// carrier (tq_kv_state.go). Callers only reach here when kvTQState.on(li) —
// every armed layer is TurboQuant by construction (the carrier holds no
// bf16/q8 alternative); a non-qualifying owner on a state-armed session
// routes through linearLayerPlan instead.
func stateTQLayerPlan(t *archStateKVTQ, li int) LayerCachePlan {
	return LayerCachePlan{
		Kind:  CacheKindTurboQuant,
		Owner: li,
		Bytes: safeBufLen(t.kCaches[li]) + safeBufLen(t.vCaches[li]) + safeBufLen(t.set.kGammas[li]) + safeBufLen(t.set.vGammas[li]),
	}
}

// linearLayerPlan classifies owning layer li on the stepToken lane (no ICB,
// no state-TQ carrier armed on this layer): the device-paged pool when the
// session built one (q8 when the pool itself is quantised — sdpa_paged.go's
// kvQ8Enabled default), else the deferred linear lb cache (materialises
// lazily on first touch — see archLayerBufs.kvCacheBytes / ensureLBKVCaches).
func linearLayerPlan(st *archDecodeState, li int) LayerCachePlan {
	lp := LayerCachePlan{Kind: CacheKindNative, Owner: li}
	if pc := st.layerPagedKV(li); pc != nil {
		lp.Paged = true
		lp.Bytes = pagedCacheBytes(pc)
		if pc.quantQ8 {
			lp.Kind = CacheKindQ8
		}
		return lp
	}
	if li < len(st.lb) {
		lb := st.lb[li]
		if lb.kCache != nil || lb.vCache != nil {
			lp.Bytes = safeBufLen(lb.kCache) + safeBufLen(lb.vCache)
		} else {
			lp.Deferred = true
			lp.Bytes = int64(lb.kvCacheBytes) * 2 // K+V planned; ensureLBKVCaches has not materialised either yet
		}
	}
	return lp
}

// buildCachePlan walks the just-built session's per-layer resident state and
// classifies it — additive observation over the real allocations the
// constructor already made; it allocates nothing itself and mutates
// nothing. mode is the raw -kv-cache request string (archSessionConfig.
// kvCacheMode) — carried through only for the summary's Mode field, never
// re-parsed to decide a layer's kind here (the kind comes from what got
// built, which is the whole point).
func buildCachePlan(arch model.Arch, st *archDecodeState, mode string) CachePlanSummary {
	sum := CachePlanSummary{Mode: mode}
	switch {
	case st.icb != nil:
		sum.Lane = "icb"
	case st.tqStateArmed():
		sum.Lane = "state-tq"
	default:
		sum.Lane = "stepToken"
	}
	sum.Layers = make([]LayerCachePlan, len(arch.Layer))
	for li := range arch.Layer {
		sp := arch.Layer[li]
		var lp LayerCachePlan
		switch {
		case sp.Mixer == model.MixerGatedDelta:
			lp = LayerCachePlan{Kind: CacheKindRecurrent, Owner: li}
		case !sp.OwnsCache():
			lp = LayerCachePlan{Kind: CacheKindShared, Owner: sp.KVShareFrom}
		case st.icb != nil:
			lp = icbLayerPlan(st.icb, li)
		case st.kvTQState.on(li):
			lp = stateTQLayerPlan(st.kvTQState, li)
		default:
			lp = linearLayerPlan(st, li)
		}
		sum.Layers[li] = lp
		switch lp.Kind {
		case CacheKindNative:
			sum.NativeCount++
			sum.NativeBytes += lp.Bytes
		case CacheKindQ8:
			sum.Q8Count++
			sum.Q8Bytes += lp.Bytes
		case CacheKindTurboQuant:
			sum.TQCount++
			sum.TQBytes += lp.Bytes
		case CacheKindShared:
			sum.SharedCount++
		case CacheKindRecurrent:
			sum.RecurrentCount++
		}
		if lp.Kind == CacheKindNative || lp.Kind == CacheKindQ8 || lp.Kind == CacheKindTurboQuant {
			if lp.Paged {
				sum.PagedCount++
			} else {
				sum.LinearCount++
			}
		}
	}
	for _, pc := range st.pagedKV {
		sum.PagedPoolBytes += pagedCacheBytes(pc)
	}
	return sum
}

// logLine formats the summary as ONE structured stderr line — the
// clampContextToRAM idiom (load_context_ram.go): a plain nativeTraceLog
// call, always on, never gated behind a diag env.
func (sum CachePlanSummary) logLine() string {
	mode := core.Trim(sum.Mode)
	if mode == "" {
		mode = "native"
	}
	return core.Sprintf(
		"native: cache-plan mode=%s lane=%s layers=%d native=%d(%dMiB) q8=%d(%dMiB) turboquant=%d(%dMiB) shared=%d recurrent=%d linear=%d paged=%d pagedPoolBytes=%dMiB\n",
		mode, sum.Lane, len(sum.Layers),
		sum.NativeCount, sum.NativeBytes>>20,
		sum.Q8Count, sum.Q8Bytes>>20,
		sum.TQCount, sum.TQBytes>>20,
		sum.SharedCount, sum.RecurrentCount,
		sum.LinearCount, sum.PagedCount,
		sum.PagedPoolBytes>>20,
	)
}

// logCachePlan computes the session's ACTUAL cache plan and prints it as one
// always-on stderr line, returning the summary so a caller (or a test) can
// inspect the counts directly instead of parsing the line.
func logCachePlan(arch model.Arch, st *archDecodeState, mode string) CachePlanSummary {
	sum := buildCachePlan(arch, st, mode)
	nativeTraceLog(sum.logLine())
	return sum
}
