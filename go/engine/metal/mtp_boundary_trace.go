// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// mtp_boundary_trace.go — dev instrument (LTHN_MTP_BOUNDARY_TRACE=1): one
// stderr line per MTP boundary event (plain-stretch drain, draft entry/exit,
// verify boundary argmax, verify row verdicts) carrying the session position
// and an FNV-1a hash of the retained boundary hidden, so two runs' traces
// diff to the FIRST divergent event — separating "the hidden bytes wobble"
// from "the compute over identical bytes wobbles".
var mtpBoundaryTraceOn = os.Getenv("LTHN_MTP_BOUNDARY_TRACE") == "1"

// mtpBoundaryHash is FNV-1a 64 over the given bytes — cheap enough per event
// (a hidden row is a few KB) to stay under the timing-perturbation threshold
// heavy diag logging crosses.
func mtpBoundaryHash(b []byte) uint64 {
	h := uint64(14695981039346656037)
	for _, c := range b {
		h ^= uint64(c)
		h *= 1099511628211
	}
	return h
}

func mtpBoundaryTrace(event string, pos int, tok int32, hidden []byte) {
	if !mtpBoundaryTraceOn {
		return
	}
	nativeTraceLog(core.Sprintf("mtp-boundary %s pos=%d tok=%d h=%016x n=%d\n",
		event, pos, tok, mtpBoundaryHash(hidden), len(hidden)))
}

// mtpBoundaryTraceRows logs one verify round's verdict: the accepted count and
// the replacement choice, keyed by position.
func mtpBoundaryTraceRows(pos, accepted int, replacement int32, rows []int32) {
	if !mtpBoundaryTraceOn {
		return
	}
	nativeTraceLog(core.Sprintf("mtp-boundary verify-rows pos=%d accepted=%d repl=%d rows=%v\n",
		pos, accepted, replacement, rows))
}

// mtpBoundaryTraceKV hashes the newest KV rows of the first two owner layers
// (codes + q8 scale rows) at a quiesced cycle boundary — two structurally
// different runs that token-agree over a window must row-hash-agree over it
// too; the first row where they split names the lane parity break.
func mtpBoundaryTraceKV(s *ArchSession, event string) {
	if !mtpBoundaryTraceOn || s == nil || s.state.icb == nil {
		return
	}
	icb := s.state.icb
	// One sliding owner + one global (q8-armed when the lane is armed) owner:
	// the parity break can live in either population.
	sample := make([]int, 0, 2)
	for li := range s.state.specs {
		if !s.state.specs[li].OwnsCache() || icb.kCaches[li] == nil {
			continue
		}
		if s.state.specs[li].Attention == model.GlobalAttention {
			sample = append(sample, li)
			break
		}
		if len(sample) == 0 {
			sample = append(sample, li)
		}
	}
	for _, li := range sample {
		rows, rb := icb.cacheRows[li], icb.rowBytes[li]
		if rows <= 0 || rb <= 0 || s.pos > rows {
			continue
		}
		// rowBytes is bf16-shaped for every owner; a q8 layer's code row is
		// kvd bytes (one byte per element) with a 4-byte-per-group scale row.
		q8 := icb.kvQ8.on(li)
		kvd := rb / bf16Size
		if !q8 {
			kvd = rb
		}
		lo := max(0, s.pos-10)
		kBytes := unsafe.Slice((*byte)(icb.kCaches[li].Contents()), rows*kvd)
		var sBytes []byte
		if q8 && icb.kvQ8.kScales[li] != nil {
			sBytes = unsafe.Slice((*byte)(icb.kvQ8.kScales[li].Contents()), rows*(kvd/kvQ8GroupSize)*4)
		}
		line := core.Sprintf("mtp-boundary kv %s l=%d q8=%v pos=%d rows", event, li, q8, s.pos)
		for r := lo; r < s.pos; r++ {
			h := mtpBoundaryHash(kBytes[r*kvd : (r+1)*kvd])
			if sBytes != nil {
				sw := (kvd / kvQ8GroupSize) * 4
				h ^= mtpBoundaryHash(sBytes[r*sw : (r+1)*sw])
			}
			line += core.Sprintf(" %d:%08x", r, uint32(h))
		}
		nativeTraceLog(line + "\n")
	}
}
