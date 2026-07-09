// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"os"
	"strconv"
	"sync"

	core "dappco.re/go"
)

// LTHN_MTP_CONF=<path> — the #359 instrument: append one JSON line per MTP
// verify cycle pairing the drafter's per-position self-confidence (softmax
// probability of its own greedy pick) with the target's accepted prefix
// length. The reliability curve derived from this capture is where the
// confidence-scheduling threshold θ comes from — DSpark's 0.4 is calibrated
// to their trained confidence head on Qwen3, not to a gemma4 drafter's raw
// softmax, so we measure our own. Diag-only: when unset the draft loop takes
// zero extra work (no softmax, no allocation, no I/O). Greedy lane only —
// the sampled draft lane is not instrumented.
var mtpConfCapturePath = os.Getenv("LTHN_MTP_CONF")

func mtpConfEnabled() bool { return mtpConfCapturePath != "" }

// LTHN_MTP_CONF_FORCE=1 — calibration mode (greedy lane): with the capture
// armed, draft every cycle by bypassing the low-accept bail and the deep
// bootstrap. The re-engagement gate's economics select AGAINST the very
// cycles the curve needs (it stops drafting exactly where the drafter is
// weak), so an unforced capture oversamples the good regimes. Never set on a
// serving run — it deliberately trades tok/s for unbiased coverage.
var mtpConfForce = os.Getenv("LTHN_MTP_CONF_FORCE") == "1" && mtpConfCapturePath != ""

// mtpConfProb returns softmax probability of tokID over a host bf16 logits
// row, mirroring draftGreedyTokenWithSuppress's iteration exactly: suppressed
// ids are excluded from the distribution (the greedy pick was made without
// them, so its confidence must be too). Two passes, f64 accumulation.
func mtpConfProb(logits []byte, tokID int32, suppressed []int32) float32 {
	vocab := len(logits) / bf16Size
	if vocab == 0 || tokID < 0 || int(tokID) >= vocab {
		return 0
	}
	maxV := float32(math.Inf(-1))
	for id := range vocab {
		if nativeAssistantSuppressed(int32(id), suppressed) {
			continue
		}
		if v := bf16ToF32(logits[id*bf16Size], logits[id*bf16Size+1]); v > maxV {
			maxV = v
		}
	}
	var sum float64
	for id := range vocab {
		if nativeAssistantSuppressed(int32(id), suppressed) {
			continue
		}
		v := bf16ToF32(logits[id*bf16Size], logits[id*bf16Size+1])
		sum += math.Exp(float64(v - maxV))
	}
	if sum <= 0 {
		return 0
	}
	tok := bf16ToF32(logits[tokID*bf16Size], logits[tokID*bf16Size+1])
	return float32(math.Exp(float64(tok-maxV)) / sum)
}

// mtpConfSink is the process-lifetime append writer. Open is lazy so serving
// binaries that never draft pay nothing; a failed open disables capture for
// the rest of the process (a diag lever must never break generation).
type mtpConfSink struct {
	mu   sync.Mutex
	f    *os.File
	dead bool
}

var mtpConfOut mtpConfSink

// mtpConfRecordCycle appends one cycle line:
//
//	{"pos":18432,"carry":1,"probs":[0.9531,0.8125,0.4023,0.1201],"accepted":3}
//
// pos is the target position before verify; carry is the block offset (a
// pending lead token verified ahead of the draft), so drafted position k is
// prefix-accepted iff k+carry < accepted. Labels are derived at analysis
// time; the line stores raw facts only.
func mtpConfRecordCycle(pos, carry int, probs []float32, accepted int) {
	mtpConfOut.mu.Lock()
	defer mtpConfOut.mu.Unlock()
	if mtpConfOut.dead {
		return
	}
	if mtpConfOut.f == nil {
		f, err := os.OpenFile(mtpConfCapturePath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		if err != nil {
			mtpConfOut.dead = true
			nativeTraceLog(core.Sprintf("mtp-conf: capture disabled, open %s: %v\n", mtpConfCapturePath, err))
			return
		}
		mtpConfOut.f = f
	}
	buf := make([]byte, 0, 64+12*len(probs))
	buf = append(buf, `{"pos":`...)
	buf = strconv.AppendInt(buf, int64(pos), 10)
	buf = append(buf, `,"carry":`...)
	buf = strconv.AppendInt(buf, int64(carry), 10)
	buf = append(buf, `,"probs":[`...)
	for i, p := range probs {
		if i > 0 {
			buf = append(buf, ',')
		}
		buf = strconv.AppendFloat(buf, float64(p), 'f', 4, 32)
	}
	buf = append(buf, `],"accepted":`...)
	buf = strconv.AppendInt(buf, int64(accepted), 10)
	buf = append(buf, '}', '\n')
	if _, err := mtpConfOut.f.Write(buf); err != nil {
		mtpConfOut.dead = true
		nativeTraceLog(core.Sprintf("mtp-conf: capture disabled, write: %v\n", err))
	}
}
