// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"os"
	"testing"
	"unsafe"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
	g4 "dappco.re/go/inference/model/gemma4"
	"dappco.re/go/inference/model/safetensors"
	"github.com/tmc/apple/metal"
)

// TestNoCopyMisalignedWeightReadsCorrectly guards the bf16 zero-copy weight path against the E4B-bf16
// regression: a non-bf16 odd-length tensor early in the checkpoint shifts every weight after it to an
// ODD byte offset, and Metal's setBuffer:offset cannot do a misaligned (odd-byte) bf16 read — it reads
// shifted, valid-looking but WRONG bytes (→ NaN downstream). bufFor copies misaligned weights into an
// aligned owned buffer; aligned weights stay zero-copy. Either way the bytes the GPU reads must equal
// the weight. RMSNorm(ones, weight) == weight (rms(ones)=1), so it reads back exactly what the GPU sees
// from bufFor's buffer vs the same weight copied — they must match. Set E4B_BF16_DIR (the model that
// exhibits the odd-offset layout); skips otherwise.
func TestNoCopyMisalignedWeightReadsCorrectly(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	dir := os.Getenv("E4B_BF16_DIR")
	if dir == "" {
		t.Skip("set E4B_BF16_DIR")
	}
	if err := ensureInit(); err != nil {
		t.Fatal(err)
	}
	cfgStr, err := coreio.Local.Read(core.PathJoin(dir, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cfg g4.Config
	if r := core.JSONUnmarshal([]byte(cfgStr), &cfg); !r.OK {
		t.Fatal("config parse failed")
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatal(err)
	}
	dm, err := safetensors.LoadDirMmap(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer dm.Close()
	sb, err := buildShardBuffers(dm)
	if err != nil {
		t.Fatal(err)
	}
	lm, err := g4Assemble(dm.Tensors, arch)
	if err != nil {
		t.Fatal(err)
	}
	g := loadedToBF16(lm)      // the same conversion LoadDir runs — no byte copy, keeps the mmap views
	w := g.Layers[0].AttnNormW // L0 input_layernorm — a no-copy view into the mmap
	dModel := arch.Hidden

	// raw offset (before bufFor's alignment handling) — odd here means the misalignment-copy path runs
	p := uintptr(unsafe.Pointer(&w[0]))
	var rawOff uint
	for i := range sb.bufs {
		if p >= sb.bases[i] && p < sb.ends[i] {
			rawOff = uint(p - sb.bases[i])
			break
		}
	}
	bv, err := sb.bufFor(w)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("L0 input_layernorm raw offset mod 2 = %d (odd ⇒ misalignment-copy path), bufFor off = %d", rawOff%2, bv.off)

	ones := make([]float32, dModel)
	for i := range ones {
		ones[i] = 1.0
	}
	xb := toBF16Bytes(ones)
	var outFix, outCopy []byte
	withAutoreleasePool(func() {
		xBuf := sharedBytes(xb)
		outBuf := scratchBF16(dModel)
		run := func(wBuf metal.MTLBuffer, woff uint) []byte {
			cb := queue.CommandBuffer()
			enc := cb.ComputeCommandEncoder()
			_ = encRMSNormBF16(enc, xBuf, wBuf, outBuf, woff, dModel, arch.Eps)
			enc.EndEncoding()
			cb.Commit()
			cb.WaitUntilCompleted()
			r := make([]byte, dModel*bf16Size)
			copy(r, unsafe.Slice((*byte)(outBuf.Contents()), dModel*bf16Size))
			return r
		}
		outFix = run(bv.buf, bv.off)     // what bufFor resolved (no-copy if aligned, owned copy if not)
		outCopy = run(sharedBytes(w), 0) // control: the weight copied into a fresh aligned buffer
	})
	nan := 0
	for i := 0; i+1 < len(outFix); i += 2 {
		if v := bf16ToF32(outFix[i], outFix[i+1]); v != v {
			nan++
		}
	}
	if nan > 0 {
		t.Errorf("bufFor weight produced %d/%d NaN through RMSNorm — misaligned GPU read", nan, dModel)
	}
	if !bytes.Equal(outFix, outCopy) {
		t.Errorf("bufFor weight ≠ copied weight through RMSNorm — Metal read the wrong (shifted) bytes")
	}
}
