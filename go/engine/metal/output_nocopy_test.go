// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"
	"unsafe"

	"github.com/tmc/apple/metal"
)

func requirePinnedOwnerOutputBuffer(t *testing.T, name string, got metal.MTLBuffer, pinned *pinnedNoCopyBytes) {
	t.Helper()
	requirePinnedOwnerBuffer(t, name+" output view", got, pinned)
}

func requirePinnedOwnerBuffer(t *testing.T, name string, got metal.MTLBuffer, pinned *pinnedNoCopyBytes) {
	t.Helper()
	if got == nil {
		t.Fatalf("%s returned nil buffer", name)
	}
	if gotID, wantID := got.GetID(), pinned.buf.GetID(); gotID != wantID {
		t.Fatalf("%s buffer id = %d, want pinned owner buffer %d", name, gotID, wantID)
	}
	if gotPtr, wantPtr := uintptr(got.Contents()), uintptr(unsafe.Pointer(&pinned.bytes[0])); gotPtr != wantPtr {
		t.Fatalf("%s pointer = %#x, want pinned backing %#x", name, gotPtr, wantPtr)
	}
}

func TestSimpleScratchOutputViewsReusePinnedOwnerBuffer(t *testing.T) {
	requireNativeRuntime(t)

	t.Run("SDPA BF16", func(t *testing.T) {
		pinned, err := newPinnedNoCopyBytes(64 * bf16Size)
		if err != nil {
			t.Fatalf("newPinnedNoCopyBytes: %v", err)
		}
		defer pinned.Close()

		scratch, err := getSDPABF16Scratch(64*bf16Size, 64*bf16Size, 64*bf16Size, len(pinned.bytes))
		if err != nil {
			t.Fatalf("getSDPABF16Scratch: %v", err)
		}
		defer scratch.Close()

		outBuf, ok := scratch.outputView(pinned.bytes)
		if !ok {
			t.Fatal("SDPA output view did not accept pinned caller bytes")
		}
		requirePinnedOwnerOutputBuffer(t, "SDPA", outBuf, pinned)
	})

	t.Run("RMS residual BF16", func(t *testing.T) {
		const axisSize = 64
		pinned, err := newPinnedNoCopyBytes(axisSize * bf16Size)
		if err != nil {
			t.Fatalf("newPinnedNoCopyBytes: %v", err)
		}
		defer pinned.Close()

		scratch, err := getRMSNormResidualBF16Scratch(axisSize)
		if err != nil {
			t.Fatalf("getRMSNormResidualBF16Scratch: %v", err)
		}
		defer scratch.Close()

		outBuf, ok := scratch.outputView(pinned.bytes)
		if !ok {
			t.Fatal("RMS residual output view did not accept pinned caller bytes")
		}
		requirePinnedOwnerOutputBuffer(t, "RMS residual", outBuf, pinned)
	})

	t.Run("embed gather", func(t *testing.T) {
		const dModel = 64
		pinned, err := newPinnedNoCopyBytes(dModel * bf16Size)
		if err != nil {
			t.Fatalf("newPinnedNoCopyBytes: %v", err)
		}
		defer pinned.Close()

		scratch, err := getEmbedGatherScratch(dModel)
		if err != nil {
			t.Fatalf("getEmbedGatherScratch: %v", err)
		}
		defer scratch.Close()

		outBuf, ok := scratch.outputView(pinned.bytes)
		if !ok {
			t.Fatal("embed gather output view did not accept pinned caller bytes")
		}
		requirePinnedOwnerOutputBuffer(t, "embed gather", outBuf, pinned)
	})

	t.Run("matmul BF16 steel", func(t *testing.T) {
		const m, k, n = 4, 8, 16
		pinned, err := newPinnedNoCopyBytes(m * n * bf16Size)
		if err != nil {
			t.Fatalf("newPinnedNoCopyBytes: %v", err)
		}
		defer pinned.Close()

		scratch, err := getMatMulBF16SteelScratch(m, k, n)
		if err != nil {
			t.Fatalf("getMatMulBF16SteelScratch: %v", err)
		}
		defer scratch.Close()

		outBuf, ok := scratch.outputView(pinned.bytes)
		if !ok {
			t.Fatal("matmul BF16 steel output view did not accept pinned caller bytes")
		}
		requirePinnedOwnerOutputBuffer(t, "matmul BF16 steel", outBuf, pinned)
	})

	t.Run("matmul float32 steel", func(t *testing.T) {
		const m, k, n = 4, 8, 16
		pinned, err := newPinnedNoCopyBytes(m * n * 4)
		if err != nil {
			t.Fatalf("newPinnedNoCopyBytes: %v", err)
		}
		defer pinned.Close()
		out := unsafe.Slice((*float32)(unsafe.Pointer(&pinned.bytes[0])), m*n)

		scratch, err := getMatMulF32SteelScratch(m, k, n, k, steelNT)
		if err != nil {
			t.Fatalf("getMatMulF32SteelScratch: %v", err)
		}
		defer scratch.Close()

		outBuf, ok := scratch.outputView(out)
		if !ok {
			t.Fatal("matmul float32 steel output view did not accept pinned caller bytes")
		}
		requirePinnedOwnerOutputBuffer(t, "matmul float32 steel", outBuf, pinned)
	})

	t.Run("per-layer GPU", func(t *testing.T) {
		const plDim, dModel = 16, 64
		pinned, err := newPinnedNoCopyBytes(dModel * bf16Size)
		if err != nil {
			t.Fatalf("newPinnedNoCopyBytes: %v", err)
		}
		defer pinned.Close()

		scratch, err := getPerLayerInputsGPUScratch(plDim, dModel, 1)
		if err != nil {
			t.Fatalf("getPerLayerInputsGPUScratch: %v", err)
		}
		defer scratch.Close()

		outBuf, outPtr, ok := scratch.outputView(pinned.bytes)
		if !ok {
			t.Fatal("per-layer GPU output view did not accept pinned caller bytes")
		}
		requirePinnedOwnerOutputBuffer(t, "per-layer GPU", outBuf, pinned)
		if got, want := uintptr(unsafe.Pointer(outPtr)), uintptr(unsafe.Pointer(&pinned.bytes[0])); got != want {
			t.Fatalf("per-layer GPU output pointer = %#x, want pinned backing %#x", got, want)
		}
	})
}

func TestMoEScratchInputViewsReusePinnedOwnerBuffer(t *testing.T) {
	requireNativeRuntime(t)

	t.Run("MoE experts input", func(t *testing.T) {
		const dModel, dFF, topK = 64, 128, 2
		pinned, err := newPinnedNoCopyBytes(dModel * bf16Size)
		if err != nil {
			t.Fatalf("newPinnedNoCopyBytes: %v", err)
		}
		defer pinned.Close()

		scratch, err := getMoEExpertsScratch(dModel, dFF, topK)
		if err != nil {
			t.Fatalf("getMoEExpertsScratch: %v", err)
		}
		defer scratch.Close()

		xBuf, ok := scratch.inputView(pinned.bytes)
		if !ok {
			t.Fatal("MoE experts input view did not accept pinned caller bytes")
		}
		requirePinnedOwnerBuffer(t, "MoE experts input view", xBuf, pinned)
	})

	t.Run("MoE experts weights", func(t *testing.T) {
		const dModel, dFF, topK = 64, 128, 2
		pinned, err := newPinnedNoCopyBytes(topK * bf16Size)
		if err != nil {
			t.Fatalf("newPinnedNoCopyBytes: %v", err)
		}
		defer pinned.Close()

		scratch, err := getMoEExpertsScratch(dModel, dFF, topK)
		if err != nil {
			t.Fatalf("getMoEExpertsScratch: %v", err)
		}
		defer scratch.Close()

		weightsBuf, ok := scratch.weightsView(pinned.bytes)
		if !ok {
			t.Fatal("MoE experts weights view did not accept pinned caller bytes")
		}
		requirePinnedOwnerBuffer(t, "MoE experts weights view", weightsBuf, pinned)
	})

	t.Run("MoE block input", func(t *testing.T) {
		const dModel, dFF, expertDFF, topK = 64, 128, 96, 2
		pinned, err := newPinnedNoCopyBytes(dModel * bf16Size)
		if err != nil {
			t.Fatalf("newPinnedNoCopyBytes: %v", err)
		}
		defer pinned.Close()

		scratch, err := getMoEBlockBF16Scratch(dModel, dFF, expertDFF, topK)
		if err != nil {
			t.Fatalf("getMoEBlockBF16Scratch: %v", err)
		}
		defer scratch.Close()

		hBuf, ok := scratch.inputView(pinned.bytes)
		if !ok {
			t.Fatal("MoE block input view did not accept pinned caller bytes")
		}
		requirePinnedOwnerBuffer(t, "MoE block input view", hBuf, pinned)
	})

	t.Run("MoE block weights", func(t *testing.T) {
		const dModel, dFF, expertDFF, topK = 64, 128, 96, 2
		pinned, err := newPinnedNoCopyBytes(topK * bf16Size)
		if err != nil {
			t.Fatalf("newPinnedNoCopyBytes: %v", err)
		}
		defer pinned.Close()

		scratch, err := getMoEBlockBF16Scratch(dModel, dFF, expertDFF, topK)
		if err != nil {
			t.Fatalf("getMoEBlockBF16Scratch: %v", err)
		}
		defer scratch.Close()

		weightsBuf, ok := scratch.weightsView(pinned.bytes)
		if !ok {
			t.Fatal("MoE block weights view did not accept pinned caller bytes")
		}
		requirePinnedOwnerBuffer(t, "MoE block weights view", weightsBuf, pinned)
	})

	t.Run("MoE block index", func(t *testing.T) {
		const dModel, dFF, expertDFF, topK = 64, 128, 96, 2
		pinned, err := newPinnedNoCopyBytes(topK * 4)
		if err != nil {
			t.Fatalf("newPinnedNoCopyBytes: %v", err)
		}
		defer pinned.Close()
		idx := unsafe.Slice((*int32)(unsafe.Pointer(&pinned.bytes[0])), topK)
		idx[0], idx[1] = 0, 1

		scratch, err := getMoEBlockBF16Scratch(dModel, dFF, expertDFF, topK)
		if err != nil {
			t.Fatalf("getMoEBlockBF16Scratch: %v", err)
		}
		defer scratch.Close()

		idxBuf, ok := scratch.indexView(idx)
		if !ok {
			t.Fatal("MoE block index view did not accept pinned caller bytes")
		}
		requirePinnedOwnerBuffer(t, "MoE block index view", idxBuf, pinned)
	})

	t.Run("MLP transform input", func(t *testing.T) {
		const dModel, dFF = 64, 128
		pinned, err := newPinnedNoCopyBytes(dModel * bf16Size)
		if err != nil {
			t.Fatalf("newPinnedNoCopyBytes: %v", err)
		}
		defer pinned.Close()

		scratch, err := getMLPTransformScratch(dModel, dFF)
		if err != nil {
			t.Fatalf("getMLPTransformScratch: %v", err)
		}
		defer scratch.Close()

		xBuf, ok := scratch.inputView(pinned.bytes)
		if !ok {
			t.Fatal("MLP transform input view did not accept pinned caller bytes")
		}
		requirePinnedOwnerBuffer(t, "MLP transform input view", xBuf, pinned)
	})

	t.Run("MLP transform mega input", func(t *testing.T) {
		const dModel, dFF = 256, 512
		pinned, err := newPinnedNoCopyBytes(dModel * bf16Size)
		if err != nil {
			t.Fatalf("newPinnedNoCopyBytes: %v", err)
		}
		defer pinned.Close()

		scratch, err := getMLPTransformMegaScratch(dModel, dFF)
		if err != nil {
			t.Fatalf("getMLPTransformMegaScratch: %v", err)
		}
		defer scratch.Close()

		xBuf, ok := scratch.inputView(pinned.bytes)
		if !ok {
			t.Fatal("MLP transform mega input view did not accept pinned caller bytes")
		}
		requirePinnedOwnerBuffer(t, "MLP transform mega input view", xBuf, pinned)
	})

	t.Run("MoE post-combine inputs", func(t *testing.T) {
		const dModel = 64
		scratch, err := getMoEBlockPostCombineScratch(dModel)
		if err != nil {
			t.Fatalf("getMoEBlockPostCombineScratch: %v", err)
		}
		defer scratch.Close()

		for _, tt := range []struct {
			name string
			view func([]byte) (metal.MTLBuffer, bool)
		}{
			{name: "residual", view: scratch.residualView},
			{name: "branch1", view: scratch.branch1View},
			{name: "branch2", view: scratch.branch2View},
		} {
			t.Run(tt.name, func(t *testing.T) {
				pinned, err := newPinnedNoCopyBytes(dModel * bf16Size)
				if err != nil {
					t.Fatalf("newPinnedNoCopyBytes: %v", err)
				}
				defer pinned.Close()

				buf, ok := tt.view(pinned.bytes)
				if !ok {
					t.Fatalf("MoE post-combine %s view did not accept pinned caller bytes", tt.name)
				}
				requirePinnedOwnerBuffer(t, "MoE post-combine "+tt.name+" view", buf, pinned)
			})
		}
	})
}

func TestMoEScratchOutputViewsReusePinnedOwnerBuffer(t *testing.T) {
	requireNativeRuntime(t)

	t.Run("MoE experts", func(t *testing.T) {
		const dModel, dFF, topK = 64, 128, 2
		pinned, err := newPinnedNoCopyBytes(dModel * bf16Size)
		if err != nil {
			t.Fatalf("newPinnedNoCopyBytes: %v", err)
		}
		defer pinned.Close()

		scratch, err := getMoEExpertsScratch(dModel, dFF, topK)
		if err != nil {
			t.Fatalf("getMoEExpertsScratch: %v", err)
		}
		defer scratch.Close()

		outBuf, ok := scratch.outputView(pinned.bytes)
		if !ok {
			t.Fatal("MoE experts output view did not accept pinned caller bytes")
		}
		requirePinnedOwnerOutputBuffer(t, "MoE experts", outBuf, pinned)
	})

	t.Run("MoE block BF16", func(t *testing.T) {
		const dModel, dFF, expertDFF, topK = 64, 128, 96, 2
		pinned, err := newPinnedNoCopyBytes(dModel * bf16Size)
		if err != nil {
			t.Fatalf("newPinnedNoCopyBytes: %v", err)
		}
		defer pinned.Close()

		scratch, err := getMoEBlockBF16Scratch(dModel, dFF, expertDFF, topK)
		if err != nil {
			t.Fatalf("getMoEBlockBF16Scratch: %v", err)
		}
		defer scratch.Close()

		outBuf, ok := scratch.outputView(pinned.bytes)
		if !ok {
			t.Fatal("MoE block BF16 output view did not accept pinned caller bytes")
		}
		requirePinnedOwnerOutputBuffer(t, "MoE block BF16", outBuf, pinned)
	})

	t.Run("MLP transform", func(t *testing.T) {
		const dModel, dFF = 64, 128
		pinned, err := newPinnedNoCopyBytes(dModel * bf16Size)
		if err != nil {
			t.Fatalf("newPinnedNoCopyBytes: %v", err)
		}
		defer pinned.Close()

		scratch, err := getMLPTransformScratch(dModel, dFF)
		if err != nil {
			t.Fatalf("getMLPTransformScratch: %v", err)
		}
		defer scratch.Close()

		outBuf, ok := scratch.outputView(pinned.bytes)
		if !ok {
			t.Fatal("MLP transform output view did not accept pinned caller bytes")
		}
		requirePinnedOwnerOutputBuffer(t, "MLP transform", outBuf, pinned)
	})

	t.Run("MLP transform mega", func(t *testing.T) {
		const dModel, dFF = 256, 512
		pinned, err := newPinnedNoCopyBytes(dModel * bf16Size)
		if err != nil {
			t.Fatalf("newPinnedNoCopyBytes: %v", err)
		}
		defer pinned.Close()

		scratch, err := getMLPTransformMegaScratch(dModel, dFF)
		if err != nil {
			t.Fatalf("getMLPTransformMegaScratch: %v", err)
		}
		defer scratch.Close()

		outBuf, ok := scratch.outputView(pinned.bytes)
		if !ok {
			t.Fatal("MLP transform mega output view did not accept pinned caller bytes")
		}
		requirePinnedOwnerOutputBuffer(t, "MLP transform mega", outBuf, pinned)
	})
}
