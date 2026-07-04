// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"runtime"
	"slices"
	"sync"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/foundation"
	"github.com/tmc/apple/metal"
	"github.com/tmc/apple/objc"
)

type decodeForwardICBCoreScratch struct {
	dModel, qDim, kvDim, dFF, nLayers int
	asc                               attnScratch
	msc                               mlpScratch
	ping                              [2]metal.MTLBuffer
	hBufs                             []metal.MTLBuffer
	offBuf, nBuf                      metal.MTLBuffer
	offPtr, nPtr                      *int32
	kRopeCmd, vCmd                    []metal.MTLIndirectComputeCommand
	residentBufs                      []metal.MTLBuffer
	residentRes                       []metal.MTLResource
	residentIDs                       []objc.ID
	outputViewPtrs                    []uintptr
	outputViewLens                    []int
	outputViewBufs                    []metal.MTLBuffer
	outputViewPinned                  []*pinnedNoCopyBytes
}

type decodeForwardICBCoreScratchKey struct {
	dModel, qDim, kvDim, dFF, nLayers int
}

type decodeForwardICBCoreScratchPool struct {
	mu    sync.Mutex
	items []*decodeForwardICBCoreScratch
}

var decodeForwardICBCoreScratchPools sync.Map

type decodeForwardICBLayerProjBuffers struct {
	wq, wk, wv, wo, wg, wu, wd metal.MTLBuffer
}

type decodeForwardICBSetupScratch struct {
	anwBufs, mnwBufs []metal.MTLBuffer
	kCaches, vCaches []metal.MTLBuffer
	lb               []decodeForwardICBLayerProjBuffers
	projResident     []metal.MTLBuffer
}

var decodeForwardICBSetupScratchPool sync.Pool

func newDecodeForwardICBSetupScratch(nLayers int) *decodeForwardICBSetupScratch {
	return &decodeForwardICBSetupScratch{
		anwBufs:      make([]metal.MTLBuffer, nLayers),
		mnwBufs:      make([]metal.MTLBuffer, nLayers),
		kCaches:      make([]metal.MTLBuffer, nLayers),
		vCaches:      make([]metal.MTLBuffer, nLayers),
		lb:           make([]decodeForwardICBLayerProjBuffers, nLayers),
		projResident: make([]metal.MTLBuffer, 0, nLayers*7+19),
	}
}

func (s *decodeForwardICBSetupScratch) fits(nLayers int) bool {
	return s != nil &&
		cap(s.anwBufs) >= nLayers &&
		cap(s.mnwBufs) >= nLayers &&
		cap(s.kCaches) >= nLayers &&
		cap(s.vCaches) >= nLayers &&
		cap(s.lb) >= nLayers &&
		cap(s.projResident) >= nLayers*7+19
}

func (s *decodeForwardICBSetupScratch) reset(nLayers int) *decodeForwardICBSetupScratch {
	clear(s.anwBufs)
	clear(s.mnwBufs)
	clear(s.kCaches)
	clear(s.vCaches)
	clear(s.lb)
	clear(s.projResident)
	s.anwBufs = s.anwBufs[:nLayers]
	s.mnwBufs = s.mnwBufs[:nLayers]
	s.kCaches = s.kCaches[:nLayers]
	s.vCaches = s.vCaches[:nLayers]
	s.lb = s.lb[:nLayers]
	s.projResident = s.projResident[:0]
	return s
}

func getDecodeForwardICBSetupScratch(nLayers int) *decodeForwardICBSetupScratch {
	if v := decodeForwardICBSetupScratchPool.Get(); v != nil {
		s := v.(*decodeForwardICBSetupScratch)
		if s.fits(nLayers) {
			return s.reset(nLayers)
		}
	}
	return newDecodeForwardICBSetupScratch(nLayers)
}

func putDecodeForwardICBSetupScratch(s *decodeForwardICBSetupScratch) {
	if s != nil {
		decodeForwardICBSetupScratchPool.Put(s.reset(0))
	}
}

func newDecodeForwardICBCoreScratch(dModel, qDim, kvDim, dFF, nLayers int) *decodeForwardICBCoreScratch {
	hBufs := make([]metal.MTLBuffer, nLayers)
	for i := range hBufs {
		hBufs[i] = scratchBF16(dModel)
	}
	offBuf := device.NewBufferWithLengthOptions(4, metal.MTLResourceStorageModeShared)
	nBuf := device.NewBufferWithLengthOptions(4, metal.MTLResourceStorageModeShared)
	return &decodeForwardICBCoreScratch{
		dModel: dModel, qDim: qDim, kvDim: kvDim, dFF: dFF, nLayers: nLayers,
		asc:          newAttnScratch(dModel, qDim, kvDim, 0, 0),
		msc:          newMLPScratch(dModel, dFF),
		ping:         [2]metal.MTLBuffer{scratchBF16(dModel), scratchBF16(dModel)},
		hBufs:        hBufs,
		offBuf:       offBuf,
		nBuf:         nBuf,
		offPtr:       (*int32)(offBuf.Contents()),
		nPtr:         (*int32)(nBuf.Contents()),
		kRopeCmd:     make([]metal.MTLIndirectComputeCommand, nLayers),
		vCmd:         make([]metal.MTLIndirectComputeCommand, nLayers),
		residentBufs: make([]metal.MTLBuffer, 0, 12*nLayers+64),
		residentRes:  make([]metal.MTLResource, 0, 12*nLayers+64),
	}
}

func (s *decodeForwardICBCoreScratch) matches(dModel, qDim, kvDim, dFF, nLayers int) bool {
	if s == nil || s.dModel != dModel || s.qDim != qDim || s.kvDim != kvDim || s.dFF != dFF || s.nLayers != nLayers {
		return false
	}
	if s.asc.normed == nil || s.asc.q == nil || s.asc.qr == nil || s.asc.kProj == nil || s.asc.attn == nil || s.asc.attnOut == nil {
		return false
	}
	if s.msc.mlpNormed == nil || s.msc.gate == nil || s.msc.up == nil || s.msc.gated == nil || s.msc.down == nil {
		return false
	}
	if s.ping[0] == nil || s.ping[1] == nil || len(s.hBufs) != nLayers {
		return false
	}
	if s.offBuf == nil || s.nBuf == nil || s.offPtr == nil || s.nPtr == nil || len(s.kRopeCmd) != nLayers || len(s.vCmd) != nLayers {
		return false
	}
	for _, h := range s.hBufs {
		if h == nil {
			return false
		}
	}
	return true
}

func (s *decodeForwardICBCoreScratch) closeOutputViewAt(i int) {
	if s == nil || i < 0 || i >= len(s.outputViewBufs) {
		return
	}
	if i < len(s.outputViewPinned) && s.outputViewPinned[i] != nil {
		s.outputViewPinned[i].Close()
		s.outputViewPinned[i] = nil
	}
	s.outputViewPtrs[i] = 0
	s.outputViewLens[i] = 0
	s.outputViewBufs[i] = nil
}

func (s *decodeForwardICBCoreScratch) closeOutputViews() {
	if s == nil {
		return
	}
	for i := range s.outputViewBufs {
		s.closeOutputViewAt(i)
	}
	s.outputViewPtrs = nil
	s.outputViewLens = nil
	s.outputViewBufs = nil
	s.outputViewPinned = nil
}

func (s *decodeForwardICBCoreScratch) outputViews(outputs [][]byte, outLen int) ([]metal.MTLBuffer, bool) {
	if s == nil || outLen <= 0 || len(outputs) == 0 {
		return nil, false
	}
	for i := range outputs {
		if len(outputs[i]) != outLen {
			return nil, false
		}
	}
	T := len(outputs)
	if cap(s.outputViewBufs) < T {
		s.closeOutputViews()
		s.outputViewPtrs = make([]uintptr, T)
		s.outputViewLens = make([]int, T)
		s.outputViewBufs = make([]metal.MTLBuffer, T)
		s.outputViewPinned = make([]*pinnedNoCopyBytes, T)
	} else {
		for i := T; i < len(s.outputViewBufs); i++ {
			s.closeOutputViewAt(i)
		}
		s.outputViewPtrs = s.outputViewPtrs[:T]
		s.outputViewLens = s.outputViewLens[:T]
		s.outputViewBufs = s.outputViewBufs[:T]
		s.outputViewPinned = s.outputViewPinned[:T]
	}
	for i := range outputs {
		ptr := uintptr(unsafe.Pointer(&outputs[i][0]))
		if s.outputViewBufs[i] != nil && s.outputViewPtrs[i] == ptr && s.outputViewLens[i] == outLen {
			continue
		}
		s.closeOutputViewAt(i)
		if buf, ok := registeredPinnedNoCopyBytes(outputs[i]); ok {
			s.outputViewPtrs[i] = ptr
			s.outputViewLens[i] = outLen
			s.outputViewBufs[i] = buf
			s.outputViewPinned[i] = nil
			continue
		}
		buf, pinner, noCopy := residentNoCopyBytes(outputs[i])
		if !noCopy {
			if pinner != nil {
				pinner.Unpin()
			}
			return nil, false
		}
		pinned := &pinnedNoCopyBytes{bytes: outputs[i], buf: buf, pinner: pinner}
		runtime.SetFinalizer(pinned, (*pinnedNoCopyBytes).Close)
		s.outputViewPtrs[i] = ptr
		s.outputViewLens[i] = outLen
		s.outputViewBufs[i] = buf
		s.outputViewPinned[i] = pinned
	}
	return s.outputViewBufs, true
}

func decodeForwardICBCoreScratchPoolFor(dModel, qDim, kvDim, dFF, nLayers int) *decodeForwardICBCoreScratchPool {
	key := decodeForwardICBCoreScratchKey{dModel: dModel, qDim: qDim, kvDim: kvDim, dFF: dFF, nLayers: nLayers}
	if v, ok := decodeForwardICBCoreScratchPools.Load(key); ok {
		return v.(*decodeForwardICBCoreScratchPool)
	}
	pool := &decodeForwardICBCoreScratchPool{}
	actual, _ := decodeForwardICBCoreScratchPools.LoadOrStore(key, pool)
	return actual.(*decodeForwardICBCoreScratchPool)
}

func getDecodeForwardICBCoreScratch(dModel, qDim, kvDim, dFF, nLayers int) *decodeForwardICBCoreScratch {
	if s := decodeForwardICBCoreScratchPoolFor(dModel, qDim, kvDim, dFF, nLayers).Get(); s != nil {
		if s.matches(dModel, qDim, kvDim, dFF, nLayers) {
			return s
		}
	}
	return newDecodeForwardICBCoreScratch(dModel, qDim, kvDim, dFF, nLayers)
}

func putDecodeForwardICBCoreScratch(s *decodeForwardICBCoreScratch) {
	if s != nil {
		decodeForwardICBCoreScratchPoolFor(s.dModel, s.qDim, s.kvDim, s.dFF, s.nLayers).Put(s)
	}
}

func (p *decodeForwardICBCoreScratchPool) Get() *decodeForwardICBCoreScratch {
	p.mu.Lock()
	defer p.mu.Unlock()
	n := len(p.items)
	if n == 0 {
		return nil
	}
	s := p.items[n-1]
	p.items[n-1] = nil
	p.items = p.items[:n-1]
	return s
}

func (p *decodeForwardICBCoreScratchPool) Put(s *decodeForwardICBCoreScratch) {
	if s == nil {
		return
	}
	p.mu.Lock()
	p.items = append(p.items, s)
	p.mu.Unlock()
}

// decodeForwardICBCore is the backend-agnostic cache-grow ICB recorder + replay:
// it records the full N-layer decode stack (24 ops/layer) ONCE and replays it per
// token over a GROWING seq-major KV cache. The seven projections are the only ops
// that differ between a bf16 and a 4-bit layer, so they're recorded through the
// `recordProj` closure (gemv or qmv); everything else — rms, rope, sdpa, the gelu
// chain, the residual adds, the cache layout, the per-token rebind, the optimize
// pass and the single-submit replay — is shared here.
//
// recordProj(li, c, vec, out, outOff, p) records projection p of layer li at the
// already-barriered command c (reading vec, writing out at outOff bytes); vOutBind
// is the projection output's bind index (gemv 3 / qmv 4), re-set per token for the
// V cache row. projResident lists the backend's weight + scalar buffers so they're
// made resident. anwBufs/mnwBufs are the per-layer bf16 norm buffers (norms aren't
// quantised); kCaches/vCaches are the per-layer growing caches the caller created.
//
// The crux a fixed ICB can't express directly is the cache WRITE row, which
// advances every token. The lever (TestICBRebindOffset / TestQMVICB): an ICB
// command's bindings are recorded once, but re-setting ONE buffer offset between
// replays is cheap and takes effect. So per token only offBuf, nBuf and each
// layer's two cache-write offsets (K-RoPE @ idx 1, V projection @ vOutBind) change.
func decodeForwardICBCore(
	outputs [][]byte,
	inputs [][]byte,
	anwBufs, mnwBufs, kCaches, vCaches, projResident []metal.MTLBuffer,
	recordProj func(li int, c metal.MTLIndirectComputeCommand, vec, out metal.MTLBuffer, outOff uint, p projIndex),
	vOutBind uint,
	dModel, nHeads, nKVHeads, headDim, dFF, maxLen int,
	base, scale, eps float32,
	useCallerOut bool,
) ([][]byte, error) {
	nLayers, T := len(anwBufs), len(inputs)
	qDim, kvDim := nHeads*headDim, nKVHeads*headDim

	// shared (non-projection) ICB-capable pipelines
	rmsPSO, err := pipelineForICB("rmsbfloat16")
	if err != nil {
		return nil, err
	}
	ropePSO, err := ropePipelineICB(false)
	if err != nil {
		return nil, err
	}
	sdpaPSO, err := sdpaVectorPipelineICBForHeadDim(headDim)
	if err != nil {
		return nil, err
	}
	addPSO, err := pipelineForICB("vv_Addbfloat16")
	if err != nil {
		return nil, err
	}
	hasFusedGELU := gpuHasGeluKernel()
	var mulPSO, tanhPSO metal.MTLComputePipelineState
	var geluICBPSO metal.MTLComputePipelineState
	if hasFusedGELU {
		if geluICBPSO, err = geluPipelineICB(); err != nil {
			return nil, err
		}
	} else {
		mulPSO, err = pipelineForICB("vv_Multiplybfloat16")
		if err != nil {
			return nil, err
		}
		tanhPSO, err = pipelineForICB("v_Tanhbfloat16bfloat16")
		if err != nil {
			return nil, err
		}
	}

	outLen := dModel * bf16Size
	if cap(outputs) < T {
		outputs = make([][]byte, T)
	} else {
		outputs = outputs[:T]
	}
	for i := range outputs {
		if useCallerOut && cap(outputs[i]) >= outLen {
			outputs[i] = outputs[i][:outLen]
			continue
		}
		outputs[i] = make([]byte, outLen)
	}
	withAutoreleasePool(func() {
		sc := getDecodeForwardICBCoreScratch(dModel, qDim, kvDim, dFF, nLayers)

		// shared scratch + gelu constants + residual ping-pong
		normed := sc.asc.normed
		q, qr, kProj, attn := sc.asc.q, sc.asc.qr, sc.asc.kProj, sc.asc.attn
		attnOut := sc.asc.attnOut
		mlpNormed := sc.msc.mlpNormed
		gate, up := sc.msc.gate, sc.msc.up
		gated, down := sc.msc.gated, sc.msc.down
		var x2, x3, x3s, inner metal.MTLBuffer
		var scaled, tnh, onePlus, halfG metal.MTLBuffer
		var gelu metal.MTLBuffer
		var c044, c079, c1c, c05 metal.MTLBuffer
		if !hasFusedGELU {
			x2, x3, x3s, inner = sc.msc.x2, sc.msc.x3, sc.msc.x3s, sc.msc.inner
			scaled, tnh, onePlus, halfG = sc.msc.scaled, sc.msc.tnh, sc.msc.onePlus, sc.msc.halfG
			gelu = sc.msc.gelu
			c044, c079, c1c, c05 = sc.msc.c044, sc.msc.c079, sc.msc.c1, sc.msc.c05
		}
		ping := sc.ping
		hBufs := sc.hBufs

		// shared (non-projection) scalar buffers; offBuf + nBuf bumped per token
		offBuf, nBuf := sc.offBuf, sc.nBuf
		offPtr, nPtr := sc.offPtr, sc.nPtr
		epsBuf, axisBuf, wsBuf := scalarF32(eps), scalarI32(int32(dModel)), scalarI32(1)
		ropeScaleB := scalarF32(scale)
		ropeMatB := scalarI64(int64(headDim))
		ropeBaseB := scalarF32(float32(math.Log2(float64(base))))
		gqaB := scalarI32(int32(nHeads / nKVHeads))
		// seq-major cache strides: head jumps headDim, seq jumps kvDim (one row)
		khsB, kssB := scalarI64(int64(headDim)), scalarI64(int64(kvDim))
		vhsB, vssB := scalarI64(int64(headDim)), scalarI64(int64(kvDim))
		sdpaScaleB := scalarF32(scale)
		addModelB, cntFFB := scalarI32(int32(dModel)), scalarI32(int32(dFF))
		var tanhCntB metal.MTLBuffer
		if !hasFusedGELU {
			tanhCntB = scalarI32(int32(dFF))
		}

		resident := sc.residentBufs[:0]
		resident = append(resident,
			ping[0], ping[1], normed, q, qr, kProj, attn, attnOut, mlpNormed,
			gate, up, gated, down,
			offBuf, nBuf, epsBuf, axisBuf, wsBuf,
			ropeScaleB, ropeMatB, ropeBaseB, gqaB, khsB, kssB, vhsB, vssB, sdpaScaleB, addModelB, cntFFB,
		)
		if !hasFusedGELU {
			resident = append(resident,
				x2, x3, x3s, inner, scaled, tnh, onePlus, halfG, gelu,
				c044, c079, c1c, c05, tanhCntB,
			)
		}
		// reserve the upper-bound capacity for the appends that follow (projResident + 5 per-layer
		// buffer slices = 12 buffers/layer + the 19 projResident scalars) so the resident slice never
		// geometrically regrows. Grow changes capacity only — contents and kernel bindings unchanged.
		resident = slices.Grow(resident, 12*nLayers+20)
		resident = append(resident, projResident...)
		resident = append(resident, anwBufs...)
		resident = append(resident, mnwBufs...)
		resident = append(resident, kCaches...)
		resident = append(resident, vCaches...)
		resident = append(resident, hBufs...)

		opsPerLayer := 24
		if hasFusedGELU { // fused gelu is 1 command vs the composed chain's 10
			opsPerLayer = 15
		}
		total := opsPerLayer * nLayers
		icbDesc := metal.NewMTLIndirectCommandBufferDescriptor()
		icbDesc.SetCommandTypes(metal.MTLIndirectCommandTypeConcurrentDispatch)
		icbDesc.SetInheritBuffers(false)
		icbDesc.SetInheritPipelineState(false)
		icbDesc.SetMaxKernelBufferBindCount(16)
		icb := device.NewIndirectCommandBufferWithDescriptorMaxCommandCountOptions(icbDesc, uint(total), metal.MTLResourceStorageModeShared)

		rmsTG := uint(rmsSimdSize * ((((dModel + rmsNReads - 1) / rmsNReads) + rmsSimdSize - 1) / rmsSimdSize))
		setRMS := func(c metal.MTLIndirectComputeCommand, in, w, o metal.MTLBuffer) {
			emitRMSNorm(fastICBSink{c}, rmsPSO, in, w, o, 0, dModel, eps, rmsTG)
		}
		setBin := func(c metal.MTLIndirectComputeCommand, pso metal.MTLComputePipelineState, a, b, o, cntB metal.MTLBuffer, n int) {
			emitBinary(fastICBSink{c}, pso, a, 0, b, 0, o, 0, n)
		}

		// per-layer cache-write commands whose OUTPUT offset is re-set per token
		kRopeCmd := sc.kRopeCmd[:nLayers]
		vCmd := sc.vCmd[:nLayers]
		log2base := float32(math.Log2(float64(base)))
		var finalOutCmd metal.MTLIndirectComputeCommand

		for li := 0; li < nLayers; li++ {
			opBase := opsPerLayer * li
			inBuf, outBuf := ping[li%2], ping[(li+1)%2]
			hBuf := hBufs[li]
			cmd := func(op int) metal.MTLIndirectComputeCommand {
				c := indirectComputeCommandAtIndexFast(icb, uint(opBase+op))
				if opBase+op != 0 {
					setICBBarrier(c)
				}
				return c
			}
			// --- attention half with cache write (ops 0-8) ---
			setRMS(cmd(0), inBuf, anwBufs[li], normed)
			recordProj(li, cmd(1), normed, q, 0, projQ) // Q
			// 2: rope q -> qr
			c := cmd(2)
			emitRope(fastICBSink{c}, ropePSO, q, qr, 0, 0, offBuf, nil, nHeads, headDim, headDim, scale, log2base)
			recordProj(li, cmd(3), normed, kProj, 0, projK) // K -> kProj
			// 4: rope K -> kCache @ row pos  (OUTPUT OFFSET re-set per token)
			c = cmd(4)
			emitRope(fastICBSink{c}, ropePSO, kProj, kCaches[li], 0, 0, offBuf, nil, nKVHeads, headDim, headDim, scale, log2base)
			kRopeCmd[li] = c
			// 5: V projection -> vCache @ row pos  (OUTPUT OFFSET re-set per token)
			cv := cmd(5)
			recordProj(li, cv, normed, vCaches[li], 0, projV)
			vCmd[li] = cv
			// 6: sdpa over the grown window (N from nBuf; seq-major strides)
			c = cmd(6)
			emitSDPA(fastICBSink{c}, sdpaPSO, qr, kCaches[li], vCaches[li], attn, 0, nBuf, nHeads, nKVHeads, 0, int64(headDim), int64(kvDim), int64(headDim), int64(kvDim), scale)
			recordProj(li, cmd(7), attn, attnOut, 0, projO) // Wo
			setBin(cmd(8), addPSO, inBuf, attnOut, hBuf, addModelB, dModel)

			// --- MLP half (ops 9-23) ---
			setRMS(cmd(9), hBuf, mnwBufs[li], mlpNormed)
			recordProj(li, cmd(10), mlpNormed, gate, 0, projGate)
			recordProj(li, cmd(11), mlpNormed, up, 0, projUp)
			dpIdx := 22 // down-proj op index — follows the composed gelu (cmd 12-21)
			if hasFusedGELU {
				cg := cmd(12) // fused gelu(gate)·up — one command (cntFFB = dFF as the n buffer)
				emitBinary(fastICBSink{cg}, geluICBPSO, gate, 0, up, 0, gated, 0, dFF)
				dpIdx = 13
			} else {
				setBin(cmd(12), mulPSO, gate, gate, x2, cntFFB, dFF)
				setBin(cmd(13), mulPSO, x2, gate, x3, cntFFB, dFF)
				setBin(cmd(14), mulPSO, x3, c044, x3s, cntFFB, dFF)
				setBin(cmd(15), addPSO, gate, x3s, inner, cntFFB, dFF)
				setBin(cmd(16), mulPSO, inner, c079, scaled, cntFFB, dFF)
				ct := cmd(17)
				emitUnary(fastICBSink{ct}, tanhPSO, scaled, tnh, dFF)
				setBin(cmd(18), addPSO, tnh, c1c, onePlus, cntFFB, dFF)
				setBin(cmd(19), mulPSO, gate, c05, halfG, cntFFB, dFF)
				setBin(cmd(20), mulPSO, halfG, onePlus, gelu, cntFFB, dFF)
				setBin(cmd(21), mulPSO, gelu, up, gated, cntFFB, dFF)
			}
			recordProj(li, cmd(dpIdx), gated, down, 0, projDown) // Wdown
			c = cmd(dpIdx + 1)
			setBin(c, addPSO, hBuf, down, outBuf, addModelB, dModel)
			if li == nLayers-1 {
				finalOutCmd = c
			}
		}

		lastOut := ping[nLayers%2] // residual stream output after N ping-pong swaps
		ping0Ptr := (*byte)(ping[0].Contents())
		lastOutPtr := (*byte)(lastOut.Contents())
		var directOutputViews []metal.MTLBuffer
		directOutput := false
		if useCallerOut && finalOutCmd != nil {
			if views, ok := sc.outputViews(outputs, outLen); ok {
				directOutputViews = views
				directOutput = true
				resident = append(resident, directOutputViews...)
			} else {
				sc.closeOutputViews()
			}
		} else {
			sc.closeOutputViews()
		}
		if cap(sc.residentRes) < len(resident) {
			sc.residentRes = make([]metal.MTLResource, len(resident))
		}
		residentRes := sc.residentRes[:len(resident)]
		for i, b := range resident {
			residentRes[i] = b
		}
		sc.residentIDs = resourceIDsForFastUse(sc.residentIDs, residentRes)
		residentIDs := sc.residentIDs
		rng := foundation.NSRange{Location: 0, Length: uint(total)}

		// optimize the recorded ICB once (offset-only rebinds after don't re-optimize)
		optCb := commandBufferFast(queue)
		blit := blitCommandEncoderFast(optCb)
		optimizeIndirectCommandBufferWithRangeFast(blit, icb, rng)
		endBlitEncodingFast(blit)
		commitCommandBufferFast(optCb)
		waitUntilCompletedFast(optCb)

		rowBytes := kvDim * bf16Size
		for t := 0; t < T; t++ {
			*offPtr = int32(t)
			*nPtr = int32(t + 1)
			rowOff := uint(t * rowBytes)
			for li := 0; li < nLayers; li++ {
				// advance this token's cache-write row on the two recorded commands
				setICBKernelBuffer(kRopeCmd[li], kCaches[li], rowOff, 1)
				setICBKernelBuffer(vCmd[li], vCaches[li], rowOff, vOutBind)
			}
			if directOutput {
				setICBKernelBuffer(finalOutCmd, directOutputViews[t], 0, 2)
			}
			copy(unsafe.Slice(ping0Ptr, dModel*bf16Size), inputs[t])

			cb := commandBufferFast(queue)
			enc := computeCommandEncoderFast(cb)
			useResourcesIDsFast(enc, residentRes, residentIDs, metal.MTLResourceUsageRead|metal.MTLResourceUsageWrite)
			executeCommandsInBufferWithRangeFast(enc, icb, rng)
			endEncodingFast(enc)
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
			if profileForward {
				profForwardGPUSec += float64(cb.GPUEndTime() - cb.GPUStartTime())
			}
			if !directOutput {
				copy(outputs[t], unsafe.Slice(lastOutPtr, dModel*bf16Size))
			}
		}
		putDecodeForwardICBCoreScratch(sc)
	})
	return outputs, nil
}

// DecodeForwardICB is the bf16 cache-grow ICB: it builds a gemv recorder + the
// per-layer weight/cache buffers and runs the shared decodeForwardICBCore. Same
// signature/semantics as DecodeForward; byte-for-byte equal to it (gated). All bf16.
func DecodeForwardICB(
	inputs [][]byte, layers []DecodeLayerWeights,
	dModel, nHeads, nKVHeads, headDim, maxLen, dFF int,
	base, scale, eps float32,
) ([][]byte, error) {
	return decodeForwardICBInto(nil, inputs, layers, dModel, nHeads, nKVHeads, headDim, maxLen, dFF, base, scale, eps, false)
}

// DecodeForwardICBInto is DecodeForwardICB with caller-owned per-token output
// storage. Output slices with enough capacity are reused for the final host
// readback, avoiding per-token output allocation in streaming callers.
func DecodeForwardICBInto(
	outputs [][]byte, inputs [][]byte, layers []DecodeLayerWeights,
	dModel, nHeads, nKVHeads, headDim, maxLen, dFF int,
	base, scale, eps float32,
) ([][]byte, error) {
	return decodeForwardICBInto(outputs, inputs, layers, dModel, nHeads, nKVHeads, headDim, maxLen, dFF, base, scale, eps, true)
}

func decodeForwardICBInto(
	outputs [][]byte, inputs [][]byte, layers []DecodeLayerWeights,
	dModel, nHeads, nKVHeads, headDim, maxLen, dFF int,
	base, scale, eps float32,
	useCallerOut bool,
) ([][]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	nLayers, T := len(layers), len(inputs)
	if nLayers == 0 || T == 0 {
		return nil, core.NewError("native.DecodeForwardICB: need layers and inputs")
	}
	if T > maxLen {
		return nil, core.NewError("native.DecodeForwardICB: more tokens than maxLen cache rows")
	}
	qDim, kvDim := nHeads*headDim, nKVHeads*headDim
	for i := range inputs {
		if len(inputs[i]) != dModel*bf16Size {
			return nil, core.NewError("native.DecodeForwardICB: each input must be dModel bf16 bytes")
		}
	}
	for li := range layers {
		w := layers[li]
		if len(w.AttnNormW) != dModel*bf16Size || len(w.MLPNormW) != dModel*bf16Size ||
			len(w.WQ) != qDim*dModel*bf16Size || len(w.WO) != dModel*qDim*bf16Size ||
			len(w.WK) != kvDim*dModel*bf16Size || len(w.WV) != kvDim*dModel*bf16Size ||
			len(w.WGate) != dFF*dModel*bf16Size || len(w.WUp) != dFF*dModel*bf16Size || len(w.WDown) != dModel*dFF*bf16Size {
			return nil, core.NewError("native.DecodeForwardICB: layer weight size mismatch")
		}
	}

	// gemv ICB pipelines, one per distinct tile shape
	gemvPSO := func(inDim, outDim int) (metal.MTLComputePipelineState, int, int, int, int, error) {
		bm, bn, sm, sn, tm, tn := gemvTiles(inDim, outDim)
		p, e := pipelineForICB(gemvKernelName("bfloat16", bm, bn, sm, sn, tm, tn))
		return p, bm, bn, sm, tm, e
	}
	psoQ, bmQ, bnQ, smQ, tmQ, err := gemvPSO(dModel, qDim)
	if err != nil {
		return nil, err
	}
	psoKV, bmKV, bnKV, smKV, tmKV, err := gemvPSO(dModel, kvDim)
	if err != nil {
		return nil, err
	}
	psoO, bmO, bnO, smO, tmO, err := gemvPSO(qDim, dModel)
	if err != nil {
		return nil, err
	}
	psoF, bmF, bnF, smF, tmF, err := gemvPSO(dModel, dFF)
	if err != nil {
		return nil, err
	}
	psoD, bmD, bnD, smD, tmD, err := gemvPSO(dFF, dModel)
	if err != nil {
		return nil, err
	}

	var coreErr error
	withAutoreleasePool(func() {
		setup := getDecodeForwardICBSetupScratch(nLayers)
		anwBufs := setup.anwBufs
		mnwBufs := setup.mnwBufs
		kCaches := setup.kCaches
		vCaches := setup.vCaches
		lb := setup.lb
		cacheBytes := uint(maxLen * kvDim * bf16Size)
		// presized to the upper bound (every layer's 7 projection buffers, plus the 19 trailing
		// scalar buffers) so the per-forward build never geometrically regrows its backing array.
		// Byte-identical.
		projResident := setup.projResident
		for li := range layers {
			w := layers[li]
			anwBufs[li] = residentBytes(w.AttnNormW)
			mnwBufs[li] = residentBytes(w.MLPNormW)
			kCaches[li] = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
			vCaches[li] = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
			lb[li] = decodeForwardICBLayerProjBuffers{residentBytes(w.WQ), residentBytes(w.WK), residentBytes(w.WV), residentBytes(w.WO), residentBytes(w.WGate), residentBytes(w.WUp), residentBytes(w.WDown)}
			projResident = append(projResident, lb[li].wq, lb[li].wk, lb[li].wv, lb[li].wo, lb[li].wg, lb[li].wu, lb[li].wd)
		}
		// gemv scalar params (shared across layers)
		qInB, qOutB, qLdB := scalarI32(int32(dModel)), scalarI32(int32(qDim)), scalarI32(int32(dModel))
		kvInB, kvOutB, kvLdB := scalarI32(int32(dModel)), scalarI32(int32(kvDim)), scalarI32(int32(dModel))
		oInB, oOutB, oLdB := scalarI32(int32(qDim)), scalarI32(int32(dModel)), scalarI32(int32(qDim))
		fInB, fOutB, fLdB := scalarI32(int32(dModel)), scalarI32(int32(dFF)), scalarI32(int32(dModel))
		dInB, dOutB, dLdB := scalarI32(int32(dFF)), scalarI32(int32(dModel)), scalarI32(int32(dFF))
		bndB, bshB, vsB, msB := scalarI32(1), scalarI32(1), scalarI64(0), scalarI64(0)
		projResident = append(projResident, qInB, qOutB, qLdB, kvInB, kvOutB, kvLdB, oInB, oOutB, oLdB, fInB, fOutB, fLdB, dInB, dOutB, dLdB, bndB, bshB, vsB, msB)

		// bf16 tiled gemv through the SHARED emitGemv body (with encGemvBF16To); K/N/ld/batch bind memoised scalars.
		setGemv := func(c metal.MTLIndirectComputeCommand, pso metal.MTLComputePipelineState, mat, vec, o metal.MTLBuffer, outOff uint, inDim, outDim, bm, bn, sm, tm int) {
			emitGemv(fastICBSink{c}, pso, mat, 0, vec, o, outOff, inDim, outDim, bm, bn, sm, tm)
		}
		recordProj := func(li int, c metal.MTLIndirectComputeCommand, vec, out metal.MTLBuffer, outOff uint, p projIndex) {
			l := lb[li]
			switch p {
			case projQ:
				setGemv(c, psoQ, l.wq, vec, out, outOff, dModel, qDim, bmQ, bnQ, smQ, tmQ)
			case projK:
				setGemv(c, psoKV, l.wk, vec, out, outOff, dModel, kvDim, bmKV, bnKV, smKV, tmKV)
			case projV:
				setGemv(c, psoKV, l.wv, vec, out, outOff, dModel, kvDim, bmKV, bnKV, smKV, tmKV)
			case projO:
				setGemv(c, psoO, l.wo, vec, out, outOff, qDim, dModel, bmO, bnO, smO, tmO)
			case projGate:
				setGemv(c, psoF, l.wg, vec, out, outOff, dModel, dFF, bmF, bnF, smF, tmF)
			case projUp:
				setGemv(c, psoF, l.wu, vec, out, outOff, dModel, dFF, bmF, bnF, smF, tmF)
			case projDown:
				setGemv(c, psoD, l.wd, vec, out, outOff, dFF, dModel, bmD, bnD, smD, tmD)
			}
		}
		outputs, coreErr = decodeForwardICBCore(outputs, inputs, anwBufs, mnwBufs, kCaches, vCaches, projResident, recordProj, 3, dModel, nHeads, nKVHeads, headDim, dFF, maxLen, base, scale, eps, useCallerOut)
		putDecodeForwardICBSetupScratch(setup)
	})
	if coreErr != nil {
		return nil, coreErr
	}
	return outputs, nil
}
