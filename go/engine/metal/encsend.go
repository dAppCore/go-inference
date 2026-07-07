// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"runtime"
	"sync"
	"unsafe"

	basepurego "github.com/ebitengine/purego"
	"github.com/tmc/apple/foundation"
	"github.com/tmc/apple/metal"
	"github.com/tmc/apple/objc"
)

// Fast-path encoder setters for the hottest per-token Metal sends.
//
// WHY: tmc/apple's objc.Send has a zero-allocation fast path (msgSend0..8,
// pre-registered once at init) that triggers only when the return type is
// struct{}/uintptr AND every argument is uintptr-castable (ID, SEL, uint, …).
// Its generated MTLComputeCommandEncoder wrappers, however, pass the *interface*
// values MTLComputePipelineState and MTLBuffer straight through to Send. An
// interface value is not one of the tryFastArgs cases, so Send takes the slow
// path: purego/objc.Send, which re-declares a variadic func and calls
// purego.RegisterFunc (reflect.MakeFunc) on EVERY call. That reflect trampoline
// is the dominant per-token heap allocator on the no-cgo decode path (AX-11).
//
// These helpers do exactly what the slow path does — extract the raw objc.ID
// from the interface (the same GetID() the wrapper's slow path reaches) and
// issue the same objc_msgSend with the same arguments — but pack raw uintptr
// calls into a reusable purego ABI frame, so the generated wrapper's interface
// slow path and SyscallN's per-call frame allocation are both avoided.
// Byte-identical by construction: same selector, same receiver, same argument
// bits; only the dispatch mechanism differs.
//
// Scope is the interface-arg selectors that dominate encoder setup, scalar
// setBytes:length:atIndex:, plus the two MTLSize dispatch selectors.

var (
	selSetComputePipelineState = objc.Sel("setComputePipelineState:")
	selSetBufferOffsetAtIndex  = objc.Sel("setBuffer:offset:atIndex:")
	selSetKernelBufferAtIndex  = objc.Sel("setKernelBuffer:offset:atIndex:")
	selSetBytesLengthAtIndex   = objc.Sel("setBytes:length:atIndex:")
	selDispatchThreads         = objc.Sel("dispatchThreads:threadsPerThreadgroup:")
	selDispatchThreadgroups    = objc.Sel("dispatchThreadgroups:threadsPerThreadgroup:")
	selMemoryBarrierWithScope  = objc.Sel("memoryBarrierWithScope:")
	selConcurrentThreads       = objc.Sel("concurrentDispatchThreads:threadsPerThreadgroup:")
	selConcurrentThreadgroups  = objc.Sel("concurrentDispatchThreadgroups:threadsPerThreadgroup:")
	selCommandBuffer           = objc.Sel("commandBuffer")
	selComputeCommandEncoder   = objc.Sel("computeCommandEncoder")
	selBlitCommandEncoder      = objc.Sel("blitCommandEncoder")
	selEndEncoding             = objc.Sel("endEncoding")
	selCommit                  = objc.Sel("commit")
	selWaitUntilCompleted      = objc.Sel("waitUntilCompleted")
	selUseResourcesCountUsage  = objc.Sel("useResources:count:usage:")
	selExecuteICBWithRange     = objc.Sel("executeCommandsInBuffer:withRange:")
	selOptimizeICBWithRange    = objc.Sel("optimizeIndirectCommandBuffer:withRange:")
	selIndirectComputeCommand  = objc.Sel("indirectComputeCommandAtIndex:")
	selSetBarrier              = objc.Sel("setBarrier")
	selContents                = objc.Sel("contents")
	selBufferLength            = objc.Sel("length")
	objcMsgSendAddr            uintptr
	objcAutoreleasePoolPush    uintptr
	objcAutoreleasePoolPop     uintptr
	objcMsgSendOnce            sync.Once
	objcSyscallArgsPool        sync.Pool
)

// objcSyscallArgs mirrors purego.syscall15Args. The linknamed ABI trampoline
// reads the leading fields by offset, so keep that prefix in lockstep with
// purego v0.10.x. The trailing MTLSize slots are stable call-local storage for
// Darwin arm64 large-struct arguments, which objc_msgSend receives by pointer.
type objcSyscallArgs struct {
	fn, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15 uintptr
	f1, f2, f3, f4, f5, f6, f7, f8                                       uintptr
	arm64R8                                                              uintptr
	sizeA, sizeB                                                         metal.MTLSize
	scalar                                                               uint64
}

//go:linkname puregoRuntimeCGOCall runtime.cgocall
func puregoRuntimeCGOCall(fn uintptr, arg unsafe.Pointer) int32

//go:linkname puregoSyscall15XABI0 github.com/ebitengine/purego.syscall15XABI0
var puregoSyscall15XABI0 uintptr

func objcSyscallArgsGet() *objcSyscallArgs {
	if v := objcSyscallArgsPool.Get(); v != nil {
		return v.(*objcSyscallArgs)
	}
	return new(objcSyscallArgs)
}

func objcSyscallArgsPut(a *objcSyscallArgs) {
	*a = objcSyscallArgs{}
	objcSyscallArgsPool.Put(a)
}

func objcMsgSendRaw1(fn, id, sel, a1 uintptr) {
	args := objcSyscallArgsGet()
	args.fn = fn
	args.a1 = id
	args.a2 = sel
	args.a3 = a1
	puregoRuntimeCGOCall(puregoSyscall15XABI0, unsafe.Pointer(args))
	objcSyscallArgsPut(args)
}

func objcMsgSendRaw1Ret(fn, id, sel, a1 uintptr) uintptr {
	args := objcSyscallArgsGet()
	args.fn = fn
	args.a1 = id
	args.a2 = sel
	args.a3 = a1
	puregoRuntimeCGOCall(puregoSyscall15XABI0, unsafe.Pointer(args))
	rv := args.a1
	objcSyscallArgsPut(args)
	return rv
}

func objcMsgSendRaw0(fn, id, sel uintptr) uintptr {
	args := objcSyscallArgsGet()
	args.fn = fn
	args.a1 = id
	args.a2 = sel
	puregoRuntimeCGOCall(puregoSyscall15XABI0, unsafe.Pointer(args))
	rv := args.a1
	objcSyscallArgsPut(args)
	return rv
}

func puregoCallRaw0(fn uintptr) uintptr {
	args := objcSyscallArgsGet()
	args.fn = fn
	puregoRuntimeCGOCall(puregoSyscall15XABI0, unsafe.Pointer(args))
	rv := args.a1
	objcSyscallArgsPut(args)
	return rv
}

func puregoCallRaw1(fn, a1 uintptr) {
	args := objcSyscallArgsGet()
	args.fn = fn
	args.a1 = a1
	puregoRuntimeCGOCall(puregoSyscall15XABI0, unsafe.Pointer(args))
	objcSyscallArgsPut(args)
}

func objcMsgSendRaw3(fn, id, sel, a1, a2, a3 uintptr) {
	args := objcSyscallArgsGet()
	args.fn = fn
	args.a1 = id
	args.a2 = sel
	args.a3 = a1
	args.a4 = a2
	args.a5 = a3
	puregoRuntimeCGOCall(puregoSyscall15XABI0, unsafe.Pointer(args))
	objcSyscallArgsPut(args)
}

func objcMsgSendICBKernelBufferAtIndex(fn, icbID, cmdIdx, bufID, offset, index uintptr) {
	args := objcSyscallArgsGet()
	args.fn = fn
	args.a1 = icbID
	args.a2 = uintptr(selIndirectComputeCommand)
	args.a3 = cmdIdx
	puregoRuntimeCGOCall(puregoSyscall15XABI0, unsafe.Pointer(args))
	cmdID := args.a1
	args.fn = fn
	args.a1 = cmdID
	args.a2 = uintptr(selSetKernelBufferAtIndex)
	args.a3 = bufID
	args.a4 = offset
	args.a5 = index
	puregoRuntimeCGOCall(puregoSyscall15XABI0, unsafe.Pointer(args))
	objcSyscallArgsPut(args)
}

func objcMsgSendRawSize2(fn, id, sel uintptr, a1, a2 metal.MTLSize) {
	args := objcSyscallArgsGet()
	args.fn = fn
	args.a1 = id
	args.a2 = sel
	args.sizeA = a1
	args.sizeB = a2
	args.a3 = uintptr(unsafe.Pointer(&args.sizeA))
	args.a4 = uintptr(unsafe.Pointer(&args.sizeB))
	puregoRuntimeCGOCall(puregoSyscall15XABI0, unsafe.Pointer(args))
	objcSyscallArgsPut(args)
}

func objcMsgSendRawBytes4(fn, id, sel uintptr, bits uint32, index uintptr) {
	args := objcSyscallArgsGet()
	args.fn = fn
	args.a1 = id
	args.a2 = sel
	args.scalar = uint64(bits)
	args.a3 = uintptr(unsafe.Pointer(&args.scalar))
	args.a4 = 4
	args.a5 = index
	puregoRuntimeCGOCall(puregoSyscall15XABI0, unsafe.Pointer(args))
	objcSyscallArgsPut(args)
}

func objcMsgSendRawBytes8(fn, id, sel uintptr, bits uint64, index uintptr) {
	args := objcSyscallArgsGet()
	args.fn = fn
	args.a1 = id
	args.a2 = sel
	args.scalar = bits
	args.a3 = uintptr(unsafe.Pointer(&args.scalar))
	args.a4 = 8
	args.a5 = index
	puregoRuntimeCGOCall(puregoSyscall15XABI0, unsafe.Pointer(args))
	objcSyscallArgsPut(args)
}

func initObjCMsgSendStubs() {
	defer func() {
		if recover() != nil {
			objcMsgSendAddr = 0
			objcAutoreleasePoolPush = 0
			objcAutoreleasePoolPop = 0
		}
	}()
	h, err := basepurego.Dlopen("/usr/lib/libobjc.A.dylib", basepurego.RTLD_LAZY|basepurego.RTLD_GLOBAL)
	if err != nil {
		return
	}
	if addr, err := basepurego.Dlsym(h, "objc_msgSend"); err == nil {
		objcMsgSendAddr = addr
	}
	if addr, err := basepurego.Dlsym(h, "objc_autoreleasePoolPush"); err == nil {
		objcAutoreleasePoolPush = addr
	}
	if addr, err := basepurego.Dlsym(h, "objc_autoreleasePoolPop"); err == nil {
		objcAutoreleasePoolPop = addr
	}
}

// setPSO binds a compute pipeline state on enc via the zero-alloc fast send.
// Equivalent to enc.SetComputePipelineState(pso).
func setPSO(enc metal.MTLComputeCommandEncoder, pso metal.MTLComputePipelineState) {
	objcMsgSendOnce.Do(initObjCMsgSendStubs)
	if objcMsgSendAddr != 0 && puregoSyscall15XABI0 != 0 {
		objcMsgSendRaw1(objcMsgSendAddr, uintptr(enc.GetID()), uintptr(selSetComputePipelineState), uintptr(pso.GetID()))
		runtime.KeepAlive(enc)
		runtime.KeepAlive(pso)
		return
	}
	objc.Send[struct{}](enc.GetID(), selSetComputePipelineState, pso.GetID())
}

func setPSOObject(enc metal.MTLComputeCommandEncoderObject, pso metal.MTLComputePipelineState) {
	objcMsgSendOnce.Do(initObjCMsgSendStubs)
	if objcMsgSendAddr != 0 && puregoSyscall15XABI0 != 0 {
		objcMsgSendRaw1(objcMsgSendAddr, uintptr(enc.GetID()), uintptr(selSetComputePipelineState), uintptr(pso.GetID()))
		runtime.KeepAlive(enc)
		runtime.KeepAlive(pso)
		return
	}
	objc.Send[struct{}](enc.GetID(), selSetComputePipelineState, pso.GetID())
}

// setBuf binds buf at (offset, index) on enc via the zero-alloc fast send.
// Equivalent to enc.SetBufferWithOffsetAtIndex(buf, offset, index).
func setBuf(enc metal.MTLComputeCommandEncoder, buf metal.MTLBuffer, offset, index uint) {
	var bufID uintptr
	if buf != nil {
		bufID = uintptr(buf.GetID())
	}
	objcMsgSendOnce.Do(initObjCMsgSendStubs)
	if objcMsgSendAddr != 0 && puregoSyscall15XABI0 != 0 {
		objcMsgSendRaw3(objcMsgSendAddr, uintptr(enc.GetID()), uintptr(selSetBufferOffsetAtIndex), bufID, uintptr(offset), uintptr(index))
		runtime.KeepAlive(enc)
		runtime.KeepAlive(buf)
		return
	}
	objc.Send[struct{}](enc.GetID(), selSetBufferOffsetAtIndex, objc.ID(bufID), offset, index)
}

func setBufObject(enc metal.MTLComputeCommandEncoderObject, buf metal.MTLBuffer, offset, index uint) {
	var bufID uintptr
	if buf != nil {
		bufID = uintptr(buf.GetID())
	}
	objcMsgSendOnce.Do(initObjCMsgSendStubs)
	if objcMsgSendAddr != 0 && puregoSyscall15XABI0 != 0 {
		objcMsgSendRaw3(objcMsgSendAddr, uintptr(enc.GetID()), uintptr(selSetBufferOffsetAtIndex), bufID, uintptr(offset), uintptr(index))
		runtime.KeepAlive(enc)
		runtime.KeepAlive(buf)
		return
	}
	objc.Send[struct{}](enc.GetID(), selSetBufferOffsetAtIndex, objc.ID(bufID), offset, index)
}

// setBytes binds a small inline byte constant on enc via the zero-alloc fast send.
// Equivalent to enc.SetBytesLengthAtIndex(bytes, length, index). Metal copies the
// pointed bytes into the encoded command during the call, so stack scalar storage
// is valid here.
func setBytes(enc metal.MTLComputeCommandEncoder, ptr unsafe.Pointer, length, index uint) {
	objcMsgSendOnce.Do(initObjCMsgSendStubs)
	if objcMsgSendAddr != 0 && puregoSyscall15XABI0 != 0 {
		objcMsgSendRaw3(objcMsgSendAddr, uintptr(enc.GetID()), uintptr(selSetBytesLengthAtIndex), uintptr(ptr), uintptr(length), uintptr(index))
		runtime.KeepAlive(enc)
		runtime.KeepAlive(ptr)
		return
	}
	enc.SetBytesLengthAtIndex(unsafe.Slice((*byte)(ptr), length), length, index)
	runtime.KeepAlive(ptr)
}

func setBytesObject(enc metal.MTLComputeCommandEncoderObject, ptr unsafe.Pointer, length, index uint) {
	objcMsgSendOnce.Do(initObjCMsgSendStubs)
	if objcMsgSendAddr != 0 && puregoSyscall15XABI0 != 0 {
		objcMsgSendRaw3(objcMsgSendAddr, uintptr(enc.GetID()), uintptr(selSetBytesLengthAtIndex), uintptr(ptr), uintptr(length), uintptr(index))
		runtime.KeepAlive(enc)
		runtime.KeepAlive(ptr)
		return
	}
	enc.SetBytesLengthAtIndex(unsafe.Slice((*byte)(ptr), length), length, index)
	runtime.KeepAlive(ptr)
}

func setBytesI32(enc metal.MTLComputeCommandEncoder, v int32, index uint) {
	objcMsgSendOnce.Do(initObjCMsgSendStubs)
	if objcMsgSendAddr != 0 && puregoSyscall15XABI0 != 0 {
		objcMsgSendRawBytes4(objcMsgSendAddr, uintptr(enc.GetID()), uintptr(selSetBytesLengthAtIndex), uint32(v), uintptr(index))
		runtime.KeepAlive(enc)
		return
	}
	setBytesI32Slow(enc, v, index)
}

func setBytesI64(enc metal.MTLComputeCommandEncoder, v int64, index uint) {
	objcMsgSendOnce.Do(initObjCMsgSendStubs)
	if objcMsgSendAddr != 0 && puregoSyscall15XABI0 != 0 {
		objcMsgSendRawBytes8(objcMsgSendAddr, uintptr(enc.GetID()), uintptr(selSetBytesLengthAtIndex), uint64(v), uintptr(index))
		runtime.KeepAlive(enc)
		return
	}
	setBytesI64Slow(enc, v, index)
}

func setBytesF32(enc metal.MTLComputeCommandEncoder, v float32, index uint) {
	objcMsgSendOnce.Do(initObjCMsgSendStubs)
	if objcMsgSendAddr != 0 && puregoSyscall15XABI0 != 0 {
		objcMsgSendRawBytes4(objcMsgSendAddr, uintptr(enc.GetID()), uintptr(selSetBytesLengthAtIndex), math.Float32bits(v), uintptr(index))
		runtime.KeepAlive(enc)
		return
	}
	setBytesF32Slow(enc, v, index)
}

func setBytesI32Object(enc metal.MTLComputeCommandEncoderObject, v int32, index uint) {
	objcMsgSendOnce.Do(initObjCMsgSendStubs)
	if objcMsgSendAddr != 0 && puregoSyscall15XABI0 != 0 {
		objcMsgSendRawBytes4(objcMsgSendAddr, uintptr(enc.GetID()), uintptr(selSetBytesLengthAtIndex), uint32(v), uintptr(index))
		runtime.KeepAlive(enc)
		return
	}
	setBytesI32ObjectSlow(enc, v, index)
}

func setBytesI64Object(enc metal.MTLComputeCommandEncoderObject, v int64, index uint) {
	objcMsgSendOnce.Do(initObjCMsgSendStubs)
	if objcMsgSendAddr != 0 && puregoSyscall15XABI0 != 0 {
		objcMsgSendRawBytes8(objcMsgSendAddr, uintptr(enc.GetID()), uintptr(selSetBytesLengthAtIndex), uint64(v), uintptr(index))
		runtime.KeepAlive(enc)
		return
	}
	setBytesI64ObjectSlow(enc, v, index)
}

func setBytesF32Object(enc metal.MTLComputeCommandEncoderObject, v float32, index uint) {
	objcMsgSendOnce.Do(initObjCMsgSendStubs)
	if objcMsgSendAddr != 0 && puregoSyscall15XABI0 != 0 {
		objcMsgSendRawBytes4(objcMsgSendAddr, uintptr(enc.GetID()), uintptr(selSetBytesLengthAtIndex), math.Float32bits(v), uintptr(index))
		runtime.KeepAlive(enc)
		return
	}
	setBytesF32ObjectSlow(enc, v, index)
}

//go:noinline
func setBytesI32Slow(enc metal.MTLComputeCommandEncoder, v int32, index uint) {
	setBytes(enc, unsafe.Pointer(&v), 4, index)
}

//go:noinline
func setBytesI64Slow(enc metal.MTLComputeCommandEncoder, v int64, index uint) {
	setBytes(enc, unsafe.Pointer(&v), 8, index)
}

//go:noinline
func setBytesF32Slow(enc metal.MTLComputeCommandEncoder, v float32, index uint) {
	setBytes(enc, unsafe.Pointer(&v), 4, index)
}

//go:noinline
func setBytesI32ObjectSlow(enc metal.MTLComputeCommandEncoderObject, v int32, index uint) {
	setBytesObject(enc, unsafe.Pointer(&v), 4, index)
}

//go:noinline
func setBytesI64ObjectSlow(enc metal.MTLComputeCommandEncoderObject, v int64, index uint) {
	setBytesObject(enc, unsafe.Pointer(&v), 8, index)
}

//go:noinline
func setBytesF32ObjectSlow(enc metal.MTLComputeCommandEncoderObject, v float32, index uint) {
	setBytesObject(enc, unsafe.Pointer(&v), 4, index)
}

func commandBufferFast(q metal.MTLCommandQueue) metal.MTLCommandBufferObject {
	objcMsgSendOnce.Do(initObjCMsgSendStubs)
	if objcMsgSendAddr != 0 && puregoSyscall15XABI0 != 0 {
		rv := objcMsgSendRaw0(objcMsgSendAddr, uintptr(q.GetID()), uintptr(selCommandBuffer))
		runtime.KeepAlive(q)
		return metal.MTLCommandBufferObjectFromID(objc.ID(rv))
	}
	cb := q.CommandBuffer()
	return metal.MTLCommandBufferObjectFromID(cb.GetID())
}

func computeCommandEncoderFast(cb metal.MTLCommandBufferObject) metal.MTLComputeCommandEncoderObject {
	objcMsgSendOnce.Do(initObjCMsgSendStubs)
	if objcMsgSendAddr != 0 && puregoSyscall15XABI0 != 0 {
		rv := objcMsgSendRaw0(objcMsgSendAddr, uintptr(cb.GetID()), uintptr(selComputeCommandEncoder))
		runtime.KeepAlive(cb)
		return metal.MTLComputeCommandEncoderObjectFromID(objc.ID(rv))
	}
	enc := cb.ComputeCommandEncoder()
	return metal.MTLComputeCommandEncoderObjectFromID(enc.GetID())
}

func blitCommandEncoderFast(cb metal.MTLCommandBufferObject) metal.MTLBlitCommandEncoderObject {
	objcMsgSendOnce.Do(initObjCMsgSendStubs)
	if objcMsgSendAddr != 0 && puregoSyscall15XABI0 != 0 {
		rv := objcMsgSendRaw0(objcMsgSendAddr, uintptr(cb.GetID()), uintptr(selBlitCommandEncoder))
		runtime.KeepAlive(cb)
		return metal.MTLBlitCommandEncoderObjectFromID(objc.ID(rv))
	}
	blit := cb.BlitCommandEncoder()
	return metal.MTLBlitCommandEncoderObjectFromID(blit.GetID())
}

// concurrentComputeEncoderFast opens a CONCURRENT-dispatch compute encoder on cb: dispatches
// may overlap and the encoder orders NOTHING between them — callers place explicit
// memoryBarrierObject calls at every true dependency edge.
func concurrentComputeEncoderFast(cb metal.MTLCommandBufferObject) metal.MTLComputeCommandEncoderObject {
	pd := metal.NewMTLComputePassDescriptor()
	pd.SetDispatchType(metal.MTLDispatchTypeConcurrent)
	return metal.MTLComputeCommandEncoderObjectFromID(cb.ComputeCommandEncoderWithDescriptor(pd).GetID())
}

func endEncodingFast(enc metal.MTLComputeCommandEncoderObject) {
	objcMsgSendOnce.Do(initObjCMsgSendStubs)
	if objcMsgSendAddr != 0 && puregoSyscall15XABI0 != 0 {
		objcMsgSendRaw0(objcMsgSendAddr, uintptr(enc.GetID()), uintptr(selEndEncoding))
		runtime.KeepAlive(enc)
		return
	}
	enc.EndEncoding()
}

func endBlitEncodingFast(enc metal.MTLBlitCommandEncoderObject) {
	objcMsgSendOnce.Do(initObjCMsgSendStubs)
	if objcMsgSendAddr != 0 && puregoSyscall15XABI0 != 0 {
		objcMsgSendRaw0(objcMsgSendAddr, uintptr(enc.GetID()), uintptr(selEndEncoding))
		runtime.KeepAlive(enc)
		return
	}
	enc.EndEncoding()
}

func commitCommandBufferFast(cb metal.MTLCommandBufferObject) {
	objcMsgSendOnce.Do(initObjCMsgSendStubs)
	if objcMsgSendAddr != 0 && puregoSyscall15XABI0 != 0 {
		objcMsgSendRaw0(objcMsgSendAddr, uintptr(cb.GetID()), uintptr(selCommit))
		runtime.KeepAlive(cb)
		return
	}
	cb.Commit()
}

func waitUntilCompletedFast(cb metal.MTLCommandBufferObject) {
	objcMsgSendOnce.Do(initObjCMsgSendStubs)
	if objcMsgSendAddr != 0 && puregoSyscall15XABI0 != 0 {
		objcMsgSendRaw0(objcMsgSendAddr, uintptr(cb.GetID()), uintptr(selWaitUntilCompleted))
		runtime.KeepAlive(cb)
		return
	}
	cb.WaitUntilCompleted()
}

func bufferLengthFast(buf metal.MTLBuffer) uint {
	objcMsgSendOnce.Do(initObjCMsgSendStubs)
	if objcMsgSendAddr != 0 && puregoSyscall15XABI0 != 0 {
		n := objcMsgSendRaw0(objcMsgSendAddr, uintptr(buf.GetID()), uintptr(selBufferLength))
		runtime.KeepAlive(buf)
		return uint(n)
	}
	return buf.Length()
}

func bufferContentsFast(buf metal.MTLBuffer) unsafe.Pointer {
	objcMsgSendOnce.Do(initObjCMsgSendStubs)
	if objcMsgSendAddr != 0 && puregoSyscall15XABI0 != 0 {
		ptr := objcMsgSendRaw0(objcMsgSendAddr, uintptr(buf.GetID()), uintptr(selContents))
		runtime.KeepAlive(buf)
		return unsafePointerFromObjCReturn(ptr)
	}
	return buf.Contents()
}

func unsafePointerFromObjCReturn(ptr uintptr) unsafe.Pointer {
	return *(*unsafe.Pointer)(unsafe.Pointer(&ptr))
}

func useResourcesIDsFast(enc metal.MTLComputeCommandEncoder, resources []metal.MTLResource, ids []objc.ID, usage metal.MTLResourceUsage) {
	if len(ids) == 0 {
		return
	}
	ptr := unsafe.Pointer(unsafe.SliceData(ids))
	objcMsgSendOnce.Do(initObjCMsgSendStubs)
	if objcMsgSendAddr != 0 && puregoSyscall15XABI0 != 0 {
		objcMsgSendRaw3(objcMsgSendAddr, uintptr(enc.GetID()), uintptr(selUseResourcesCountUsage), uintptr(ptr), uintptr(len(ids)), uintptr(usage))
		runtime.KeepAlive(enc)
		runtime.KeepAlive(resources)
		runtime.KeepAlive(ids)
		return
	}
	objc.Send[struct{}](enc.GetID(), selUseResourcesCountUsage, ptr, uint(len(ids)), usage)
	runtime.KeepAlive(resources)
	runtime.KeepAlive(ids)
}

func useResourcesIDsFastObject(enc metal.MTLComputeCommandEncoderObject, resources []metal.MTLResource, ids []objc.ID, usage metal.MTLResourceUsage) {
	if len(ids) == 0 {
		return
	}
	ptr := unsafe.Pointer(unsafe.SliceData(ids))
	objcMsgSendOnce.Do(initObjCMsgSendStubs)
	if objcMsgSendAddr != 0 && puregoSyscall15XABI0 != 0 {
		objcMsgSendRaw3(objcMsgSendAddr, uintptr(enc.GetID()), uintptr(selUseResourcesCountUsage), uintptr(ptr), uintptr(len(ids)), uintptr(usage))
		runtime.KeepAlive(enc)
		runtime.KeepAlive(resources)
		runtime.KeepAlive(ids)
		return
	}
	objc.Send[struct{}](enc.GetID(), selUseResourcesCountUsage, ptr, uint(len(ids)), usage)
	runtime.KeepAlive(resources)
	runtime.KeepAlive(ids)
}

func resourceIDsForFastUse(dst []objc.ID, resources []metal.MTLResource) []objc.ID {
	if cap(dst) < len(resources) {
		dst = make([]objc.ID, len(resources))
	} else {
		dst = dst[:len(resources)]
	}
	for i, r := range resources {
		if r == nil {
			dst[i] = 0
			continue
		}
		dst[i] = r.GetID()
	}
	return dst
}

func executeCommandsInBufferWithRangeFast(enc metal.MTLComputeCommandEncoder, icb metal.MTLIndirectCommandBuffer, rng foundation.NSRange) {
	objcMsgSendOnce.Do(initObjCMsgSendStubs)
	if objcMsgSendAddr != 0 && puregoSyscall15XABI0 != 0 {
		objcMsgSendRaw3(objcMsgSendAddr, uintptr(enc.GetID()), uintptr(selExecuteICBWithRange), uintptr(icb.GetID()), uintptr(rng.Location), uintptr(rng.Length))
		runtime.KeepAlive(enc)
		runtime.KeepAlive(icb)
		return
	}
	objc.Send[struct{}](enc.GetID(), selExecuteICBWithRange, icb, rng)
}

func executeCommandsInBufferWithRangeObjectFast(enc metal.MTLComputeCommandEncoderObject, icb metal.MTLIndirectCommandBuffer, rng foundation.NSRange) {
	objcMsgSendOnce.Do(initObjCMsgSendStubs)
	if objcMsgSendAddr != 0 && puregoSyscall15XABI0 != 0 {
		objcMsgSendRaw3(objcMsgSendAddr, uintptr(enc.GetID()), uintptr(selExecuteICBWithRange), uintptr(icb.GetID()), uintptr(rng.Location), uintptr(rng.Length))
		runtime.KeepAlive(enc)
		runtime.KeepAlive(icb)
		return
	}
	objc.Send[struct{}](enc.GetID(), selExecuteICBWithRange, icb, rng)
}

func indirectComputeCommandAtIndexFast(icb metal.MTLIndirectCommandBuffer, idx uint) metal.MTLIndirectComputeCommand {
	objcMsgSendOnce.Do(initObjCMsgSendStubs)
	if objcMsgSendAddr != 0 && puregoSyscall15XABI0 != 0 {
		id := objcMsgSendRaw1Ret(objcMsgSendAddr, uintptr(icb.GetID()), uintptr(selIndirectComputeCommand), uintptr(idx))
		runtime.KeepAlive(icb)
		return metal.MTLIndirectComputeCommandObjectFromID(objc.ID(id))
	}
	return icb.IndirectComputeCommandAtIndex(idx)
}

func optimizeIndirectCommandBufferWithRangeFast(enc metal.MTLBlitCommandEncoderObject, icb metal.MTLIndirectCommandBuffer, rng foundation.NSRange) {
	objcMsgSendOnce.Do(initObjCMsgSendStubs)
	if objcMsgSendAddr != 0 && puregoSyscall15XABI0 != 0 {
		objcMsgSendRaw3(objcMsgSendAddr, uintptr(enc.GetID()), uintptr(selOptimizeICBWithRange), uintptr(icb.GetID()), uintptr(rng.Location), uintptr(rng.Length))
		runtime.KeepAlive(enc)
		runtime.KeepAlive(icb)
		return
	}
	objc.Send[struct{}](enc.GetID(), selOptimizeICBWithRange, icb, rng)
}

// dispatchCountForTest counts encoder dispatches while pieceTimingOn — the decode-piece
// diagnostic's "how many kernels per token" companion. Zero cost in production (one bool).
var dispatchCountForTest int64

// dispatchThreads binds the same dispatchThreads:threadsPerThreadgroup: call as
// the generated wrapper without routing MTLSize through objc.Send's reflect path.
func dispatchThreads(enc metal.MTLComputeCommandEncoder, grid, group metal.MTLSize) {
	if pieceTimingOn {
		dispatchCountForTest++
	}
	objcMsgSendOnce.Do(initObjCMsgSendStubs)
	if objcMsgSendAddr != 0 && puregoSyscall15XABI0 != 0 {
		objcMsgSendRawSize2(objcMsgSendAddr, uintptr(enc.GetID()), uintptr(selDispatchThreads), grid, group)
		runtime.KeepAlive(enc)
		return
	}
	enc.DispatchThreadsThreadsPerThreadgroup(grid, group)
}

func dispatchThreadsObject(enc metal.MTLComputeCommandEncoderObject, grid, group metal.MTLSize) {
	objcMsgSendOnce.Do(initObjCMsgSendStubs)
	if objcMsgSendAddr != 0 && puregoSyscall15XABI0 != 0 {
		objcMsgSendRawSize2(objcMsgSendAddr, uintptr(enc.GetID()), uintptr(selDispatchThreads), grid, group)
		runtime.KeepAlive(enc)
		return
	}
	enc.DispatchThreadsThreadsPerThreadgroup(grid, group)
}

// dispatchThreadgroups binds the same dispatchThreadgroups:threadsPerThreadgroup:
// call as the generated wrapper without routing MTLSize through objc.Send's reflect path.
func dispatchThreadgroups(enc metal.MTLComputeCommandEncoder, grid, group metal.MTLSize) {
	if pieceTimingOn {
		dispatchCountForTest++
	}
	objcMsgSendOnce.Do(initObjCMsgSendStubs)
	if objcMsgSendAddr != 0 && puregoSyscall15XABI0 != 0 {
		objcMsgSendRawSize2(objcMsgSendAddr, uintptr(enc.GetID()), uintptr(selDispatchThreadgroups), grid, group)
		runtime.KeepAlive(enc)
		return
	}
	enc.DispatchThreadgroupsThreadsPerThreadgroup(grid, group)
}

func dispatchThreadgroupsObject(enc metal.MTLComputeCommandEncoderObject, grid, group metal.MTLSize) {
	objcMsgSendOnce.Do(initObjCMsgSendStubs)
	if objcMsgSendAddr != 0 && puregoSyscall15XABI0 != 0 {
		objcMsgSendRawSize2(objcMsgSendAddr, uintptr(enc.GetID()), uintptr(selDispatchThreadgroups), grid, group)
		runtime.KeepAlive(enc)
		return
	}
	enc.DispatchThreadgroupsThreadsPerThreadgroup(grid, group)
}

func memoryBarrier(enc metal.MTLComputeCommandEncoder, scope metal.MTLBarrierScope) {
	objcMsgSendOnce.Do(initObjCMsgSendStubs)
	if objcMsgSendAddr != 0 && puregoSyscall15XABI0 != 0 {
		objcMsgSendRaw1(objcMsgSendAddr, uintptr(enc.GetID()), uintptr(selMemoryBarrierWithScope), uintptr(scope))
		runtime.KeepAlive(enc)
		return
	}
	enc.MemoryBarrierWithScope(scope)
}

func memoryBarrierObject(enc metal.MTLComputeCommandEncoderObject, scope metal.MTLBarrierScope) {
	objcMsgSendOnce.Do(initObjCMsgSendStubs)
	if objcMsgSendAddr != 0 && puregoSyscall15XABI0 != 0 {
		objcMsgSendRaw1(objcMsgSendAddr, uintptr(enc.GetID()), uintptr(selMemoryBarrierWithScope), uintptr(scope))
		runtime.KeepAlive(enc)
		return
	}
	enc.MemoryBarrierWithScope(scope)
}

func setICBPSO(cmd metal.MTLIndirectComputeCommand, pso metal.MTLComputePipelineState) {
	objcMsgSendOnce.Do(initObjCMsgSendStubs)
	if objcMsgSendAddr != 0 && puregoSyscall15XABI0 != 0 {
		objcMsgSendRaw1(objcMsgSendAddr, uintptr(cmd.GetID()), uintptr(selSetComputePipelineState), uintptr(pso.GetID()))
		runtime.KeepAlive(cmd)
		runtime.KeepAlive(pso)
		return
	}
	cmd.SetComputePipelineState(pso)
}

func setICBKernelBuffer(cmd metal.MTLIndirectComputeCommand, buf metal.MTLBuffer, offset, index uint) {
	var bufID uintptr
	if buf != nil {
		bufID = uintptr(buf.GetID())
	}
	objcMsgSendOnce.Do(initObjCMsgSendStubs)
	if objcMsgSendAddr != 0 && puregoSyscall15XABI0 != 0 {
		objcMsgSendRaw3(objcMsgSendAddr, uintptr(cmd.GetID()), uintptr(selSetKernelBufferAtIndex), bufID, uintptr(offset), uintptr(index))
		runtime.KeepAlive(cmd)
		runtime.KeepAlive(buf)
		return
	}
	cmd.SetKernelBufferOffsetAtIndex(buf, offset, index)
}

func setICBKernelBufferAtCommandIndexFast(icb metal.MTLIndirectCommandBuffer, cmdIdx uint, buf metal.MTLBuffer, offset, index uint) {
	var bufID uintptr
	if buf != nil {
		bufID = uintptr(buf.GetID())
	}
	objcMsgSendOnce.Do(initObjCMsgSendStubs)
	if objcMsgSendAddr != 0 && puregoSyscall15XABI0 != 0 {
		objcMsgSendICBKernelBufferAtIndex(objcMsgSendAddr, uintptr(icb.GetID()), uintptr(cmdIdx), bufID, uintptr(offset), uintptr(index))
		runtime.KeepAlive(icb)
		runtime.KeepAlive(buf)
		return
	}
	setICBKernelBuffer(icb.IndirectComputeCommandAtIndex(cmdIdx), buf, offset, index)
}

func setICBBarrier(cmd metal.MTLIndirectComputeCommand) {
	objcMsgSendOnce.Do(initObjCMsgSendStubs)
	if objcMsgSendAddr != 0 && puregoSyscall15XABI0 != 0 {
		objcMsgSendRaw0(objcMsgSendAddr, uintptr(cmd.GetID()), uintptr(selSetBarrier))
		runtime.KeepAlive(cmd)
		return
	}
	cmd.SetBarrier()
}

func concurrentDispatchThreads(cmd metal.MTLIndirectComputeCommand, grid, group metal.MTLSize) {
	objcMsgSendOnce.Do(initObjCMsgSendStubs)
	if objcMsgSendAddr != 0 && puregoSyscall15XABI0 != 0 {
		objcMsgSendRawSize2(objcMsgSendAddr, uintptr(cmd.GetID()), uintptr(selConcurrentThreads), grid, group)
		runtime.KeepAlive(cmd)
		return
	}
	cmd.ConcurrentDispatchThreadsThreadsPerThreadgroup(grid, group)
}

func concurrentDispatchThreadgroups(cmd metal.MTLIndirectComputeCommand, grid, group metal.MTLSize) {
	objcMsgSendOnce.Do(initObjCMsgSendStubs)
	if objcMsgSendAddr != 0 && puregoSyscall15XABI0 != 0 {
		objcMsgSendRawSize2(objcMsgSendAddr, uintptr(cmd.GetID()), uintptr(selConcurrentThreadgroups), grid, group)
		runtime.KeepAlive(cmd)
		return
	}
	cmd.ConcurrentDispatchThreadgroupsThreadsPerThreadgroup(grid, group)
}
