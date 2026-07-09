// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"runtime"
	"unsafe"

	"github.com/tmc/apple/metal"
)

// noCopyOutputView is the shared no-copy Metal output-buffer view: it caches the
// pinned MTLBuffer for an output slice, re-resolving the pin only when the
// slice's pointer or length changes. Every kernel scratch struct with a
// caller-supplied output buffer embeds this — it is the identical
// outputView/closeOutputView lifecycle that was previously copy-pasted per
// struct.
type noCopyOutputView struct {
	outViewPtr    uintptr
	outViewLen    int
	outView       metal.MTLBuffer
	outViewPinned *pinnedNoCopyBytes
}

func (s *noCopyOutputView) closeOutputView() {
	if s == nil {
		return
	}
	if s.outViewPinned != nil {
		s.outViewPinned.Close()
	}
	s.outViewPtr = 0
	s.outViewLen = 0
	s.outView = nil
	s.outViewPinned = nil
}

func (s *noCopyOutputView) outputView(out []byte) (metal.MTLBuffer, bool) {
	if s == nil || len(out) == 0 {
		return nil, false
	}
	ptr := uintptr(unsafe.Pointer(&out[0]))
	if s.outView != nil && s.outViewPtr == ptr && s.outViewLen == len(out) {
		return s.outView, true
	}
	s.closeOutputView()
	if buf, ok := registeredPinnedNoCopyBytes(out); ok {
		s.outViewPtr = ptr
		s.outViewLen = len(out)
		s.outView = buf
		s.outViewPinned = nil
		return buf, true
	}
	buf, pinner, noCopy := residentNoCopyBytes(out)
	if !noCopy {
		if pinner != nil {
			pinner.Unpin()
		}
		return nil, false
	}
	pinned := &pinnedNoCopyBytes{bytes: out, buf: buf, pinner: pinner}
	runtime.SetFinalizer(pinned, (*pinnedNoCopyBytes).Close)
	s.outViewPtr = ptr
	s.outViewLen = len(out)
	s.outView = buf
	s.outViewPinned = pinned
	return buf, true
}
