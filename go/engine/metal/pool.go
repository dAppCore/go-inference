// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"runtime"

	"github.com/tmc/apple/objc"
)

// withAutoreleasePool runs fn inside an Objective-C autorelease pool, pinned to a
// single OS thread for the pool's lifetime.
//
// An autorelease pool is thread-local: objc_autoreleasePoolPush and the matching
// Pop must run on the SAME OS thread. objc.AutoreleasePool pushes, runs fn, then
// pops in a defer — but fn makes many purego/cgo calls, any of which is a Go
// scheduling point where the goroutine may migrate to another OS thread. Without
// pinning, the Pop can land on a different thread than the Push, corrupting the
// pool stack — an intermittent use-after-free crash during the drain. LockOSThread
// holds the goroutine on one thread across push→fn→pop, which is mandatory for
// objc autorelease pools driven from Go. Every native op funnels its GPU work
// through here.
func withAutoreleasePool(fn func()) {
	if pool, ok := beginAutoreleasePoolRaw(); ok {
		defer endAutoreleasePoolRaw(pool)
		fn()
		return
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	objc.AutoreleasePool(fn)
}

func beginAutoreleasePoolRaw() (uintptr, bool) {
	objcMsgSendOnce.Do(initObjCMsgSendStubs)
	if objcAutoreleasePoolPush == 0 || objcAutoreleasePoolPop == 0 || puregoSyscall15XABI0 == 0 {
		return 0, false
	}
	runtime.LockOSThread()
	return puregoCallRaw0(objcAutoreleasePoolPush), true
}

func endAutoreleasePoolRaw(pool uintptr) {
	puregoCallRaw1(objcAutoreleasePoolPop, pool)
	runtime.UnlockOSThread()
}
