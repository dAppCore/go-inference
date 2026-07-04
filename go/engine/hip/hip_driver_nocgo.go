// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !cgo && !rocm_legacy_server

package hip

import (
	"runtime"
	"unsafe"

	core "dappco.re/go"
)

type unavailableHIPDriver struct{}

const rocmHIPPinnedHostCopySupported = false

func newSystemHIPDriver() nativeHIPDriver {
	return unavailableHIPDriver{}
}

func (unavailableHIPDriver) Available() bool { return false }
func (unavailableHIPDriver) DeviceInfo() nativeDeviceInfo {
	info, err := GetVRAMInfo()
	if err != nil {
		return nativeDeviceInfo{}
	}
	return nativeDeviceInfo{Name: "rocm", MemoryBytes: info.Total, FreeBytes: info.Free}
}
func (unavailableHIPDriver) Malloc(uint64) (nativeDevicePointer, error) {
	return 0, core.E("rocm.hip.Malloc", "cgo is disabled; native HIP driver is unavailable", nil)
}
func (unavailableHIPDriver) Free(nativeDevicePointer) error { return nil }
func (unavailableHIPDriver) CopyHostToDevice(nativeDevicePointer, []byte) error {
	return core.E("rocm.hip.CopyHostToDevice", "cgo is disabled; native HIP driver is unavailable", nil)
}
func (unavailableHIPDriver) CopyDeviceToHost(nativeDevicePointer, []byte) error {
	return core.E("rocm.hip.CopyDeviceToHost", "cgo is disabled; native HIP driver is unavailable", nil)
}

type nativeHIPPinnedHostToDevice interface {
	CopyPinnedHostToDevice(pointer nativeDevicePointer, host unsafe.Pointer, sizeBytes int) error
}

type nativeHIPLabeledPinnedHostToDevice interface {
	CopyPinnedHostToDeviceLabeled(pointer nativeDevicePointer, host unsafe.Pointer, sizeBytes int, operation, label string) error
}

func hipCopyPinnedHostToDevice(driver nativeHIPDriver, pointer nativeDevicePointer, data []byte) error {
	return hipCopyPinnedHostToDeviceLabeled(driver, pointer, data, "", "")
}

func hipCopyPinnedHostToDeviceLabeled(driver nativeHIPDriver, pointer nativeDevicePointer, data []byte, operation, label string) error {
	if len(data) == 0 {
		return nil
	}
	if pointer == 0 {
		return core.E("rocm.hip.CopyPinnedHostToDevice", "device pointer is nil", nil)
	}
	if labeled, ok := driver.(nativeHIPLabeledPinnedHostToDevice); ok {
		var view core.PinnedView
		core.PinSlice(data, &view)
		defer view.Release()
		if err := labeled.CopyPinnedHostToDeviceLabeled(pointer, view.Ptr(), view.Bytes(), operation, label); err != nil {
			return err
		}
		runtime.KeepAlive(data)
		return nil
	}
	if pinned, ok := driver.(nativeHIPPinnedHostToDevice); ok {
		var view core.PinnedView
		core.PinSlice(data, &view)
		defer view.Release()
		if err := pinned.CopyPinnedHostToDevice(pointer, view.Ptr(), view.Bytes()); err != nil {
			return err
		}
		runtime.KeepAlive(data)
		return nil
	}
	if operation != "" || label != "" {
		return hipCopyHostToDeviceLabeled(driver, pointer, data, operation, label)
	}
	return hipCopyHostToDevice(driver, pointer, data)
}
