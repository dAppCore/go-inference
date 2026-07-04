// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func TestHIPDriverFake_BadNilDriver(t *testing.T) {
	model, err := newHIPRuntime(nil).LoadModel("missing.gguf", validHIPDriverFakeLoadConfig())

	core.AssertError(t, err)
	core.AssertNil(t, model)
	core.AssertContains(t, err.Error(), "HIP driver is nil")
}

func TestHIPDriverFake_BadUnavailableDriver(t *testing.T) {
	driver := &fakeHIPDriver{available: false}

	model, err := newHIPRuntime(driver).LoadModel("missing.gguf", validHIPDriverFakeLoadConfig())

	core.AssertError(t, err)
	core.AssertNil(t, model)
	core.AssertContains(t, err.Error(), "HIP driver is not available")
	core.AssertEqual(t, 0, len(driver.allocations))
}

func TestHIPDriverFake_BadMallocFailure(t *testing.T) {
	driver := &failingHIPDriver{available: true, mallocErr: core.NewError("oom")}
	path, dataOffset := nativeHIPTensorGGUF(t)

	model, err := newHIPRuntime(driver).LoadModel(path, validHIPDriverFakeLoadConfigWithOffset(dataOffset))

	core.AssertError(t, err)
	core.AssertNil(t, model)
	core.AssertContains(t, err.Error(), "allocate tensor tok_embeddings.weight")
	core.AssertEqual(t, []uint64{16}, driver.allocations)
	core.AssertEqual(t, 0, len(driver.copies))
	core.AssertEqual(t, 0, len(driver.frees))
}

func TestHIPDriverFake_BadFreeFailureOnClose(t *testing.T) {
	driver := &failingHIPDriver{available: true, freeErr: core.NewError("free failed")}
	path, dataOffset := nativeHIPTensorGGUF(t)
	model, err := newHIPRuntime(driver).LoadModel(path, validHIPDriverFakeLoadConfigWithOffset(dataOffset))
	core.RequireNoError(t, err)

	err = model.Close()

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "free tensor")
	core.AssertEqual(t, 2, len(driver.frees))
}

func validHIPDriverFakeLoadConfig() nativeLoadConfig {
	return validHIPDriverFakeLoadConfigWithOffset(0)
}

func validHIPDriverFakeLoadConfigWithOffset(dataOffset int64) nativeLoadConfig {
	return nativeLoadConfig{
		ModelInfo:  inference.ModelInfo{Architecture: "qwen3"},
		DataOffset: dataOffset,
		Tensors: []nativeTensorInfo{
			{Name: "tok_embeddings.weight", Type: 0, Offset: 0, ByteSize: 16},
			{Name: "output.weight", Type: 0, Offset: 16, ByteSize: 16},
		},
	}
}

type failingHIPDriver struct {
	available   bool
	nextPointer nativeDevicePointer
	mallocErr   error
	freeErr     error
	copyErr     error
	allocations []uint64
	copies      []uint64
	frees       []nativeDevicePointer
}

func (driver *failingHIPDriver) Available() bool { return driver.available }
func (driver *failingHIPDriver) DeviceInfo() nativeDeviceInfo {
	return nativeDeviceInfo{Name: "fake"}
}
func (driver *failingHIPDriver) Malloc(size uint64) (nativeDevicePointer, error) {
	driver.allocations = append(driver.allocations, size)
	if driver.mallocErr != nil {
		return 0, driver.mallocErr
	}
	driver.nextPointer++
	return driver.nextPointer, nil
}
func (driver *failingHIPDriver) Free(pointer nativeDevicePointer) error {
	driver.frees = append(driver.frees, pointer)
	return driver.freeErr
}
func (driver *failingHIPDriver) CopyHostToDevice(_ nativeDevicePointer, data []byte) error {
	driver.copies = append(driver.copies, uint64(len(data)))
	return driver.copyErr
}
func (driver *failingHIPDriver) CopyDeviceToHost(_ nativeDevicePointer, data []byte) error {
	driver.copies = append(driver.copies, uint64(len(data)))
	return driver.copyErr
}
