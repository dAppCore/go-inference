// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"encoding/binary"

	core "dappco.re/go"
)

type hipDeviceTokenBuffer struct {
	driver    nativeHIPDriver
	pointer   nativeDevicePointer
	count     int
	sizeBytes uint64
	borrowed  bool
	closed    bool
}

func hipTokenIDsPayload(tokenIDs []int32) ([]byte, error) {
	if len(tokenIDs) == 0 {
		return nil, core.E("rocm.hip.Tokens", "token IDs are required", nil)
	}
	return hipTokenIDsPayloadInto(nil, tokenIDs)
}

func hipTokenIDsPayloadInto(payload []byte, tokenIDs []int32) ([]byte, error) {
	if len(tokenIDs) == 0 {
		return nil, core.E("rocm.hip.Tokens", "token IDs are required", nil)
	}
	byteCount := len(tokenIDs) * 4
	if cap(payload) < byteCount {
		payload = make([]byte, byteCount)
	} else {
		payload = payload[:byteCount]
	}
	for index, id := range tokenIDs {
		if id < 0 {
			return nil, core.E("rocm.hip.Tokens", "token IDs must be non-negative", nil)
		}
		binary.LittleEndian.PutUint32(payload[index*4:], uint32(id))
	}
	return payload, nil
}

func hipUploadTokenIDs(driver nativeHIPDriver, tokenIDs []int32) (*hipDeviceTokenBuffer, error) {
	if driver == nil {
		return nil, core.E("rocm.hip.Tokens", "HIP driver is nil", nil)
	}
	if !driver.Available() {
		return nil, core.E("rocm.hip.Tokens", "HIP driver is not available", nil)
	}
	payload, err := hipTokenIDsPayload(tokenIDs)
	if err != nil {
		return nil, err
	}
	pointer, err := hipMallocLabeled(driver, "rocm.hip.Tokens", "token buffer", uint64(len(payload)))
	if err != nil {
		return nil, core.E("rocm.hip.Tokens", "allocate token buffer", err)
	}
	if err := hipCopyHostToDeviceLabeled(driver, pointer, payload, "rocm.hip.Tokens", "token buffer"); err != nil {
		_ = driver.Free(pointer)
		return nil, core.E("rocm.hip.Tokens", "copy token buffer", err)
	}
	return &hipDeviceTokenBuffer{
		driver:    driver,
		pointer:   pointer,
		count:     len(tokenIDs),
		sizeBytes: uint64(len(payload)),
	}, nil
}

func hipWriteSingleTokenID(driver nativeHIPDriver, pointer nativeDevicePointer, tokenID int32) error {
	if driver == nil {
		return core.E("rocm.hip.Tokens", "HIP driver is nil", nil)
	}
	if !driver.Available() {
		return core.E("rocm.hip.Tokens", "HIP driver is not available", nil)
	}
	if pointer == 0 {
		return core.E("rocm.hip.Tokens", "token buffer is required", nil)
	}
	if tokenID < 0 {
		return core.E("rocm.hip.Tokens", "token IDs must be non-negative", nil)
	}
	var payload [4]byte
	binary.LittleEndian.PutUint32(payload[:], uint32(tokenID))
	if err := hipCopyHostToDeviceLabeled(driver, pointer, payload[:], "rocm.hip.Tokens", "single token buffer"); err != nil {
		return core.E("rocm.hip.Tokens", "copy token buffer", err)
	}
	return nil
}

func (buffer *hipDeviceTokenBuffer) Pointer() nativeDevicePointer {
	if buffer == nil || buffer.closed {
		return 0
	}
	return buffer.pointer
}

func (buffer *hipDeviceTokenBuffer) Count() int {
	if buffer == nil || buffer.closed {
		return 0
	}
	return buffer.count
}

func (buffer *hipDeviceTokenBuffer) SizeBytes() uint64 {
	if buffer == nil || buffer.closed {
		return 0
	}
	return buffer.sizeBytes
}

func (buffer *hipDeviceTokenBuffer) Close() error {
	if buffer == nil || buffer.closed {
		return nil
	}
	if buffer.pointer != 0 {
		if buffer.borrowed {
			buffer.pointer = 0
			buffer.closed = true
			return nil
		}
		if buffer.driver == nil {
			return core.E("rocm.hip.Tokens", "HIP driver is nil", nil)
		}
		if err := buffer.driver.Free(buffer.pointer); err != nil {
			return core.E("rocm.hip.Tokens", "free token buffer", err)
		}
		buffer.pointer = 0
	}
	buffer.closed = true
	return nil
}
