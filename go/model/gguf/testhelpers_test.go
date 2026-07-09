// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"encoding/binary"
	"testing"

	core "dappco.re/go"

	"dappco.re/go/inference/model/safetensors"
)

// ggufMetaSpec describes one metadata key/value entry for writeTestGGUF.
type ggufMetaSpec struct {
	Key       string
	ValueType uint32
	Value     any
}

// ggufArraySpec describes a GGUF array-typed metadata value for
// writeTestGGUF/writeGGUFValue.
type ggufArraySpec struct {
	ElementType uint32
	Values      []any
}

// ggufTensorSpec describes one tensor-directory entry for writeTestGGUF.
// No tensor payload bytes are written — ReadInfo/Metadata/parseGGUF only
// read the directory, never the data section.
type ggufTensorSpec struct {
	Name string
	Type uint32
	Dims []uint64
}

// writeTestGGUF builds a minimal-but-real GGUF v3 file at path: header +
// metadata entries + tensor directory (zero data section — the read paths
// this package exercises never read tensor payload bytes).
func writeTestGGUF(t *testing.T, path string, metadata []ggufMetaSpec, tensors []ggufTensorSpec) {
	t.Helper()

	created := core.Create(path)
	if !created.OK {
		t.Fatalf("create gguf: %v", created.Value)
	}
	file := created.Value.(*core.OSFile)
	defer file.Close()

	write := func(value any) {
		t.Helper()
		if err := binary.Write(file, binary.LittleEndian, value); err != nil {
			t.Fatalf("binary write failed: %v", err)
		}
	}

	if _, err := file.Write([]byte("GGUF")); err != nil {
		t.Fatalf("write magic: %v", err)
	}
	write(uint32(3))
	write(uint64(len(tensors)))
	write(uint64(len(metadata)))

	for _, entry := range metadata {
		writeGGUFTestString(t, file, entry.Key)
		write(entry.ValueType)
		writeGGUFTestValue(t, file, entry.ValueType, entry.Value)
	}

	for _, tensor := range tensors {
		writeGGUFTestString(t, file, tensor.Name)
		write(uint32(len(tensor.Dims)))
		for _, dim := range tensor.Dims {
			write(dim)
		}
		write(tensor.Type)
		write(uint64(0)) // offset — unused by the read paths under test
	}
}

func writeGGUFTestString(t *testing.T, file *core.OSFile, value string) {
	t.Helper()
	if err := binary.Write(file, binary.LittleEndian, uint64(len(value))); err != nil {
		t.Fatalf("write string length: %v", err)
	}
	if _, err := file.Write([]byte(value)); err != nil {
		t.Fatalf("write string bytes: %v", err)
	}
}

func writeGGUFTestValue(t *testing.T, file *core.OSFile, valueType uint32, value any) {
	t.Helper()
	switch valueType {
	case ValueTypeString:
		stringValue, ok := value.(string)
		if !ok {
			t.Fatalf("write string: got %T, want string", value)
		}
		writeGGUFTestString(t, file, stringValue)
	case ValueTypeUint32:
		uint32Value, ok := value.(uint32)
		if !ok {
			t.Fatalf("write uint32: got %T, want uint32", value)
		}
		if err := binary.Write(file, binary.LittleEndian, uint32Value); err != nil {
			t.Fatalf("write uint32: %v", err)
		}
	case ValueTypeFloat32:
		floatValue, ok := value.(float32)
		if !ok {
			t.Fatalf("write float32: got %T, want float32", value)
		}
		if err := binary.Write(file, binary.LittleEndian, floatValue); err != nil {
			t.Fatalf("write float32: %v", err)
		}
	case ggufValueTypeBool:
		boolValue, ok := value.(bool)
		if !ok {
			t.Fatalf("write bool: got %T, want bool", value)
		}
		var encoded uint8
		if boolValue {
			encoded = 1
		}
		if err := binary.Write(file, binary.LittleEndian, encoded); err != nil {
			t.Fatalf("write bool: %v", err)
		}
	case ggufValueTypeArray:
		arrayValue, ok := value.(ggufArraySpec)
		if !ok {
			t.Fatalf("write array: got %T, want ggufArraySpec", value)
		}
		if err := binary.Write(file, binary.LittleEndian, arrayValue.ElementType); err != nil {
			t.Fatalf("write array element type: %v", err)
		}
		if err := binary.Write(file, binary.LittleEndian, uint64(len(arrayValue.Values))); err != nil {
			t.Fatalf("write array length: %v", err)
		}
		for _, item := range arrayValue.Values {
			writeGGUFTestValue(t, file, arrayValue.ElementType, item)
		}
	default:
		t.Fatalf("unsupported test gguf value type %d", valueType)
	}
}

// writeMinimalExampleGGUF writes a valid, minimal GGUF v3 file (a single
// general.architecture metadata entry, no tensors) at path. Runnable
// Example functions cannot take a *testing.T, so this variant reports
// errors via a return value rather than t.Fatalf.
func writeMinimalExampleGGUF(path, architecture string) error {
	created := core.Create(path)
	if !created.OK {
		return created.Err()
	}
	file := created.Value.(*core.OSFile)
	defer file.Close()

	write := func(value any) error {
		return binary.Write(file, binary.LittleEndian, value)
	}
	if _, err := file.Write([]byte("GGUF")); err != nil {
		return err
	}
	if err := write(uint32(3)); err != nil {
		return err
	}
	if err := write(uint64(0)); err != nil { // tensor count
		return err
	}
	if err := write(uint64(1)); err != nil { // metadata count
		return err
	}
	key := "general.architecture"
	if err := write(uint64(len(key))); err != nil {
		return err
	}
	if _, err := file.Write([]byte(key)); err != nil {
		return err
	}
	if err := write(uint32(ValueTypeString)); err != nil {
		return err
	}
	if err := write(uint64(len(architecture))); err != nil {
		return err
	}
	_, err := file.Write([]byte(architecture))
	return err
}

// writeTestSafetensors writes a valid safetensors file at path via
// safetensors.WriteSafetensors (F32 tensors only — the dtype
// QuantizeModelPack's test fixtures need).
func writeTestSafetensors(t *testing.T, path string, tensors map[string][]float32, shapes map[string][]int) {
	t.Helper()
	info := make(map[string]safetensors.SafetensorsTensorInfo, len(tensors))
	data := make(map[string][]byte, len(tensors))
	for name, values := range tensors {
		info[name] = safetensors.SafetensorsTensorInfo{Dtype: "F32", Shape: shapes[name]}
		data[name] = safetensors.EncodeFloat32(values)
	}
	if result := safetensors.WriteSafetensors(path, info, data); !result.OK {
		t.Fatalf("write test safetensors: %v", result.Value)
	}
}

// writeMinimalExampleSafetensors writes a single-tensor F32 safetensors
// file at path. Like writeMinimalExampleGGUF, this reports errors via a
// return value so runnable Example functions (which cannot take a
// *testing.T) can use it.
func writeMinimalExampleSafetensors(path, tensorName string, values []float32, shape []int) error {
	info := map[string]safetensors.SafetensorsTensorInfo{
		tensorName: {Dtype: "F32", Shape: shape},
	}
	data := map[string][]byte{tensorName: safetensors.EncodeFloat32(values)}
	if result := safetensors.WriteSafetensors(path, info, data); !result.OK {
		return result.Err()
	}
	return nil
}
