// SPDX-Licence-Identifier: EUPL-1.2

package inference

import (
	"encoding/binary"
	"testing"

	core "dappco.re/go"
)

func TestGGUF_ReadGGUFInfo_Good(t *testing.T) {
	path := writeMinimalGGUF(t, map[string]any{
		"general.architecture": "qwen3",
		"general.file_type":    uint32(15),
		"qwen3.block_count":    uint32(28),
		"qwen3.context_length": uint32(40960),
	})

	info, err := ReadGGUFInfo(path)

	checkNoError(t, err)
	checkEqual(t, "qwen3", info.Architecture)
	checkEqual(t, 4, info.QuantBits)
	checkEqual(t, 28, info.NumLayers)
	checkEqual(t, 40960, info.ContextLength)
}

func TestGGUF_ReadGGUFInfo_Bad(t *testing.T) {
	info, err := ReadGGUFInfo(core.JoinPath(t.TempDir(), "missing.gguf"))

	checkError(t, err)
	checkEqual(t, GGUFInfo{}, info)
}

func TestGGUF_DiscoverModels_Ugly(t *testing.T) {
	dir := t.TempDir()
	path := writeMinimalGGUFAt(t, core.JoinPath(dir, "model.gguf"), map[string]any{
		"general.architecture": "gemma4_text",
		"general.file_type":    uint32(7),
	})

	models := DiscoverModels(dir)

	checkLen(t, models, 1)
	checkEqual(t, path, models[0].Path)
	checkEqual(t, "gemma4_text", models[0].ModelType)
	checkEqual(t, "gguf", models[0].Format)
}

func writeMinimalGGUF(t *testing.T, metadata map[string]any) string {
	t.Helper()
	return writeMinimalGGUFAt(t, core.JoinPath(t.TempDir(), "model.gguf"), metadata)
}

func writeMinimalGGUFAt(t *testing.T, path string, metadata map[string]any) string {
	t.Helper()
	buf := core.NewBuffer()
	mustWrite := func(value any) {
		checkNoError(t, binary.Write(buf, binary.LittleEndian, value))
	}
	writeString := func(value string) {
		mustWrite(uint64(len(value)))
		_, err := buf.Write([]byte(value))
		checkNoError(t, err)
	}

	mustWrite(uint32(0x46554747))
	mustWrite(uint32(3))
	mustWrite(uint64(0))
	mustWrite(uint64(len(metadata)))
	for key, value := range metadata {
		writeString(key)
		switch typed := value.(type) {
		case string:
			mustWrite(uint32(8))
			writeString(typed)
		case uint32:
			mustWrite(uint32(4))
			mustWrite(typed)
		default:
			t.Fatalf("unsupported metadata test value %T", value)
		}
	}
	result := core.WriteFile(path, buf.Bytes(), 0o644)
	checkResultOK(t, result)
	return path
}
