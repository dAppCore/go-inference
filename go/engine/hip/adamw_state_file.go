// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"encoding/binary"
	"io"
	"math"
	"os"
	"strings"

	core "dappco.re/go"
)

const (
	nativeAdamWStateFileVersion uint32 = 1
	nativeAdamWStateMagic              = "ROCMADW1"
	nativeAdamWTrackMagic              = "ROCMADT1"
)

const (
	NativeAdamWTrackContainerBinary = "binary"
	NativeAdamWTrackContainerKV     = "kv"
	NativeAdamWTrackContainerMP4    = "mp4"
)

// NativeAdamWTrackRecord describes one append-only optimizer-state frame.
type NativeAdamWTrackRecord struct {
	Offset      int64 `json:"offset"`
	PayloadSize int   `json:"payload_size"`
	Step        int   `json:"step"`
}

// NativeAdamWTrackContainer returns the intended retained-state container for
// an optimizer track path.
func NativeAdamWTrackContainer(path string) string {
	switch lower := strings.ToLower(path); {
	case strings.HasSuffix(lower, ".kv"):
		return NativeAdamWTrackContainerKV
	case strings.HasSuffix(lower, ".mp4"):
		return NativeAdamWTrackContainerMP4
	default:
		return NativeAdamWTrackContainerBinary
	}
}

// SaveNativeAdamWState writes a single binary AdamW state snapshot.
func SaveNativeAdamWState(path string, state *NativeAdamWState) error {
	if path == "" {
		return core.NewError("rocm: AdamW state path is required")
	}
	payload, err := MarshalNativeAdamWState(state)
	if err != nil {
		return err
	}
	if err := ensureNativeAdamWStateDir(path); err != nil {
		return err
	}
	if result := core.WriteFile(path, payload, 0o644); !result.OK {
		return core.E("rocm.AdamW.Save", "write state", nativeAdamWResultError(result))
	}
	return nil
}

// LoadNativeAdamWState reads a single binary AdamW state snapshot.
func LoadNativeAdamWState(path string) (*NativeAdamWState, error) {
	if path == "" {
		return nil, core.NewError("rocm: AdamW state path is required")
	}
	read := core.ReadFile(path)
	if !read.OK {
		return nil, core.E("rocm.AdamW.Load", "read state", nativeAdamWResultError(read))
	}
	return UnmarshalNativeAdamWState(read.Value.([]byte))
}

// AppendNativeAdamWStateTrack appends one length-framed state snapshot to path.
func AppendNativeAdamWStateTrack(path string, state *NativeAdamWState) (NativeAdamWTrackRecord, error) {
	if path == "" {
		return NativeAdamWTrackRecord{}, core.NewError("rocm: AdamW track path is required")
	}
	payload, err := MarshalNativeAdamWState(state)
	if err != nil {
		return NativeAdamWTrackRecord{}, err
	}
	if err := ensureNativeAdamWStateDir(path); err != nil {
		return NativeAdamWTrackRecord{}, err
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return NativeAdamWTrackRecord{}, core.E("rocm.AdamW.Track", "open track", err)
	}
	defer file.Close()
	offset, err := file.Seek(0, io.SeekEnd)
	if err != nil {
		return NativeAdamWTrackRecord{}, core.E("rocm.AdamW.Track", "seek track", err)
	}
	var header [16]byte
	copy(header[:8], nativeAdamWTrackMagic)
	binary.LittleEndian.PutUint64(header[8:], uint64(len(payload)))
	if _, err := file.Write(header[:]); err != nil {
		return NativeAdamWTrackRecord{}, core.E("rocm.AdamW.Track", "write frame header", err)
	}
	if _, err := file.Write(payload); err != nil {
		return NativeAdamWTrackRecord{}, core.E("rocm.AdamW.Track", "write frame payload", err)
	}
	return NativeAdamWTrackRecord{Offset: offset, PayloadSize: len(payload), Step: state.Step}, nil
}

// LoadNativeAdamWStateTrackAt reads a state snapshot from a track frame offset.
func LoadNativeAdamWStateTrackAt(path string, offset int64) (*NativeAdamWState, error) {
	if path == "" {
		return nil, core.NewError("rocm: AdamW track path is required")
	}
	read := core.ReadFile(path)
	if !read.OK {
		return nil, core.E("rocm.AdamW.Track", "read track", nativeAdamWResultError(read))
	}
	data := read.Value.([]byte)
	if offset < 0 || offset > int64(len(data)) {
		return nil, core.NewError("rocm: AdamW track offset is out of range")
	}
	payload, _, err := readNativeAdamWTrackFrame(data[offset:])
	if err != nil {
		return nil, err
	}
	return UnmarshalNativeAdamWState(payload)
}

// ListNativeAdamWStateTrack records every complete frame in an append-only
// optimizer-state track without returning the full state payloads.
func ListNativeAdamWStateTrack(path string) ([]NativeAdamWTrackRecord, error) {
	if path == "" {
		return nil, core.NewError("rocm: AdamW track path is required")
	}
	read := core.ReadFile(path)
	if !read.OK {
		return nil, core.E("rocm.AdamW.Track", "read track", nativeAdamWResultError(read))
	}
	data := read.Value.([]byte)
	records := make([]NativeAdamWTrackRecord, 0, 8)
	for offset := int64(0); offset < int64(len(data)); {
		payload, consumed, err := readNativeAdamWTrackFrame(data[offset:])
		if err != nil {
			return nil, err
		}
		step, err := nativeAdamWStatePayloadStep(payload)
		if err != nil {
			return nil, err
		}
		records = append(records, NativeAdamWTrackRecord{Offset: offset, PayloadSize: len(payload), Step: step})
		offset += int64(consumed)
	}
	return records, nil
}

// FindNativeAdamWStateTrackStep returns the first frame record for a completed
// optimizer step in an append-only track.
func FindNativeAdamWStateTrackStep(path string, step int) (NativeAdamWTrackRecord, error) {
	if step < 0 {
		return NativeAdamWTrackRecord{}, core.NewError("rocm: AdamW track step must be non-negative")
	}
	records, err := ListNativeAdamWStateTrack(path)
	if err != nil {
		return NativeAdamWTrackRecord{}, err
	}
	for _, record := range records {
		if record.Step == step {
			return record, nil
		}
	}
	return NativeAdamWTrackRecord{}, core.Errorf("rocm: AdamW track step %d was not found", step)
}

// LoadNativeAdamWStateTrackStep reads the first state snapshot recorded for a
// completed optimizer step and returns its frame metadata.
func LoadNativeAdamWStateTrackStep(path string, step int) (*NativeAdamWState, NativeAdamWTrackRecord, error) {
	record, err := FindNativeAdamWStateTrackStep(path, step)
	if err != nil {
		return nil, NativeAdamWTrackRecord{}, err
	}
	state, err := LoadNativeAdamWStateTrackAt(path, record.Offset)
	if err != nil {
		return nil, NativeAdamWTrackRecord{}, err
	}
	return state, record, nil
}

func addNativeAdamWTrackLabels(labels map[string]string, trackPath string, record NativeAdamWTrackRecord) error {
	if labels == nil {
		return nil
	}
	records, err := ListNativeAdamWStateTrack(trackPath)
	if err != nil {
		return err
	}
	labels["optimizer_track"] = "append_only"
	labels["optimizer_track_format"] = "rocm_adamw_track_v1"
	labels["optimizer_track_container"] = NativeAdamWTrackContainer(trackPath)
	labels["optimizer_track_offset"] = core.Sprintf("%d", record.Offset)
	labels["optimizer_track_path"] = trackPath
	labels["optimizer_track_payload_bytes"] = core.Sprintf("%d", record.PayloadSize)
	labels["optimizer_track_step"] = core.Sprintf("%d", record.Step)
	labels["optimizer_track_frames"] = core.Sprintf("%d", len(records))
	labels["optimizer_track_list_helper"] = "ListNativeAdamWStateTrack"
	labels["optimizer_track_find_helper"] = "FindNativeAdamWStateTrackStep"
	labels["optimizer_track_load_step_helper"] = "LoadNativeAdamWStateTrackStep"
	return nil
}

// LoadLastNativeAdamWStateTrack reads the final complete frame in a track file.
func LoadLastNativeAdamWStateTrack(path string) (*NativeAdamWState, NativeAdamWTrackRecord, error) {
	state, record, _, err := loadLastNativeAdamWStateTrackWithFrameCount(path)
	return state, record, err
}

func loadLastNativeAdamWStateTrackWithFrameCount(path string) (*NativeAdamWState, NativeAdamWTrackRecord, int, error) {
	if path == "" {
		return nil, NativeAdamWTrackRecord{}, 0, core.NewError("rocm: AdamW track path is required")
	}
	read := core.ReadFile(path)
	if !read.OK {
		return nil, NativeAdamWTrackRecord{}, 0, core.E("rocm.AdamW.Track", "read track", nativeAdamWResultError(read))
	}
	data := read.Value.([]byte)
	var lastPayload []byte
	var last NativeAdamWTrackRecord
	frames := 0
	for offset := int64(0); offset < int64(len(data)); {
		payload, consumed, err := readNativeAdamWTrackFrame(data[offset:])
		if err != nil {
			return nil, NativeAdamWTrackRecord{}, 0, err
		}
		step, err := nativeAdamWStatePayloadStep(payload)
		if err != nil {
			return nil, NativeAdamWTrackRecord{}, 0, err
		}
		lastPayload = payload
		last = NativeAdamWTrackRecord{Offset: offset, PayloadSize: len(payload), Step: step}
		offset += int64(consumed)
		frames++
	}
	if lastPayload == nil {
		return nil, NativeAdamWTrackRecord{}, 0, core.NewError("rocm: AdamW track has no frames")
	}
	state, err := UnmarshalNativeAdamWState(lastPayload)
	if err != nil {
		return nil, NativeAdamWTrackRecord{}, 0, err
	}
	return state, last, frames, nil
}

func nativeAdamWStatePayloadStep(data []byte) (int, error) {
	headerLen := len(nativeAdamWStateMagic) + 4 + 8
	if len(data) < headerLen {
		return 0, core.NewError("rocm: AdamW state payload is incomplete")
	}
	if string(data[:len(nativeAdamWStateMagic)]) != nativeAdamWStateMagic {
		return 0, core.NewError("rocm: AdamW state magic is invalid")
	}
	version := binary.LittleEndian.Uint32(data[len(nativeAdamWStateMagic):])
	if version == 0 || version > nativeAdamWStateFileVersion {
		return 0, core.NewError("rocm: AdamW state version is unsupported")
	}
	step := binary.LittleEndian.Uint64(data[len(nativeAdamWStateMagic)+4:])
	if step > uint64(^uint(0)>>1) {
		return 0, core.NewError("rocm: AdamW state step is too large")
	}
	return int(step), nil
}

// MarshalNativeAdamWState encodes state as a portable little-endian binary blob.
func MarshalNativeAdamWState(state *NativeAdamWState) ([]byte, error) {
	if err := validateNativeAdamWState(state); err != nil {
		return nil, err
	}
	buf := core.NewBuffer()
	buf.WriteString(nativeAdamWStateMagic)
	writeUint32(buf, nativeAdamWStateFileVersion)
	writeUint64(buf, uint64(state.Step))
	writeUint32(buf, uint32(len(state.Layout)))
	writeUint64(buf, uint64(len(state.Slab)))
	writeFloat64(buf, state.Config.LearningRate)
	writeFloat64(buf, state.Config.Beta1)
	writeFloat64(buf, state.Config.Beta2)
	writeFloat64(buf, state.Config.Eps)
	writeFloat64(buf, state.Config.WeightDecay)
	if state.Config.Packed {
		buf.WriteByte(1)
	} else {
		buf.WriteByte(0)
	}
	for _, desc := range state.Layout {
		writeString(buf, desc.Name)
		writeUint64(buf, uint64(desc.Offset))
		writeUint64(buf, uint64(desc.Length))
		writeUint32(buf, uint32(len(desc.Shape)))
		for _, dim := range desc.Shape {
			writeUint64(buf, uint64(dim))
		}
	}
	for _, value := range state.Slab {
		writeUint32(buf, math.Float32bits(value))
	}
	return buf.Bytes(), nil
}

// UnmarshalNativeAdamWState decodes a binary state snapshot.
func UnmarshalNativeAdamWState(data []byte) (*NativeAdamWState, error) {
	reader := nativeAdamWStateReader{data: data}
	if string(reader.readBytes(len(nativeAdamWStateMagic))) != nativeAdamWStateMagic {
		return nil, core.NewError("rocm: AdamW state magic is invalid")
	}
	version := reader.readUint32()
	if version == 0 || version > nativeAdamWStateFileVersion {
		return nil, core.NewError("rocm: AdamW state version is unsupported")
	}
	step := reader.readUint64()
	layoutLen := reader.readUint32()
	slabLen := reader.readUint64()
	cfg := NativeAdamWConfig{}
	cfg.LearningRate = reader.readFloat64()
	cfg.Beta1 = reader.readFloat64()
	cfg.Beta2 = reader.readFloat64()
	cfg.Eps = reader.readFloat64()
	cfg.WeightDecay = reader.readFloat64()
	packed := reader.readByte()
	cfg.Packed = packed != 0
	layout := make([]NativeAdamWParamLayout, int(layoutLen))
	for i := range layout {
		name := reader.readString()
		offset := reader.readUint64()
		length := reader.readUint64()
		shapeLen := reader.readUint32()
		shape := make([]int, int(shapeLen))
		for j := range shape {
			shape[j] = int(reader.readUint64())
		}
		layout[i] = NativeAdamWParamLayout{Name: name, Offset: int(offset), Length: int(length), Shape: shape}
	}
	slab := make([]float32, int(slabLen))
	for i := range slab {
		slab[i] = math.Float32frombits(reader.readUint32())
	}
	if reader.err != nil {
		return nil, reader.err
	}
	if reader.remaining() != 0 {
		return nil, core.NewError("rocm: AdamW state has trailing bytes")
	}
	state := &NativeAdamWState{Config: cfg, Step: int(step), Layout: layout, Slab: slab}
	if err := validateNativeAdamWState(state); err != nil {
		return nil, err
	}
	return state, nil
}

func validateNativeAdamWState(state *NativeAdamWState) error {
	if state == nil {
		return core.NewError("rocm: AdamW state is nil")
	}
	if err := validateNativeAdamWConfig(state.Config); err != nil {
		return err
	}
	if state.Step < 0 {
		return core.NewError("rocm: AdamW state step must be non-negative")
	}
	total := stateTotalLen(state)
	if total == 0 || len(state.Slab) != total*3 {
		return core.NewError("rocm: AdamW packed slab shape is invalid")
	}
	if !rocmFloat32SliceFinite(state.Slab) {
		return core.NewError("rocm: AdamW slab values must be finite")
	}
	offset := 0
	for i, desc := range state.Layout {
		if desc.Offset != offset {
			return core.Errorf("rocm: AdamW layout %d offset %d does not match expected %d", i, desc.Offset, offset)
		}
		if desc.Length <= 0 {
			return core.Errorf("rocm: AdamW layout %d length must be positive", i)
		}
		if err := validateNativeAdamWShape(desc.Shape, desc.Length); err != nil {
			return core.E("rocm.AdamW.State", "layout shape", err)
		}
		offset += desc.Length
	}
	if offset != total {
		return core.Errorf("rocm: AdamW layout total %d does not match slab parameter length %d", offset, total)
	}
	return nil
}

func readNativeAdamWTrackFrame(data []byte) ([]byte, int, error) {
	if len(data) < 16 {
		return nil, 0, core.NewError("rocm: AdamW track frame header is incomplete")
	}
	if string(data[:8]) != nativeAdamWTrackMagic {
		return nil, 0, core.NewError("rocm: AdamW track magic is invalid")
	}
	size := int(binary.LittleEndian.Uint64(data[8:16]))
	if size <= 0 || 16+size > len(data) {
		return nil, 0, core.NewError("rocm: AdamW track frame payload is incomplete")
	}
	return data[16 : 16+size], 16 + size, nil
}

func ensureNativeAdamWStateDir(path string) error {
	dir := core.PathDir(path)
	if dir == "" || dir == "." {
		return nil
	}
	if result := core.MkdirAll(dir, 0o755); !result.OK {
		return core.E("rocm.AdamW.State", "create directory", nativeAdamWResultError(result))
	}
	return nil
}

func writeString(buf *core.Buffer, value string) {
	writeUint32(buf, uint32(len(value)))
	buf.WriteString(value)
}

func writeUint32(buf *core.Buffer, value uint32) {
	var payload [4]byte
	binary.LittleEndian.PutUint32(payload[:], value)
	buf.Write(payload[:])
}

func writeUint64(buf *core.Buffer, value uint64) {
	var payload [8]byte
	binary.LittleEndian.PutUint64(payload[:], value)
	buf.Write(payload[:])
}

func writeFloat64(buf *core.Buffer, value float64) {
	writeUint64(buf, math.Float64bits(value))
}

func nativeAdamWResultError(result core.Result) error {
	if result.OK {
		return nil
	}
	if err, ok := result.Value.(error); ok {
		return err
	}
	return core.NewError("core result failed")
}

type nativeAdamWStateReader struct {
	data  []byte
	index int
	err   error
}

func (reader *nativeAdamWStateReader) remaining() int {
	if reader == nil || reader.index >= len(reader.data) {
		return 0
	}
	return len(reader.data) - reader.index
}

func (reader *nativeAdamWStateReader) readBytes(size int) []byte {
	if reader.err != nil {
		return nil
	}
	if size < 0 || size > reader.remaining() {
		reader.err = core.NewError("rocm: AdamW state payload is incomplete")
		return nil
	}
	out := reader.data[reader.index : reader.index+size]
	reader.index += size
	return out
}

func (reader *nativeAdamWStateReader) readByte() byte {
	payload := reader.readBytes(1)
	if len(payload) == 0 {
		return 0
	}
	return payload[0]
}

func (reader *nativeAdamWStateReader) readUint32() uint32 {
	payload := reader.readBytes(4)
	if len(payload) < 4 {
		return 0
	}
	return binary.LittleEndian.Uint32(payload)
}

func (reader *nativeAdamWStateReader) readUint64() uint64 {
	payload := reader.readBytes(8)
	if len(payload) < 8 {
		return 0
	}
	return binary.LittleEndian.Uint64(payload)
}

func (reader *nativeAdamWStateReader) readFloat64() float64 {
	return math.Float64frombits(reader.readUint64())
}

func (reader *nativeAdamWStateReader) readString() string {
	size := reader.readUint32()
	payload := reader.readBytes(int(size))
	if reader.err != nil {
		return ""
	}
	return string(payload)
}
