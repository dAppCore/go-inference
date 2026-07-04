// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"bytes"
	"encoding/binary"

	core "dappco.re/go"
)

const (
	rocmKVBlockRawVersion     uint32 = 1
	rocmKVBlockRawHeaderBytes        = 96
)

var rocmKVBlockRawMagic = [8]byte{'R', 'K', 'V', 'B', 'L', 'K', '1', 0}

func (cache *rocmKVCache) rawBlock(block rocmKVCacheBlock) ([]byte, error) {
	if cache == nil {
		return nil, core.E("rocm.KVCache.RawBlock", "cache is nil", nil)
	}
	keyPayload, err := block.key.deviceBytes()
	if err != nil {
		return nil, core.E("rocm.KVCache.RawBlock", "encode key tensor", err)
	}
	valuePayload, err := block.value.deviceBytes()
	if err != nil {
		return nil, core.E("rocm.KVCache.RawBlock", "encode value tensor", err)
	}
	keyEncoding, ok := rocmKVEncodingCode(block.key.encoding)
	if !ok {
		return nil, core.E("rocm.KVCache.RawBlock", "unsupported key tensor encoding", nil)
	}
	valueEncoding, ok := rocmKVEncodingCode(block.value.encoding)
	if !ok {
		return nil, core.E("rocm.KVCache.RawBlock", "unsupported value tensor encoding", nil)
	}
	if block.tokenStart < 0 || block.tokenCount <= 0 || block.keyWidth <= 0 || block.valueWidth <= 0 {
		return nil, core.E("rocm.KVCache.RawBlock", "invalid block metadata", nil)
	}
	if block.key.length != block.tokenCount*block.keyWidth || block.value.length != block.tokenCount*block.valueWidth {
		return nil, core.E("rocm.KVCache.RawBlock", "block tensor length mismatch", nil)
	}
	total := rocmKVBlockRawHeaderBytes + len(keyPayload) + len(valuePayload)
	payload := make([]byte, total)
	copy(payload[0:8], rocmKVBlockRawMagic[:])
	binary.LittleEndian.PutUint32(payload[8:], rocmKVBlockRawVersion)
	binary.LittleEndian.PutUint32(payload[12:], uint32(rocmKVBlockRawHeaderBytes))
	binary.LittleEndian.PutUint64(payload[16:], uint64(block.tokenStart))
	binary.LittleEndian.PutUint64(payload[24:], uint64(block.tokenCount))
	binary.LittleEndian.PutUint32(payload[32:], uint32(block.keyWidth))
	binary.LittleEndian.PutUint32(payload[36:], uint32(block.valueWidth))
	binary.LittleEndian.PutUint32(payload[40:], keyEncoding)
	binary.LittleEndian.PutUint32(payload[44:], valueEncoding)
	binary.LittleEndian.PutUint64(payload[48:], uint64(block.key.length))
	binary.LittleEndian.PutUint64(payload[56:], uint64(block.value.length))
	binary.LittleEndian.PutUint64(payload[64:], uint64(len(keyPayload)))
	binary.LittleEndian.PutUint64(payload[72:], uint64(len(valuePayload)))
	binary.LittleEndian.PutUint64(payload[80:], uint64(block.key.sizeBytes))
	binary.LittleEndian.PutUint64(payload[88:], uint64(block.value.sizeBytes))
	copy(payload[rocmKVBlockRawHeaderBytes:], keyPayload)
	copy(payload[rocmKVBlockRawHeaderBytes+len(keyPayload):], valuePayload)
	return payload, nil
}

func rocmKVCacheBlockFromRawPayload(payload []byte) (rocmKVCacheBlock, error) {
	meta, keyPayload, valuePayload, err := rocmKVBlockRawPayloadParts(payload)
	if err != nil {
		return rocmKVCacheBlock{}, err
	}
	return rocmKVCacheBlockFromRawParts(meta, keyPayload, valuePayload)
}

func rocmKVCacheBlockFromRawParts(meta rocmKVBlockRawMeta, keyPayload, valuePayload []byte) (rocmKVCacheBlock, error) {
	key, err := rocmKVTensorFromDeviceBytesRows(meta.keyEncoding, meta.keyLength, meta.tokenCount, keyPayload)
	if err != nil {
		return rocmKVCacheBlock{}, core.E("rocm.KVCache.RawBlock", "decode key tensor", err)
	}
	value, err := rocmKVTensorFromDeviceBytesRows(meta.valueEncoding, meta.valueLength, meta.tokenCount, valuePayload)
	if err != nil {
		return rocmKVCacheBlock{}, core.E("rocm.KVCache.RawBlock", "decode value tensor", err)
	}
	return rocmKVCacheBlock{
		tokenStart: meta.tokenStart,
		tokenCount: meta.tokenCount,
		keyWidth:   meta.keyWidth,
		valueWidth: meta.valueWidth,
		key:        key,
		value:      value,
	}, nil
}

func rocmKVCacheBlockPrefixFromRawPayload(payload []byte, prefixTokens int) (rocmKVCacheBlock, error) {
	meta, keyPayload, valuePayload, err := rocmKVBlockRawPayloadParts(payload)
	if err != nil {
		return rocmKVCacheBlock{}, err
	}
	return rocmKVCacheBlockPrefixFromRawParts(meta, keyPayload, valuePayload, prefixTokens)
}

func rocmKVCacheBlockPrefixFromRawParts(meta rocmKVBlockRawMeta, keyPayload, valuePayload []byte, prefixTokens int) (rocmKVCacheBlock, error) {
	if prefixTokens <= 0 || prefixTokens > meta.tokenCount {
		return rocmKVCacheBlock{}, core.E("rocm.KVCache.RawBlock", "prefix token count mismatch", nil)
	}
	if prefixTokens == meta.tokenCount {
		return rocmKVCacheBlockFromRawParts(meta, keyPayload, valuePayload)
	}
	key, err := rocmKVTensorPrefixFromDeviceBytesRows(meta.keyEncoding, meta.keyLength, meta.tokenCount, keyPayload, prefixTokens)
	if err != nil {
		return rocmKVCacheBlock{}, core.E("rocm.KVCache.RawBlock", "decode prefix key tensor", err)
	}
	value, err := rocmKVTensorPrefixFromDeviceBytesRows(meta.valueEncoding, meta.valueLength, meta.tokenCount, valuePayload, prefixTokens)
	if err != nil {
		return rocmKVCacheBlock{}, core.E("rocm.KVCache.RawBlock", "decode prefix value tensor", err)
	}
	return rocmKVCacheBlock{
		tokenStart: meta.tokenStart,
		tokenCount: prefixTokens,
		keyWidth:   meta.keyWidth,
		valueWidth: meta.valueWidth,
		key:        key,
		value:      value,
	}, nil
}

type rocmKVBlockRawMeta struct {
	tokenStart    int
	tokenCount    int
	keyWidth      int
	valueWidth    int
	keyEncoding   string
	valueEncoding string
	keyLength     int
	valueLength   int
	keyBytes      int
	valueBytes    int
}

func rocmKVBlockRawPayloadParts(payload []byte) (rocmKVBlockRawMeta, []byte, []byte, error) {
	if len(payload) < rocmKVBlockRawHeaderBytes {
		return rocmKVBlockRawMeta{}, nil, nil, core.E("rocm.KVCache.RawBlock", "raw block payload is too small", nil)
	}
	if !bytes.Equal(payload[0:8], rocmKVBlockRawMagic[:]) {
		return rocmKVBlockRawMeta{}, nil, nil, core.E("rocm.KVCache.RawBlock", "invalid raw block magic", nil)
	}
	if version := binary.LittleEndian.Uint32(payload[8:]); version != rocmKVBlockRawVersion {
		return rocmKVBlockRawMeta{}, nil, nil, core.E("rocm.KVCache.RawBlock", core.Sprintf("unsupported raw block version %d", version), nil)
	}
	headerBytes := binary.LittleEndian.Uint32(payload[12:])
	if headerBytes != rocmKVBlockRawHeaderBytes {
		return rocmKVBlockRawMeta{}, nil, nil, core.E("rocm.KVCache.RawBlock", "unsupported raw block header size", nil)
	}
	tokenStart, ok := rocmIntFromUint64("token start", binary.LittleEndian.Uint64(payload[16:]))
	if !ok {
		return rocmKVBlockRawMeta{}, nil, nil, core.E("rocm.KVCache.RawBlock", "token start is out of range", nil)
	}
	tokenCount, ok := rocmIntFromUint64("token count", binary.LittleEndian.Uint64(payload[24:]))
	if !ok {
		return rocmKVBlockRawMeta{}, nil, nil, core.E("rocm.KVCache.RawBlock", "token count is out of range", nil)
	}
	keyWidth := int(binary.LittleEndian.Uint32(payload[32:]))
	valueWidth := int(binary.LittleEndian.Uint32(payload[36:]))
	keyEncoding, ok := rocmKVEncodingFromCode(binary.LittleEndian.Uint32(payload[40:]))
	if !ok {
		return rocmKVBlockRawMeta{}, nil, nil, core.E("rocm.KVCache.RawBlock", "unsupported key tensor encoding", nil)
	}
	valueEncoding, ok := rocmKVEncodingFromCode(binary.LittleEndian.Uint32(payload[44:]))
	if !ok {
		return rocmKVBlockRawMeta{}, nil, nil, core.E("rocm.KVCache.RawBlock", "unsupported value tensor encoding", nil)
	}
	keyLength, ok := rocmIntFromUint64("key length", binary.LittleEndian.Uint64(payload[48:]))
	if !ok {
		return rocmKVBlockRawMeta{}, nil, nil, core.E("rocm.KVCache.RawBlock", "key length is out of range", nil)
	}
	valueLength, ok := rocmIntFromUint64("value length", binary.LittleEndian.Uint64(payload[56:]))
	if !ok {
		return rocmKVBlockRawMeta{}, nil, nil, core.E("rocm.KVCache.RawBlock", "value length is out of range", nil)
	}
	keyBytes, ok := rocmIntFromUint64("key bytes", binary.LittleEndian.Uint64(payload[64:]))
	if !ok {
		return rocmKVBlockRawMeta{}, nil, nil, core.E("rocm.KVCache.RawBlock", "key byte count is out of range", nil)
	}
	valueBytes, ok := rocmIntFromUint64("value bytes", binary.LittleEndian.Uint64(payload[72:]))
	if !ok {
		return rocmKVBlockRawMeta{}, nil, nil, core.E("rocm.KVCache.RawBlock", "value byte count is out of range", nil)
	}
	if tokenStart < 0 || tokenCount <= 0 || keyWidth <= 0 || valueWidth <= 0 || keyLength <= 0 || valueLength <= 0 || keyBytes <= 0 || valueBytes <= 0 {
		return rocmKVBlockRawMeta{}, nil, nil, core.E("rocm.KVCache.RawBlock", "invalid raw block metadata", nil)
	}
	if keyLength != tokenCount*keyWidth || valueLength != tokenCount*valueWidth {
		return rocmKVBlockRawMeta{}, nil, nil, core.E("rocm.KVCache.RawBlock", "raw block tensor length mismatch", nil)
	}
	expectedKeyBytes := rocmKVEncodedTensorPayloadBytesRows(keyEncoding, keyLength, tokenCount)
	expectedValueBytes := rocmKVEncodedTensorPayloadBytesRows(valueEncoding, valueLength, tokenCount)
	if keyBytes != expectedKeyBytes || valueBytes != expectedValueBytes {
		return rocmKVBlockRawMeta{}, nil, nil, core.E("rocm.KVCache.RawBlock", "raw block tensor byte count mismatch", nil)
	}
	payloadBytes := len(payload) - rocmKVBlockRawHeaderBytes
	if keyBytes > payloadBytes || valueBytes > payloadBytes-keyBytes {
		return rocmKVBlockRawMeta{}, nil, nil, core.E("rocm.KVCache.RawBlock", "raw block payload is truncated", nil)
	}
	end := rocmKVBlockRawHeaderBytes + keyBytes + valueBytes
	if end != len(payload) {
		return rocmKVBlockRawMeta{}, nil, nil, core.E("rocm.KVCache.RawBlock", "raw block payload has trailing bytes", nil)
	}
	meta := rocmKVBlockRawMeta{
		tokenStart:    tokenStart,
		tokenCount:    tokenCount,
		keyWidth:      keyWidth,
		valueWidth:    valueWidth,
		keyEncoding:   keyEncoding,
		valueEncoding: valueEncoding,
		keyLength:     keyLength,
		valueLength:   valueLength,
		keyBytes:      keyBytes,
		valueBytes:    valueBytes,
	}
	keyPayload := payload[rocmKVBlockRawHeaderBytes : rocmKVBlockRawHeaderBytes+keyBytes]
	valuePayload := payload[rocmKVBlockRawHeaderBytes+keyBytes : end]
	return meta, keyPayload, valuePayload, nil
}

func rocmKVEncodingCode(encoding string) (uint32, bool) {
	switch encoding {
	case rocmKVEncodingFP16:
		return 1, true
	case rocmKVEncodingQ8:
		return 2, true
	case rocmKVEncodingQ4:
		return 3, true
	case rocmKVEncodingQ8Rows:
		return 4, true
	case rocmKVEncodingQ4Rows:
		return 5, true
	case rocmKVEncodingQ8RowsI:
		return 6, true
	case rocmKVEncodingQ4RowsI:
		return 7, true
	default:
		return 0, false
	}
}

func rocmKVEncodingFromCode(code uint32) (string, bool) {
	switch code {
	case 1:
		return rocmKVEncodingFP16, true
	case 2:
		return rocmKVEncodingQ8, true
	case 3:
		return rocmKVEncodingQ4, true
	case 4:
		return rocmKVEncodingQ8Rows, true
	case 5:
		return rocmKVEncodingQ4Rows, true
	case 6:
		return rocmKVEncodingQ8RowsI, true
	case 7:
		return rocmKVEncodingQ4RowsI, true
	default:
		return "", false
	}
}

func rocmKVEncodedTensorPayloadBytes(encoding string, length int) int {
	return rocmKVEncodedTensorPayloadBytesRows(encoding, length, 1)
}

func rocmKVEncodedTensorPayloadBytesRows(encoding string, length, rows int) int {
	switch encoding {
	case rocmKVEncodingFP16:
		return length * 2
	case rocmKVEncodingQ8:
		return length + 4
	case rocmKVEncodingQ4:
		return (length+1)/2 + 4
	case rocmKVEncodingQ8Rows:
		if rows <= 0 {
			return -1
		}
		return length + rows*4
	case rocmKVEncodingQ8RowsI:
		if rows <= 0 || length%rows != 0 {
			return -1
		}
		rowWidth := length / rows
		return rows * (4 + rowWidth)
	case rocmKVEncodingQ4Rows:
		if rows <= 0 {
			return -1
		}
		return (length+1)/2 + rows*4
	case rocmKVEncodingQ4RowsI:
		if rows <= 0 || length%rows != 0 {
			return -1
		}
		rowWidth := length / rows
		return rows * (4 + (rowWidth+1)/2)
	default:
		return -1
	}
}

func rocmIntFromUint64(_ string, value uint64) (int, bool) {
	if value > uint64(int(^uint(0)>>1)) {
		return 0, false
	}
	return int(value), true
}
