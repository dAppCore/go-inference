// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"bytes"

	"dappco.re/go/inference/jsonenc"
	"dappco.re/go/inference/model/state"
)

var (
	rocmKVManifestKeyKind           = []byte("kind")
	rocmKVManifestKeyMode           = []byte("mode")
	rocmKVManifestKeyBlockSize      = []byte("block_size")
	rocmKVManifestKeyTokenCount     = []byte("token_count")
	rocmKVManifestKeyBlocks         = []byte("blocks")
	rocmKVManifestKeyIndex          = []byte("index")
	rocmKVManifestKeyURI            = []byte("uri")
	rocmKVManifestKeyChunkID        = []byte("chunk_id")
	rocmKVManifestKeyState          = []byte("state")
	rocmKVManifestKeyTokenStart     = []byte("token_start")
	rocmKVManifestKeyKeyWidth       = []byte("key_width")
	rocmKVManifestKeyValueWidth     = []byte("value_width")
	rocmKVManifestKeySizeBytes      = []byte("size_bytes")
	rocmKVManifestKeyEncoding       = []byte("encoding")
	rocmKVManifestKeyFrameOffset    = []byte("frame_offset")
	rocmKVManifestKeyHasFrameOffset = []byte("has_frame_offset")
	rocmKVManifestKeyCodec          = []byte("codec")
	rocmKVManifestKeySegment        = []byte("segment")

	rocmKVManifestValueBlockBundleKind = []byte(rocmKVBlockBundleKind)
	rocmKVManifestValueFP16            = []byte(rocmKVCacheModeFP16)
	rocmKVManifestValueQ8              = []byte(rocmKVCacheModeQ8)
	rocmKVManifestValueKQ8VQ4          = []byte(rocmKVCacheModeKQ8VQ4)
	rocmKVManifestValueRawBlock        = []byte(rocmKVBlockRawEncoding)
	rocmKVManifestValueSnapshot        = []byte(rocmKVSnapshotEncoding)
	rocmKVManifestValueCodecMemory     = []byte(state.CodecMemory)
	rocmKVManifestValueCodecStateVideo = []byte(state.CodecStateVideo)
	rocmKVManifestValueCodecMemvid     = []byte("memvid/qr-video")
	rocmKVManifestValueCodecFile       = []byte("state/file-log")
	rocmKVManifestValueCodecMemvidFile = []byte("memvid/file-log")
)

type rocmKVWakeKnownString struct {
	value string
	raw   []byte
}

type rocmKVBlockBundleWakeHeader struct {
	Kind        string
	Mode        string
	BlockSize   int
	TokenCount  int
	BlocksIndex int
}

var (
	rocmKVWakeKnownBlockBundleKind = []rocmKVWakeKnownString{{value: rocmKVBlockBundleKind, raw: rocmKVManifestValueBlockBundleKind}}
	rocmKVWakeKnownCacheModes      = []rocmKVWakeKnownString{
		{value: rocmKVCacheModeFP16, raw: rocmKVManifestValueFP16},
		{value: rocmKVCacheModeQ8, raw: rocmKVManifestValueQ8},
		{value: rocmKVCacheModeKQ8VQ4, raw: rocmKVManifestValueKQ8VQ4},
	}
	rocmKVWakeKnownBlockEncodings = []rocmKVWakeKnownString{
		{value: rocmKVBlockRawEncoding, raw: rocmKVManifestValueRawBlock},
		{value: rocmKVSnapshotEncoding, raw: rocmKVManifestValueSnapshot},
	}
	rocmKVWakeKnownStateCodecs = []rocmKVWakeKnownString{
		{value: state.CodecMemory, raw: rocmKVManifestValueCodecMemory},
		{value: state.CodecStateVideo, raw: rocmKVManifestValueCodecStateVideo},
		{value: "memvid/qr-video", raw: rocmKVManifestValueCodecMemvid},
		{value: "state/file-log", raw: rocmKVManifestValueCodecFile},
		{value: "memvid/file-log", raw: rocmKVManifestValueCodecMemvidFile},
	}
)

func (bundle *rocmKVBlockBundleWakeSnapshot) UnmarshalJSON(data []byte) error {
	*bundle = rocmKVBlockBundleWakeSnapshot{}
	i, err := jsonenc.MatchObjectStart(data, 0)
	if err != nil {
		return err
	}
	i = jsonenc.SkipJSONWhitespace(data, i)
	if i < len(data) && data[i] == '}' {
		return nil
	}
	for {
		i = jsonenc.SkipJSONWhitespace(data, i)
		if i >= len(data) || data[i] != '"' {
			return jsonenc.ErrInvalidJSON
		}
		key, next, err := jsonenc.ParseJSONStringRaw(data, i)
		if err != nil {
			return err
		}
		i = jsonenc.SkipJSONWhitespace(data, next)
		if i >= len(data) || data[i] != ':' {
			return jsonenc.ErrInvalidJSON
		}
		i = jsonenc.SkipJSONWhitespace(data, i+1)
		i, err = bundle.unmarshalWakeField(data, i, key)
		if err != nil {
			return err
		}
		i = jsonenc.SkipJSONWhitespace(data, i)
		if i >= len(data) {
			return jsonenc.ErrInvalidJSON
		}
		if data[i] == ',' {
			i++
			continue
		}
		if data[i] == '}' {
			return nil
		}
		return jsonenc.ErrInvalidJSON
	}
}

func (bundle *rocmKVBlockBundleWakeSnapshot) unmarshalWakeField(data []byte, i int, key []byte) (int, error) {
	switch {
	case bytes.Equal(key, rocmKVManifestKeyKind):
		s, next, err := parseROCmKVWakeKnownString(data, i, rocmKVWakeKnownBlockBundleKind)
		bundle.Kind = s
		return next, err
	case bytes.Equal(key, rocmKVManifestKeyMode):
		s, next, err := parseROCmKVWakeKnownString(data, i, rocmKVWakeKnownCacheModes)
		bundle.Mode = s
		return next, err
	case bytes.Equal(key, rocmKVManifestKeyBlockSize):
		n, next, err := jsonenc.ParseJSONInt(data, i)
		bundle.BlockSize = int(n)
		return next, err
	case bytes.Equal(key, rocmKVManifestKeyTokenCount):
		n, next, err := jsonenc.ParseJSONInt(data, i)
		bundle.TokenCount = int(n)
		return next, err
	case bytes.Equal(key, rocmKVManifestKeyBlocks):
		blocks, next, err := parseROCmKVBlockBundleWakeRefs(data, i)
		bundle.Blocks = blocks
		return next, err
	default:
		return jsonenc.SkipJSONValue(data, i)
	}
}

func parseROCmKVBlockBundleWakeRefs(data []byte, i int) ([]rocmKVBlockBundleWakeRef, int, error) {
	i, err := jsonenc.MatchArrayStart(data, i)
	if err != nil {
		return nil, i, err
	}
	refs := make([]rocmKVBlockBundleWakeRef, 0, jsonenc.CountJSONArrayElements(data, i))
	i = jsonenc.SkipJSONWhitespace(data, i)
	if i < len(data) && data[i] == ']' {
		return refs, i + 1, nil
	}
	for {
		ref, next, err := parseROCmKVBlockBundleWakeRef(data, i)
		if err != nil {
			return nil, next, err
		}
		refs = append(refs, ref)
		i = jsonenc.SkipJSONWhitespace(data, next)
		if i >= len(data) {
			return nil, i, jsonenc.ErrInvalidJSON
		}
		if data[i] == ',' {
			i++
			continue
		}
		if data[i] == ']' {
			return refs, i + 1, nil
		}
		return nil, i, jsonenc.ErrInvalidJSON
	}
}

func parseROCmKVBlockBundleWakeHeader(data []byte) (rocmKVBlockBundleWakeHeader, error) {
	var header rocmKVBlockBundleWakeHeader
	i, err := jsonenc.MatchObjectStart(data, 0)
	if err != nil {
		return header, err
	}
	i = jsonenc.SkipJSONWhitespace(data, i)
	if i < len(data) && data[i] == '}' {
		return header, nil
	}
	for {
		i = jsonenc.SkipJSONWhitespace(data, i)
		if i >= len(data) || data[i] != '"' {
			return header, jsonenc.ErrInvalidJSON
		}
		key, next, err := jsonenc.ParseJSONStringRaw(data, i)
		if err != nil {
			return header, err
		}
		i = jsonenc.SkipJSONWhitespace(data, next)
		if i >= len(data) || data[i] != ':' {
			return header, jsonenc.ErrInvalidJSON
		}
		i = jsonenc.SkipJSONWhitespace(data, i+1)
		switch {
		case bytes.Equal(key, rocmKVManifestKeyKind):
			header.Kind, i, err = parseROCmKVWakeKnownString(data, i, rocmKVWakeKnownBlockBundleKind)
		case bytes.Equal(key, rocmKVManifestKeyMode):
			header.Mode, i, err = parseROCmKVWakeKnownString(data, i, rocmKVWakeKnownCacheModes)
		case bytes.Equal(key, rocmKVManifestKeyBlockSize):
			var n int64
			n, i, err = jsonenc.ParseJSONInt(data, i)
			header.BlockSize = int(n)
		case bytes.Equal(key, rocmKVManifestKeyTokenCount):
			var n int64
			n, i, err = jsonenc.ParseJSONInt(data, i)
			header.TokenCount = int(n)
		case bytes.Equal(key, rocmKVManifestKeyBlocks):
			header.BlocksIndex = i
			i, err = jsonenc.SkipJSONValue(data, i)
		default:
			i, err = jsonenc.SkipJSONValue(data, i)
		}
		if err != nil {
			return header, err
		}
		i = jsonenc.SkipJSONWhitespace(data, i)
		if i >= len(data) {
			return header, jsonenc.ErrInvalidJSON
		}
		if data[i] == ',' {
			i++
			continue
		}
		if data[i] == '}' {
			return header, nil
		}
		return header, jsonenc.ErrInvalidJSON
	}
}

func forEachROCmKVBlockBundleWakeRef(data []byte, i int, yield func(rocmKVBlockBundleWakeRef) (bool, error)) error {
	i, err := jsonenc.MatchArrayStart(data, i)
	if err != nil {
		return err
	}
	i = jsonenc.SkipJSONWhitespace(data, i)
	if i < len(data) && data[i] == ']' {
		return nil
	}
	for {
		ref, next, err := parseROCmKVBlockBundleWakeRef(data, i)
		if err != nil {
			return err
		}
		cont, err := yield(ref)
		if err != nil {
			return err
		}
		if !cont {
			return nil
		}
		i = jsonenc.SkipJSONWhitespace(data, next)
		if i >= len(data) {
			return jsonenc.ErrInvalidJSON
		}
		if data[i] == ',' {
			i++
			continue
		}
		if data[i] == ']' {
			return nil
		}
		return jsonenc.ErrInvalidJSON
	}
}

func parseROCmKVBlockBundleWakeRef(data []byte, i int) (rocmKVBlockBundleWakeRef, int, error) {
	var ref rocmKVBlockBundleWakeRef
	i, err := jsonenc.MatchObjectStart(data, i)
	if err != nil {
		return ref, i, err
	}
	i = jsonenc.SkipJSONWhitespace(data, i)
	if i < len(data) && data[i] == '}' {
		return ref, i + 1, nil
	}
	for {
		i = jsonenc.SkipJSONWhitespace(data, i)
		if i >= len(data) || data[i] != '"' {
			return ref, i, jsonenc.ErrInvalidJSON
		}
		key, next, err := jsonenc.ParseJSONStringRaw(data, i)
		if err != nil {
			return ref, next, err
		}
		i = jsonenc.SkipJSONWhitespace(data, next)
		if i >= len(data) || data[i] != ':' {
			return ref, i, jsonenc.ErrInvalidJSON
		}
		i = jsonenc.SkipJSONWhitespace(data, i+1)
		i, err = ref.unmarshalWakeField(data, i, key)
		if err != nil {
			return ref, i, err
		}
		i = jsonenc.SkipJSONWhitespace(data, i)
		if i >= len(data) {
			return ref, i, jsonenc.ErrInvalidJSON
		}
		if data[i] == ',' {
			i++
			continue
		}
		if data[i] == '}' {
			return ref, i + 1, nil
		}
		return ref, i, jsonenc.ErrInvalidJSON
	}
}

func (ref *rocmKVBlockBundleWakeRef) unmarshalWakeField(data []byte, i int, key []byte) (int, error) {
	switch {
	case bytes.Equal(key, rocmKVManifestKeyIndex):
		n, next, err := jsonenc.ParseJSONInt(data, i)
		ref.Index = int(n)
		return next, err
	case bytes.Equal(key, rocmKVManifestKeyURI):
		raw, next, err := jsonenc.ParseJSONStringRaw(data, i)
		ref.uriRaw = raw
		return next, err
	case bytes.Equal(key, rocmKVManifestKeyChunkID):
		n, next, err := jsonenc.ParseJSONInt(data, i)
		ref.ChunkID = int(n)
		return next, err
	case bytes.Equal(key, rocmKVManifestKeyState):
		st, next, err := parseROCmKVBlockBundleWakeStateRef(data, i)
		ref.State = st
		return next, err
	case bytes.Equal(key, rocmKVManifestKeyTokenStart):
		n, next, err := jsonenc.ParseJSONInt(data, i)
		ref.TokenStart = int(n)
		return next, err
	case bytes.Equal(key, rocmKVManifestKeyTokenCount):
		n, next, err := jsonenc.ParseJSONInt(data, i)
		ref.TokenCount = int(n)
		return next, err
	case bytes.Equal(key, rocmKVManifestKeyKeyWidth):
		n, next, err := jsonenc.ParseJSONInt(data, i)
		ref.KeyWidth = int(n)
		return next, err
	case bytes.Equal(key, rocmKVManifestKeyValueWidth):
		n, next, err := jsonenc.ParseJSONInt(data, i)
		ref.ValueWidth = int(n)
		return next, err
	case bytes.Equal(key, rocmKVManifestKeySizeBytes):
		n, next, err := jsonenc.ParseJSONInt(data, i)
		ref.SizeBytes = uint64(n)
		return next, err
	case bytes.Equal(key, rocmKVManifestKeyEncoding):
		s, next, err := parseROCmKVWakeKnownString(data, i, rocmKVWakeKnownBlockEncodings)
		ref.Encoding = s
		return next, err
	default:
		return jsonenc.SkipJSONValue(data, i)
	}
}

func parseROCmKVWakeKnownString(data []byte, i int, known []rocmKVWakeKnownString) (string, int, error) {
	raw, next, err := jsonenc.ParseJSONStringRaw(data, i)
	if err != nil {
		return "", next, err
	}
	for _, value := range known {
		if bytes.Equal(raw, value.raw) {
			return value.value, next, nil
		}
	}
	return string(raw), next, nil
}

func parseROCmKVBlockBundleWakeStateRef(data []byte, i int) (state.ChunkRef, int, error) {
	var ref state.ChunkRef
	i, err := jsonenc.MatchObjectStart(data, i)
	if err != nil {
		return ref, i, err
	}
	i = jsonenc.SkipJSONWhitespace(data, i)
	if i < len(data) && data[i] == '}' {
		return ref, i + 1, nil
	}
	for {
		i = jsonenc.SkipJSONWhitespace(data, i)
		if i >= len(data) || data[i] != '"' {
			return ref, i, jsonenc.ErrInvalidJSON
		}
		key, next, err := jsonenc.ParseJSONStringRaw(data, i)
		if err != nil {
			return ref, next, err
		}
		i = jsonenc.SkipJSONWhitespace(data, next)
		if i >= len(data) || data[i] != ':' {
			return ref, i, jsonenc.ErrInvalidJSON
		}
		i = jsonenc.SkipJSONWhitespace(data, i+1)
		i, err = unmarshalROCmKVBlockBundleWakeStateField(data, i, key, &ref)
		if err != nil {
			return ref, i, err
		}
		i = jsonenc.SkipJSONWhitespace(data, i)
		if i >= len(data) {
			return ref, i, jsonenc.ErrInvalidJSON
		}
		if data[i] == ',' {
			i++
			continue
		}
		if data[i] == '}' {
			return ref, i + 1, nil
		}
		return ref, i, jsonenc.ErrInvalidJSON
	}
}

func unmarshalROCmKVBlockBundleWakeStateField(data []byte, i int, key []byte, ref *state.ChunkRef) (int, error) {
	switch {
	case bytes.Equal(key, rocmKVManifestKeyChunkID):
		n, next, err := jsonenc.ParseJSONInt(data, i)
		ref.ChunkID = int(n)
		return next, err
	case bytes.Equal(key, rocmKVManifestKeyFrameOffset):
		n, next, err := jsonenc.ParseJSONInt(data, i)
		ref.FrameOffset = uint64(n)
		return next, err
	case bytes.Equal(key, rocmKVManifestKeyHasFrameOffset):
		v, next, err := jsonenc.ParseJSONBool(data, i)
		ref.HasFrameOffset = v
		return next, err
	case bytes.Equal(key, rocmKVManifestKeyCodec):
		s, next, err := parseROCmKVWakeKnownString(data, i, rocmKVWakeKnownStateCodecs)
		ref.Codec = s
		return next, err
	case bytes.Equal(key, rocmKVManifestKeySegment):
		s, next, err := jsonenc.ParseJSONString(data, i)
		ref.Segment = s
		return next, err
	default:
		return jsonenc.SkipJSONValue(data, i)
	}
}
