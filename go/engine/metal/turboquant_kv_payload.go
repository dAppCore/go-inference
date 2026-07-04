// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"unsafe"

	core "dappco.re/go"
)

const (
	nativeTurboQuantKVLayoutVersion = 1
	nativeTurboQuantKVCodecName     = "turboquant-kv-v1"

	nativeTurboQuantKVAlgorithmMSE  = "turboquantmse"
	nativeTurboQuantKVAlgorithmProd = "turboquantprod"

	nativeTurboQuantKVOutlierPolicyHighHalfHeadDimV1 = "high-half-head-dim-v1"
	nativeTurboQuantKVOutlierPolicyExplicitMaskV1    = "explicit-mask-v1"

	nativeTurboQuantKVNormPolicyExplicitVectorBF16V1         = "explicit-vector-norm-bf16-v1"
	nativeTurboQuantKVResidualNormPolicyExplicitVectorBF16V1 = "explicit-vector-residual-norm-bf16-v1"
	nativeTurboQuantKVReferenceCodebookUniform               = "uniform-fwht"

	nativeTurboQuantKVPayloadAlignment    uint64 = 64
	nativeTurboQuantKVPayloadEndianLittle        = "little"

	nativeTurboQuantKVPayloadKeyCentroids      = "key_centroids"
	nativeTurboQuantKVPayloadKeyQJLSigns       = "key_qjl_signs"
	nativeTurboQuantKVPayloadKeyNorms          = "key_norms_bf16"
	nativeTurboQuantKVPayloadKeyResidualNorms  = "key_residual_norms_bf16"
	nativeTurboQuantKVPayloadValueCentroids    = "value_centroids"
	nativeTurboQuantKVPayloadValueNorms        = "value_norms_bf16"
	nativeTurboQuantKVPayloadOutlierMaskHeader = "outlier_masks"
)

type nativeTurboQuantKVShape struct {
	Batch   int32 `json:"batch"`
	Heads   int32 `json:"heads"`
	SeqLen  int32 `json:"seq_len"`
	HeadDim int32 `json:"head_dim"`
}

func (shape nativeTurboQuantKVShape) valid() bool {
	return shape.Batch > 0 && shape.Heads > 0 && shape.SeqLen > 0 && shape.HeadDim > 0
}

type nativeTurboQuantKVCodec struct {
	Algorithm          string `json:"algorithm"`
	NormalBits         int    `json:"normal_bits"`
	OutlierBits        int    `json:"outlier_bits,omitempty"`
	OutlierPolicy      string `json:"outlier_policy,omitempty"`
	OutlierMask        []byte `json:"outlier_mask,omitempty"`
	NormPolicy         string `json:"norm_policy,omitempty"`
	ResidualNormPolicy string `json:"residual_norm_policy,omitempty"`
	RotationSeed       uint64 `json:"rotation_seed"`
	QJLSeed            uint64 `json:"qjl_seed,omitempty"`
	CodebookID         string `json:"codebook_id"`
}

func (codec nativeTurboQuantKVCodec) validate(kind string, headDim int32) error {
	if codec.Algorithm != nativeTurboQuantKVAlgorithmMSE && codec.Algorithm != nativeTurboQuantKVAlgorithmProd {
		return core.NewError("native: TurboQuant " + kind + " algorithm is invalid")
	}
	if codec.NormalBits <= 0 {
		return core.NewError("native: TurboQuant " + kind + " normal bit width is invalid")
	}
	if codec.NormalBits > 8 {
		return core.NewError("native: TurboQuant " + kind + " normal bit width exceeds byte storage")
	}
	if len(codec.OutlierMask) > 0 && codec.OutlierBits <= 0 {
		return core.NewError("native: TurboQuant " + kind + " outlier bit width is invalid")
	}
	if codec.OutlierBits > 8 {
		return core.NewError("native: TurboQuant " + kind + " outlier bit width exceeds byte storage")
	}
	if len(codec.OutlierMask) > 0 && codec.OutlierPolicy == "" {
		return core.NewError("native: TurboQuant " + kind + " outlier policy is missing")
	}
	if headDim <= 0 {
		return core.NewError("native: TurboQuant " + kind + " head dimension is invalid")
	}
	if len(codec.OutlierMask) > 0 && len(codec.OutlierMask) != nativeTurboQuantKVMaskBytes(headDim) {
		return core.NewError("native: TurboQuant " + kind + " outlier mask length is invalid")
	}
	if codec.OutlierPolicy != "" &&
		codec.OutlierPolicy != nativeTurboQuantKVOutlierPolicyHighHalfHeadDimV1 &&
		codec.OutlierPolicy != nativeTurboQuantKVOutlierPolicyExplicitMaskV1 {
		return core.NewError("native: TurboQuant " + kind + " outlier policy is unsupported")
	}
	if codec.OutlierPolicy == nativeTurboQuantKVOutlierPolicyHighHalfHeadDimV1 {
		want := nativeTurboQuantKVOutlierMask(headDim, codec.outlierChannels(headDim))
		if !nativeTurboQuantKVBytesEqual(codec.OutlierMask, want) {
			return core.NewError("native: TurboQuant " + kind + " outlier mask does not match high-half policy")
		}
	}
	if codec.NormPolicy == "" {
		return core.NewError("native: TurboQuant " + kind + " norm policy is missing")
	}
	if codec.NormPolicy != nativeTurboQuantKVNormPolicyExplicitVectorBF16V1 {
		return core.NewError("native: TurboQuant " + kind + " norm policy is unsupported")
	}
	if codec.Algorithm == nativeTurboQuantKVAlgorithmProd {
		if codec.ResidualNormPolicy == "" {
			return core.NewError("native: TurboQuant " + kind + " residual norm policy is missing")
		}
		if codec.ResidualNormPolicy != nativeTurboQuantKVResidualNormPolicyExplicitVectorBF16V1 {
			return core.NewError("native: TurboQuant " + kind + " residual norm policy is unsupported")
		}
	} else if codec.ResidualNormPolicy != "" {
		return core.NewError("native: TurboQuant " + kind + " residual norm policy is only valid for TurboQuantprod")
	}
	if codec.RotationSeed == 0 {
		return core.NewError("native: TurboQuant " + kind + " rotation seed is missing")
	}
	if codec.Algorithm == nativeTurboQuantKVAlgorithmProd && codec.QJLSeed == 0 {
		return core.NewError("native: TurboQuant " + kind + " QJL seed is missing")
	}
	if codec.CodebookID != nativeTurboQuantKVReferenceCodebookUniform {
		return core.NewError("native: TurboQuant " + kind + " codebook is unsupported")
	}
	return nil
}

func (codec nativeTurboQuantKVCodec) outlierChannels(headDim int32) int32 {
	if headDim <= 0 || len(codec.OutlierMask) == 0 {
		return 0
	}
	var count int32
	for channel := int32(0); channel < headDim; channel++ {
		if codec.OutlierMask[channel/8]&(1<<uint(channel%8)) != 0 {
			count++
		}
	}
	return count
}

func (codec nativeTurboQuantKVCodec) bitsForChannel(channel int32) int {
	if channel < 0 || len(codec.OutlierMask) == 0 {
		return codec.NormalBits
	}
	byteIndex := channel / 8
	bitIndex := uint(channel % 8)
	if int(byteIndex) < len(codec.OutlierMask) && codec.OutlierMask[byteIndex]&(1<<bitIndex) != 0 && codec.OutlierBits > 0 {
		return codec.OutlierBits
	}
	return codec.NormalBits
}

func (codec nativeTurboQuantKVCodec) centroidBitsPerVector(headDim int32) uint64 {
	if headDim <= 0 || codec.NormalBits <= 0 {
		return 0
	}
	outliers := uint64(codec.outlierChannels(headDim))
	normal := uint64(headDim) - outliers
	outlierBits := codec.OutlierBits
	if outlierBits <= 0 {
		outlierBits = codec.NormalBits
	}
	return normal*uint64(codec.NormalBits) + outliers*uint64(outlierBits)
}

type nativeTurboQuantKVPageLayout struct {
	Version     int                     `json:"version"`
	Codec       string                  `json:"codec"`
	CacheIndex  int                     `json:"cache_index"`
	Layer       int                     `json:"layer"`
	LayerType   string                  `json:"layer_type"`
	SharedOwner int                     `json:"shared_owner"`
	Shape       nativeTurboQuantKVShape `json:"shape"`
	TokenOffset int                     `json:"token_offset"`
	PageTokens  int                     `json:"page_tokens"`
	PageSize    int                     `json:"page_size"`
	LocalWindow int                     `json:"local_window,omitempty"`
	Key         nativeTurboQuantKVCodec `json:"key"`
	Value       nativeTurboQuantKVCodec `json:"value"`
}

func (layout nativeTurboQuantKVPageLayout) pageVectorCount() uint64 {
	if !layout.Shape.valid() || layout.PageTokens <= 0 {
		return 0
	}
	return uint64(layout.Shape.Batch) * uint64(layout.Shape.Heads) * uint64(layout.PageTokens)
}

func (layout nativeTurboQuantKVPageLayout) pageElementCount() uint64 {
	vectors := layout.pageVectorCount()
	if vectors == 0 || layout.Shape.HeadDim <= 0 {
		return 0
	}
	return vectors * uint64(layout.Shape.HeadDim)
}

func (layout nativeTurboQuantKVPageLayout) estimatePayloadBytes() (TurboQuantKVPayloadEstimate, error) {
	if err := layout.validate(); err != nil {
		return TurboQuantKVPayloadEstimate{}, err
	}
	vectors := layout.pageVectorCount()
	elements := layout.pageElementCount()
	keyCentroidBytesPerVector := nativeTurboQuantKVPackedBytes(layout.Key.centroidBitsPerVector(layout.Shape.HeadDim))
	keyQJLBytesPerVector := nativeTurboQuantKVPackedBytes(uint64(layout.Shape.HeadDim))
	valueCentroidBytesPerVector := nativeTurboQuantKVPackedBytes(layout.Value.centroidBitsPerVector(layout.Shape.HeadDim))
	estimate := TurboQuantKVPayloadEstimate{
		PageVectors:        vectors,
		PageElements:       elements,
		KeyCentroidBytes:   vectors * keyCentroidBytesPerVector,
		KeyNormBytes:       vectors * bf16Size,
		ValueCentroidBytes: vectors * valueCentroidBytesPerVector,
		ValueNormBytes:     vectors * bf16Size,
		OutlierMaskBytes:   uint64(len(layout.Key.OutlierMask) + len(layout.Value.OutlierMask)),
		FP16BaselineBytes:  elements * 2 * bf16Size,
	}
	if layout.Key.Algorithm == nativeTurboQuantKVAlgorithmProd {
		estimate.KeyQJLSignBytes = vectors * keyQJLBytesPerVector
		estimate.KeyResidualNormBytes = vectors * bf16Size
	}
	estimate.PayloadBytes = estimate.KeyCentroidBytes +
		estimate.KeyQJLSignBytes +
		estimate.KeyNormBytes +
		estimate.KeyResidualNormBytes +
		estimate.ValueCentroidBytes +
		estimate.ValueNormBytes +
		estimate.OutlierMaskBytes
	return estimate, nil
}

func (layout nativeTurboQuantKVPageLayout) validate() error {
	if layout.Version != nativeTurboQuantKVLayoutVersion {
		return core.NewError(core.Sprintf("native: TurboQuant KV layout version %d is unsupported", layout.Version))
	}
	if layout.Codec != nativeTurboQuantKVCodecName {
		return core.NewError("native: TurboQuant KV codec is invalid")
	}
	if layout.CacheIndex < 0 || layout.Layer < 0 || layout.SharedOwner < 0 {
		return core.NewError("native: TurboQuant KV layer identity is invalid")
	}
	if layout.LayerType == "" {
		return core.NewError("native: TurboQuant KV layer type is missing")
	}
	if !layout.Shape.valid() {
		return core.NewError("native: TurboQuant KV shape is invalid")
	}
	if layout.TokenOffset < 0 || layout.PageTokens <= 0 || layout.PageSize <= 0 {
		return core.NewError("native: TurboQuant KV page range is invalid")
	}
	if layout.PageTokens > layout.PageSize || int32(layout.PageTokens) > layout.Shape.SeqLen {
		return core.NewError("native: TurboQuant KV page tokens exceed shape")
	}
	if layout.LocalWindow < 0 {
		return core.NewError("native: TurboQuant KV local window is invalid")
	}
	if layout.Key.Algorithm != nativeTurboQuantKVAlgorithmProd {
		return core.NewError("native: TurboQuant KV keys require TurboQuantprod")
	}
	if err := layout.Key.validate("key", layout.Shape.HeadDim); err != nil {
		return err
	}
	if layout.Value.Algorithm != nativeTurboQuantKVAlgorithmMSE {
		return core.NewError("native: TurboQuant KV values require TurboQuantmse")
	}
	if err := layout.Value.validate("value", layout.Shape.HeadDim); err != nil {
		return err
	}
	return nil
}

type nativeTurboQuantKVPayloadSection struct {
	Name      string `json:"name"`
	Offset    uint64 `json:"offset"`
	Bytes     uint64 `json:"bytes"`
	Alignment uint64 `json:"alignment"`
}

type nativeTurboQuantKVPagePayload struct {
	Layout    nativeTurboQuantKVPageLayout       `json:"layout"`
	Endian    string                             `json:"endian"`
	Alignment uint64                             `json:"alignment"`
	Sections  []nativeTurboQuantKVPayloadSection `json:"sections"`
	Data      []byte                             `json:"data"`
}

// TurboQuantKVPayloadEstimate summarises compressed TurboQuant K/V payload bytes
// retained by native snapshot restore paths.
type TurboQuantKVPayloadEstimate struct {
	Pages                     int
	PageVectors               uint64
	PageElements              uint64
	KeyCentroidBytes          uint64
	KeyQJLSignBytes           uint64
	KeyNormBytes              uint64
	KeyResidualNormBytes      uint64
	ValueCentroidBytes        uint64
	ValueNormBytes            uint64
	OutlierMaskBytes          uint64
	PayloadBytes              uint64
	PaddedPayloadBytes        uint64
	AlignmentPaddingBytes     uint64
	FP16BaselineBytes         uint64
	PayloadToFP16Ratio        float64
	PaddedPayloadToFP16Ratio  float64
	PayloadSavingsRatio       float64
	PaddedPayloadSavingsRatio float64
}

type nativeTurboQuantKVPayloadCacheKey struct {
	ptr         uintptr
	len         int
	fingerprint uint64
}

func nativeTurboQuantKVLayerSlabs(payloadBytes [][]byte, view sessionStateLayerView) ([]byte, []byte, int, error) {
	return nativeTurboQuantKVLayerSlabsLimit(payloadBytes, view, 0)
}

func nativeTurboQuantKVLayerPrefixSlabs(payloadBytes [][]byte, view sessionStateLayerView, prefixTokens int) ([]byte, []byte, int, error) {
	if prefixTokens <= 0 {
		return nil, nil, 0, core.NewError("native.RestoreKV: turboquant prefix length is invalid")
	}
	return nativeTurboQuantKVLayerSlabsLimit(payloadBytes, view, prefixTokens)
}

func nativeTurboQuantKVLayerSlabsLimit(payloadBytes [][]byte, view sessionStateLayerView, prefixTokens int) ([]byte, []byte, int, error) {
	return nativeTurboQuantKVLayerDecodeLimitInto(payloadBytes, view, prefixTokens, false, nil, nil)
}

func nativeTurboQuantKVLayerRows(payloadBytes [][]byte, view sessionStateLayerView) ([]byte, []byte, int, error) {
	return nativeTurboQuantKVLayerDecodeLimit(payloadBytes, view, 0, true)
}

func nativeTurboQuantKVLayerPrefixRows(payloadBytes [][]byte, view sessionStateLayerView, prefixTokens int) ([]byte, []byte, int, error) {
	if prefixTokens <= 0 {
		return nil, nil, 0, core.NewError("native.RestoreKV: turboquant prefix length is invalid")
	}
	return nativeTurboQuantKVLayerDecodeLimit(payloadBytes, view, prefixTokens, true)
}

func nativeTurboQuantKVLayerDecodeLimit(payloadBytes [][]byte, view sessionStateLayerView, prefixTokens int, tokenRows bool) ([]byte, []byte, int, error) {
	return nativeTurboQuantKVLayerDecodeLimitIntoScratch(payloadBytes, view, prefixTokens, tokenRows, nil, nil, nil, nil)
}

func nativeTurboQuantKVLayerRowsInto(payloadBytes [][]byte, view sessionStateLayerView, prefixTokens int, keyRows, valueRows []byte) (int, error) {
	return nativeTurboQuantKVLayerRowsIntoScratch(payloadBytes, view, prefixTokens, keyRows, valueRows, nil, nil)
}

func nativeTurboQuantKVLayerRowsIntoScratch(payloadBytes [][]byte, view sessionStateLayerView, prefixTokens int, keyRows, valueRows []byte, rotatedScratch, normalisedScratch []float64) (int, error) {
	if len(keyRows) == 0 || len(valueRows) == 0 {
		return 0, core.NewError("native.RestoreKV: turboquant destination rows are missing")
	}
	_, _, seqLen, err := nativeTurboQuantKVLayerDecodeLimitIntoScratch(payloadBytes, view, prefixTokens, true, keyRows, valueRows, rotatedScratch, normalisedScratch)
	return seqLen, err
}

func nativeTurboQuantKVLayerPayloadsRowsIntoScratch(payloads []nativeTurboQuantKVPagePayload, view sessionStateLayerView, prefixTokens int, keyRows, valueRows []byte, rotatedScratch, normalisedScratch []float64) (int, error) {
	if len(keyRows) == 0 || len(valueRows) == 0 {
		return 0, core.NewError("native.RestoreKV: turboquant destination rows are missing")
	}
	_, _, seqLen, err := nativeTurboQuantKVLayerDecodePayloadsIntoScratch(payloads, view, prefixTokens, true, keyRows, valueRows, rotatedScratch, normalisedScratch)
	return seqLen, err
}

func nativeTurboQuantKVLayerDecodeLimitInto(payloadBytes [][]byte, view sessionStateLayerView, prefixTokens int, tokenRows bool, keyDst, valueDst []byte) ([]byte, []byte, int, error) {
	return nativeTurboQuantKVLayerDecodeLimitIntoScratch(payloadBytes, view, prefixTokens, tokenRows, keyDst, valueDst, nil, nil)
}

func nativeTurboQuantKVLayerDecodeLimitIntoScratch(payloadBytes [][]byte, view sessionStateLayerView, prefixTokens int, tokenRows bool, keyDst, valueDst []byte, rotatedScratch, normalisedScratch []float64) ([]byte, []byte, int, error) {
	payloads, err := nativeTurboQuantKVParsePayloads(payloadBytes, view, nil)
	if err != nil {
		return nil, nil, 0, err
	}
	return nativeTurboQuantKVLayerDecodePayloadsIntoScratch(payloads, view, prefixTokens, tokenRows, keyDst, valueDst, rotatedScratch, normalisedScratch)
}

func nativeTurboQuantKVLayerDecodePayloadsIntoScratch(payloads []nativeTurboQuantKVPagePayload, view sessionStateLayerView, prefixTokens int, tokenRows bool, keyDst, valueDst []byte, rotatedScratch, normalisedScratch []float64) ([]byte, []byte, int, error) {
	batch, heads, totalTokens, headDim, baseOffset, err := nativeTurboQuantKVPayloadShape(payloads)
	if err != nil {
		return nil, nil, 0, core.E("native.RestoreKV", "turboquant payload shape", err)
	}
	if batch != 1 || heads != view.kvHeads || headDim != view.headDim {
		return nil, nil, 0, core.NewError("native.RestoreKV: turboquant payload shape mismatch")
	}
	decodeTokens := totalTokens
	if prefixTokens > 0 {
		if prefixTokens > totalTokens {
			return nil, nil, 0, core.NewError("native.RestoreKV: turboquant prefix exceeds payload window")
		}
		decodeTokens = prefixTokens
	}
	wantBytes := heads * decodeTokens * headDim * bf16Size
	keySlab, valueSlab := keyDst, valueDst
	if keySlab == nil && valueSlab == nil {
		keySlab = make([]byte, wantBytes)
		valueSlab = make([]byte, wantBytes)
	} else if len(keySlab) != wantBytes || len(valueSlab) != wantBytes {
		return nil, nil, 0, core.NewError("native.RestoreKV: turboquant destination row size mismatch")
	}
	rotated := rotatedScratch
	if len(rotated) < headDim {
		rotated = make([]float64, headDim)
	} else {
		rotated = rotated[:headDim]
	}
	normalised := normalisedScratch
	if len(normalised) < headDim {
		normalised = make([]float64, headDim)
	} else {
		normalised = normalised[:headDim]
	}
	nativeTurboQuantKVSortPayloadsByTokenOffset(payloads)
	tokenStart := 0
	for idx := range payloads {
		payload := payloads[idx]
		if payload.Layout.TokenOffset != baseOffset+tokenStart {
			if payload.Layout.TokenOffset < baseOffset+tokenStart {
				return nil, nil, 0, core.NewError("native.RestoreKV: turboquant payload pages overlap")
			}
			return nil, nil, 0, core.NewError("native.RestoreKV: turboquant payload pages leave a gap")
		}
		if tokenStart < decodeTokens {
			take := payload.Layout.PageTokens
			if tokenStart+take > decodeTokens {
				take = decodeTokens - tokenStart
			}
			if take > 0 {
				var err error
				if tokenRows {
					err = payload.decodeBaseBF16PrefixRowsInto(keySlab, valueSlab, decodeTokens, tokenStart, take, rotated, normalised)
				} else {
					err = payload.decodeBaseBF16PrefixInto(keySlab, valueSlab, decodeTokens, tokenStart, take, rotated, normalised)
				}
				if err != nil {
					return nil, nil, 0, core.E("native.RestoreKV", "decode turboquant payload", err)
				}
			}
		}
		tokenStart += payload.Layout.PageTokens
	}
	if tokenStart != totalTokens {
		return nil, nil, 0, core.NewError("native.RestoreKV: turboquant payload pages leave a gap")
	}
	return keySlab, valueSlab, decodeTokens, nil
}

func nativeTurboQuantKVSortPayloadsByTokenOffset(payloads []nativeTurboQuantKVPagePayload) {
	for i := 1; i < len(payloads); i++ {
		payload := payloads[i]
		j := i - 1
		for ; j >= 0 && payloads[j].Layout.TokenOffset > payload.Layout.TokenOffset; j-- {
			payloads[j+1] = payloads[j]
		}
		payloads[j+1] = payload
	}
}

func nativeTurboQuantKVParsePayloads(payloadBytes [][]byte, view sessionStateLayerView, dst []nativeTurboQuantKVPagePayload) ([]nativeTurboQuantKVPagePayload, error) {
	if cap(dst) < len(payloadBytes) {
		dst = make([]nativeTurboQuantKVPagePayload, 0, len(payloadBytes))
	} else {
		dst = dst[:0]
	}
	for idx, raw := range payloadBytes {
		payload, err := nativeTurboQuantKVParsePayload(raw, idx)
		if err != nil {
			return nil, err
		}
		if payload.Layout.CacheIndex != view.cacheIndex || payload.Layout.Layer != view.layer {
			return nil, core.NewError("native.RestoreKV: turboquant layer identity mismatch")
		}
		dst = append(dst, payload)
	}
	return dst, nil
}

func nativeTurboQuantKVParsePayload(raw []byte, idx int) (nativeTurboQuantKVPagePayload, error) {
	if len(raw) == 0 {
		return nativeTurboQuantKVPagePayload{}, core.NewError("native.RestoreKV: empty turboquant KV payload")
	}
	var payload nativeTurboQuantKVPagePayload
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nativeTurboQuantKVPagePayload{}, core.E("native.RestoreKV", core.Sprintf("decode turboquant payload %d", idx), err)
	}
	if err := payload.Layout.validate(); err != nil {
		return nativeTurboQuantKVPagePayload{}, core.E("native.RestoreKV", "validate turboquant payload", err)
	}
	return payload, nil
}

func nativeTurboQuantKVPayloadsEstimate(payloads []nativeTurboQuantKVPagePayload) (TurboQuantKVPayloadEstimate, error) {
	if len(payloads) == 0 {
		return TurboQuantKVPayloadEstimate{}, core.NewError("native: TurboQuant KV cache has no payloads")
	}
	var estimate TurboQuantKVPayloadEstimate
	for _, payload := range payloads {
		if err := estimate.addTurboQuantKVPayload(payload); err != nil {
			return TurboQuantKVPayloadEstimate{}, err
		}
	}
	estimate.finishTurboQuantKVPayloadRatios()
	return estimate, nil
}

func (estimate *TurboQuantKVPayloadEstimate) addTurboQuantKVPayload(payload nativeTurboQuantKVPagePayload) error {
	if err := payload.validateSections(); err != nil {
		return err
	}
	pageEstimate, err := payload.Layout.estimatePayloadBytes()
	if err != nil {
		return err
	}
	payloadBytes := payload.unpaddedByteCount()
	if payloadBytes != pageEstimate.PayloadBytes {
		return core.NewError(core.Sprintf("native: TurboQuant KV payload byte accounting mismatch: payload=%d estimate=%d", payloadBytes, pageEstimate.PayloadBytes))
	}
	paddedBytes := uint64(len(payload.Data))
	if paddedBytes < payloadBytes {
		return core.NewError("native: TurboQuant KV payload padding is invalid")
	}
	estimate.Pages++
	estimate.PageVectors += pageEstimate.PageVectors
	estimate.PageElements += pageEstimate.PageElements
	estimate.KeyCentroidBytes += pageEstimate.KeyCentroidBytes
	estimate.KeyQJLSignBytes += pageEstimate.KeyQJLSignBytes
	estimate.KeyNormBytes += pageEstimate.KeyNormBytes
	estimate.KeyResidualNormBytes += pageEstimate.KeyResidualNormBytes
	estimate.ValueCentroidBytes += pageEstimate.ValueCentroidBytes
	estimate.ValueNormBytes += pageEstimate.ValueNormBytes
	estimate.OutlierMaskBytes += pageEstimate.OutlierMaskBytes
	estimate.PayloadBytes += payloadBytes
	estimate.PaddedPayloadBytes += paddedBytes
	estimate.AlignmentPaddingBytes += paddedBytes - payloadBytes
	estimate.FP16BaselineBytes += pageEstimate.FP16BaselineBytes
	return nil
}

func (estimate *TurboQuantKVPayloadEstimate) finishTurboQuantKVPayloadRatios() {
	if estimate.FP16BaselineBytes == 0 {
		return
	}
	baseline := float64(estimate.FP16BaselineBytes)
	estimate.PayloadToFP16Ratio = float64(estimate.PayloadBytes) / baseline
	estimate.PaddedPayloadToFP16Ratio = float64(estimate.PaddedPayloadBytes) / baseline
	estimate.PayloadSavingsRatio = 1 - estimate.PayloadToFP16Ratio
	estimate.PaddedPayloadSavingsRatio = 1 - estimate.PaddedPayloadToFP16Ratio
}

func (payload nativeTurboQuantKVPagePayload) unpaddedByteCount() uint64 {
	var total uint64
	for _, section := range payload.Sections {
		total += section.Bytes
	}
	return total
}

// TurboQuantKVPayloadEstimate reports compressed TurboQuant K/V payload bytes
// retained from the latest native restore path. It returns nil when the session
// has not restored any TurboQuant payloads.
func (s *ArchSession) TurboQuantKVPayloadEstimate() (*TurboQuantKVPayloadEstimate, error) {
	if s == nil {
		return nil, nil
	}
	if len(s.turboQuantPayloads) > 0 {
		estimate, err := nativeTurboQuantKVPayloadsEstimate(s.turboQuantPayloads)
		if err != nil {
			return nil, err
		}
		return &estimate, nil
	}
	if len(s.turboQuantCache) == 0 {
		return nil, nil
	}
	var estimate TurboQuantKVPayloadEstimate
	for _, payload := range s.turboQuantCache {
		if err := estimate.addTurboQuantKVPayload(payload); err != nil {
			return nil, err
		}
	}
	estimate.finishTurboQuantKVPayloadRatios()
	return &estimate, nil
}

func (s *ArchSession) turboQuantKVPayloads(payloadBytes [][]byte, view sessionStateLayerView) ([]nativeTurboQuantKVPagePayload, error) {
	if s == nil {
		return nativeTurboQuantKVParsePayloads(payloadBytes, view, nil)
	}
	if cap(s.turboQuantPayloads) < len(payloadBytes) {
		s.turboQuantPayloads = make([]nativeTurboQuantKVPagePayload, 0, len(payloadBytes))
	} else {
		s.turboQuantPayloads = s.turboQuantPayloads[:0]
	}
	if s.turboQuantCache == nil {
		s.turboQuantCache = make(map[nativeTurboQuantKVPayloadCacheKey]nativeTurboQuantKVPagePayload, len(payloadBytes))
	}
	for idx, raw := range payloadBytes {
		key, ok := nativeTurboQuantKVPayloadCacheKeyFor(raw)
		if !ok {
			return nil, core.NewError("native.RestoreKV: empty turboquant KV payload")
		}
		payload, cached := s.turboQuantCache[key]
		if !cached {
			var err error
			payload, err = nativeTurboQuantKVParsePayload(raw, idx)
			if err != nil {
				return nil, err
			}
			s.turboQuantCache[key] = payload
		}
		if payload.Layout.CacheIndex != view.cacheIndex || payload.Layout.Layer != view.layer {
			return nil, core.NewError("native.RestoreKV: turboquant layer identity mismatch")
		}
		s.turboQuantPayloads = append(s.turboQuantPayloads, payload)
	}
	return s.turboQuantPayloads, nil
}

func nativeTurboQuantKVPayloadCacheKeyFor(raw []byte) (nativeTurboQuantKVPayloadCacheKey, bool) {
	if len(raw) == 0 {
		return nativeTurboQuantKVPayloadCacheKey{}, false
	}
	return nativeTurboQuantKVPayloadCacheKey{
		ptr:         uintptr(unsafe.Pointer(unsafe.SliceData(raw))),
		len:         len(raw),
		fingerprint: nativeTurboQuantKVPayloadFingerprint(raw),
	}, true
}

func nativeTurboQuantKVPayloadFingerprint(raw []byte) uint64 {
	hash := uint64(1469598103934665603)
	for _, b := range raw {
		hash ^= uint64(b)
		hash *= 1099511628211
	}
	return hash
}

func nativeTurboQuantKVPayloadShape(payloads []nativeTurboQuantKVPagePayload) (int, int, int, int, int, error) {
	if len(payloads) == 0 {
		return 0, 0, 0, 0, 0, core.NewError("native: TurboQuant KV cache has no payloads")
	}
	first := payloads[0].Layout
	if err := first.validate(); err != nil {
		return 0, 0, 0, 0, 0, err
	}
	batch := int(first.Shape.Batch)
	heads := int(first.Shape.Heads)
	headDim := int(first.Shape.HeadDim)
	baseOffset := first.TokenOffset
	endOffset := first.TokenOffset + first.PageTokens
	for idx := range payloads {
		layout := payloads[idx].Layout
		if err := layout.validate(); err != nil {
			return 0, 0, 0, 0, 0, err
		}
		if layout.Shape.Batch != first.Shape.Batch ||
			layout.Shape.Heads != first.Shape.Heads ||
			layout.Shape.HeadDim != first.Shape.HeadDim {
			return 0, 0, 0, 0, 0, core.NewError("native: TurboQuant KV payload shapes differ")
		}
		if layout.TokenOffset < baseOffset {
			baseOffset = layout.TokenOffset
		}
		if end := layout.TokenOffset + layout.PageTokens; end > endOffset {
			endOffset = end
		}
	}
	totalTokens := endOffset - baseOffset
	if totalTokens <= 0 {
		return 0, 0, 0, 0, 0, core.NewError("native: TurboQuant KV payload token length is invalid")
	}
	return batch, heads, totalTokens, headDim, baseOffset, nil
}

func (payload nativeTurboQuantKVPagePayload) decodeBaseBF16Into(keys, values []byte, totalSeqLen, tokenStart int, rotated, normalised []float64) error {
	return payload.decodeBaseBF16PrefixInto(keys, values, totalSeqLen, tokenStart, payload.Layout.PageTokens, rotated, normalised)
}

func (payload nativeTurboQuantKVPagePayload) decodeBaseBF16PrefixInto(keys, values []byte, totalSeqLen, tokenStart, tokenCount int, rotated, normalised []float64) error {
	return payload.decodeBaseBF16PrefixIntoLayout(keys, values, totalSeqLen, tokenStart, tokenCount, rotated, normalised, false)
}

func (payload nativeTurboQuantKVPagePayload) decodeBaseBF16PrefixRowsInto(keys, values []byte, totalSeqLen, tokenStart, tokenCount int, rotated, normalised []float64) error {
	return payload.decodeBaseBF16PrefixIntoLayout(keys, values, totalSeqLen, tokenStart, tokenCount, rotated, normalised, true)
}

func (payload nativeTurboQuantKVPagePayload) decodeBaseBF16PrefixIntoLayout(keys, values []byte, totalSeqLen, tokenStart, tokenCount int, rotated, normalised []float64, tokenRows bool) error {
	if payload.Endian != nativeTurboQuantKVPayloadEndianLittle {
		return core.NewError("native: TurboQuant reference payload endian marker is invalid")
	}
	if payload.Alignment != nativeTurboQuantKVPayloadAlignment {
		return core.NewError("native: TurboQuant reference payload alignment is invalid")
	}
	if err := payload.Layout.validate(); err != nil {
		return err
	}
	if err := payload.validateSections(); err != nil {
		return err
	}
	layout := payload.Layout
	pageVectors := int(layout.pageVectorCount())
	headDim := int(layout.Shape.HeadDim)
	pageTokens := layout.PageTokens
	if totalSeqLen <= 0 || tokenStart < 0 || pageTokens <= 0 || tokenCount <= 0 || tokenCount > pageTokens || tokenStart+tokenCount > totalSeqLen {
		return core.NewError("native: TurboQuant reference payload destination sequence range is invalid")
	}
	wantBytes := int(layout.Shape.Batch) * int(layout.Shape.Heads) * totalSeqLen * headDim * bf16Size
	if len(keys) < wantBytes || len(values) < wantBytes {
		return core.NewError("native: TurboQuant reference payload destination shape is invalid")
	}
	if len(rotated) < headDim || len(normalised) < headDim {
		return core.NewError("native: TurboQuant reference payload decode scratch is invalid")
	}
	rotated = rotated[:headDim]
	normalised = normalised[:headDim]
	keyCentroids, err := payload.requiredSection(nativeTurboQuantKVPayloadKeyCentroids)
	if err != nil {
		return err
	}
	keyQJLSigns, err := payload.requiredSection(nativeTurboQuantKVPayloadKeyQJLSigns)
	if err != nil {
		return err
	}
	keyNorms, err := payload.requiredSection(nativeTurboQuantKVPayloadKeyNorms)
	if err != nil {
		return err
	}
	keyResidualNorms, err := payload.requiredSection(nativeTurboQuantKVPayloadKeyResidualNorms)
	if err != nil {
		return err
	}
	valueCentroids, err := payload.requiredSection(nativeTurboQuantKVPayloadValueCentroids)
	if err != nil {
		return err
	}
	valueNorms, err := payload.requiredSection(nativeTurboQuantKVPayloadValueNorms)
	if err != nil {
		return err
	}

	keyMSECodec := layout.Key
	keyMSECodec.Algorithm = nativeTurboQuantKVAlgorithmMSE
	keyMSECodec.QJLSeed = 0
	keyMSECodec.ResidualNormPolicy = ""
	keyCentroidBytes := int(nativeTurboQuantKVPackedBytes(keyMSECodec.centroidBitsPerVector(layout.Shape.HeadDim)))
	keyQJLBytes := int(nativeTurboQuantKVPackedBytes(uint64(headDim)))
	valueCentroidBytes := int(nativeTurboQuantKVPackedBytes(layout.Value.centroidBitsPerVector(layout.Shape.HeadDim)))
	if err := nativeTurboQuantKVCheckPayloadLength(nativeTurboQuantKVPayloadKeyCentroids, len(keyCentroids), pageVectors*keyCentroidBytes); err != nil {
		return err
	}
	if err := nativeTurboQuantKVCheckPayloadLength(nativeTurboQuantKVPayloadKeyQJLSigns, len(keyQJLSigns), pageVectors*keyQJLBytes); err != nil {
		return err
	}
	if err := nativeTurboQuantKVCheckPayloadLength(nativeTurboQuantKVPayloadKeyNorms, len(keyNorms), pageVectors*bf16Size); err != nil {
		return err
	}
	if err := nativeTurboQuantKVCheckPayloadLength(nativeTurboQuantKVPayloadKeyResidualNorms, len(keyResidualNorms), pageVectors*bf16Size); err != nil {
		return err
	}
	if err := nativeTurboQuantKVCheckPayloadLength(nativeTurboQuantKVPayloadValueCentroids, len(valueCentroids), pageVectors*valueCentroidBytes); err != nil {
		return err
	}
	if err := nativeTurboQuantKVCheckPayloadLength(nativeTurboQuantKVPayloadValueNorms, len(valueNorms), pageVectors*bf16Size); err != nil {
		return err
	}

	vectorCount := int(layout.Shape.Batch) * int(layout.Shape.Heads)
	heads := int(layout.Shape.Heads)
	for vector := 0; vector < vectorCount; vector++ {
		batch := vector / heads
		head := vector - batch*heads
		for token := 0; token < tokenCount; token++ {
			idx := vector*pageTokens + token
			start := (vector*totalSeqLen + tokenStart + token) * headDim * bf16Size
			if tokenRows {
				start = ((batch*totalSeqLen+tokenStart+token)*heads + head) * headDim * bf16Size
			}
			end := start + headDim*bf16Size
			keyNorm := nativeTurboQuantKVReadBF16Norm(keyNorms[idx*bf16Size:])
			if err := nativeTurboQuantKVDecodePackedMSEBF16(
				keys[start:end],
				keyCentroids[idx*keyCentroidBytes:(idx+1)*keyCentroidBytes],
				keyMSECodec,
				keyNorm,
				rotated,
				normalised,
			); err != nil {
				return core.E("native: TurboQuant reference payload", "decode key", err)
			}
			keyResidualNorm := nativeTurboQuantKVReadBF16Norm(keyResidualNorms[idx*bf16Size:])
			if err := nativeTurboQuantKVApplyProdResidualBF16(
				keys[start:end],
				keyQJLSigns[idx*keyQJLBytes:(idx+1)*keyQJLBytes],
				keyNorm,
				keyResidualNorm,
				layout.Key.QJLSeed,
				rotated,
				normalised,
			); err != nil {
				return core.E("native: TurboQuant reference payload", "decode key QJL residual", err)
			}
			if err := nativeTurboQuantKVDecodePackedMSEBF16(
				values[start:end],
				valueCentroids[idx*valueCentroidBytes:(idx+1)*valueCentroidBytes],
				layout.Value,
				nativeTurboQuantKVReadBF16Norm(valueNorms[idx*bf16Size:]),
				rotated,
				normalised,
			); err != nil {
				return core.E("native: TurboQuant reference payload", "decode value", err)
			}
		}
	}
	return nil
}

func nativeTurboQuantKVApplyProdResidualBF16(dst []byte, packedQJLSigns []byte, keyNorm, residualNorm float32, qjlSeed uint64, rotated, normalised []float64) error {
	headDim := len(dst) / bf16Size
	if len(dst) != headDim*bf16Size || len(rotated) < headDim || len(normalised) < headDim {
		return core.NewError("native: TurboQuantprod residual destination shape is invalid")
	}
	if keyNorm == 0 || residualNorm == 0 {
		return nil
	}
	if len(packedQJLSigns)*8 < headDim {
		return core.NewError("native: TurboQuantprod residual QJL bits are invalid")
	}
	rotated = rotated[:headDim]
	normalised = normalised[:headDim]
	for idx := range headDim {
		sign := 1.0
		if packedQJLSigns[idx/8]&(1<<uint(idx%8)) != 0 {
			sign = -1
		}
		rotated[idx] = sign
	}
	nativeTurboQuantKVRotate(normalised, rotated, qjlSeed, true)
	scale := float64(keyNorm) * float64(residualNorm) / math.Sqrt(float64(headDim))
	for idx := range headDim {
		current := bf16ToF32(dst[idx*bf16Size], dst[idx*bf16Size+1])
		h := f32ToBF16(current + float32(scale*normalised[idx]))
		dst[idx*bf16Size], dst[idx*bf16Size+1] = byte(h), byte(h>>8)
	}
	return nil
}

func nativeTurboQuantKVDecodePackedMSEBF16(dst []byte, packed []byte, codec nativeTurboQuantKVCodec, norm float32, rotated, normalised []float64) error {
	headDim := len(dst) / bf16Size
	if len(dst) != headDim*bf16Size || len(rotated) < headDim || len(normalised) < headDim {
		return core.NewError("native: TurboQuant packed MSE destination shape is invalid")
	}
	if norm == 0 {
		clear(dst)
		return nil
	}
	bitOffset := 0
	for idx := range headDim {
		bits := codec.bitsForChannel(int32(idx))
		if bits <= 0 || len(packed)*8 < bitOffset+bits {
			return core.NewError("native: TurboQuant packed MSE centroid bits are invalid")
		}
		var code byte
		for bit := range bits {
			if packed[bitOffset/8]&(1<<uint(bitOffset%8)) != 0 {
				code |= 1 << uint(bit)
			}
			bitOffset++
		}
		rotated[idx] = nativeTurboQuantKVDequantizeUniform(code, bits)
	}
	nativeTurboQuantKVRotate(normalised[:headDim], rotated[:headDim], codec.RotationSeed, true)
	for idx := range headDim {
		h := f32ToBF16(float32(normalised[idx] * float64(norm)))
		dst[idx*bf16Size], dst[idx*bf16Size+1] = byte(h), byte(h>>8)
	}
	return nil
}

func (payload nativeTurboQuantKVPagePayload) sectionBytes(name string) ([]byte, bool) {
	for _, section := range payload.Sections {
		if section.Name != name {
			continue
		}
		end := section.Offset + section.Bytes
		if section.Offset > uint64(len(payload.Data)) || end > uint64(len(payload.Data)) {
			return nil, false
		}
		return payload.Data[section.Offset:end], true
	}
	return nil, false
}

func (payload nativeTurboQuantKVPagePayload) requiredSection(name string) ([]byte, error) {
	data, ok := payload.sectionBytes(name)
	if !ok {
		return nil, core.NewError("native: TurboQuant reference payload missing " + name)
	}
	return data, nil
}

func (payload nativeTurboQuantKVPagePayload) validateSections() error {
	for _, section := range payload.Sections {
		if section.Alignment != nativeTurboQuantKVPayloadAlignment || section.Offset%nativeTurboQuantKVPayloadAlignment != 0 {
			return core.NewError("native: TurboQuant reference payload section alignment is invalid")
		}
		end := section.Offset + section.Bytes
		if section.Offset > uint64(len(payload.Data)) || end > uint64(len(payload.Data)) {
			return core.NewError("native: TurboQuant reference payload section range is invalid")
		}
	}
	return nil
}

func nativeTurboQuantKVPackedBytes(bits uint64) uint64 {
	if bits == 0 {
		return 0
	}
	return (bits + 7) / 8
}

func nativeTurboQuantKVMaskBytes(headDim int32) int {
	if headDim <= 0 {
		return 0
	}
	return int((headDim + 7) / 8)
}

func nativeTurboQuantKVOutlierMask(headDim, outlierChannels int32) []byte {
	if headDim <= 0 || outlierChannels <= 0 {
		return nil
	}
	if outlierChannels > headDim {
		outlierChannels = headDim
	}
	mask := make([]byte, nativeTurboQuantKVMaskBytes(headDim))
	start := headDim - outlierChannels
	for channel := start; channel < headDim; channel++ {
		mask[channel/8] |= 1 << uint(channel%8)
	}
	return mask
}

func nativeTurboQuantKVBytesEqual(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for idx := range a {
		if a[idx] != b[idx] {
			return false
		}
	}
	return true
}

func nativeTurboQuantKVReadBF16Norm(raw []byte) float32 {
	if len(raw) < bf16Size {
		return 0
	}
	return math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[:bf16Size])) << 16)
}

func nativeTurboQuantKVCheckPayloadLength(name string, got, want int) error {
	if got != want {
		label := core.Replace(name, "_", " ")
		return core.NewError(core.Sprintf("native: TurboQuant reference payload %s bytes = %d, want %d", label, got, want))
	}
	return nil
}

func nativeTurboQuantKVRotate(dst, src []float64, seed uint64, inverse bool) {
	if inverse {
		copy(dst, src)
		nativeTurboQuantKVFWHT(dst)
		nativeTurboQuantKVSignFlip(dst, seed)
		return
	}
	for idx, value := range src {
		if nativeTurboQuantKVSign(seed, idx) < 0 {
			dst[idx] = -value
			continue
		}
		dst[idx] = value
	}
	nativeTurboQuantKVFWHT(dst)
}

func nativeTurboQuantKVFWHT(values []float64) {
	n := len(values)
	for step := 1; step < n; step <<= 1 {
		for start := 0; start < n; start += step << 1 {
			for idx := 0; idx < step; idx++ {
				left := values[start+idx]
				right := values[start+idx+step]
				values[start+idx] = left + right
				values[start+idx+step] = left - right
			}
		}
	}
	scale := 1 / math.Sqrt(float64(n))
	for idx := range values {
		values[idx] *= scale
	}
}

func nativeTurboQuantKVSignFlip(values []float64, seed uint64) {
	for idx := range values {
		if nativeTurboQuantKVSign(seed, idx) < 0 {
			values[idx] = -values[idx]
		}
	}
}

func nativeTurboQuantKVSign(seed uint64, idx int) int {
	mixed := seed + uint64(idx)*0x9e3779b97f4a7c15
	mixed ^= mixed >> 30
	mixed *= 0xbf58476d1ce4e5b9
	mixed ^= mixed >> 27
	mixed *= 0x94d049bb133111eb
	mixed ^= mixed >> 31
	if mixed&1 == 0 {
		return 1
	}
	return -1
}

func nativeTurboQuantKVDequantizeUniform(code byte, bits int) float64 {
	levels := (1 << bits) - 1
	if levels <= 0 {
		return 0
	}
	if int(code) > levels {
		code = byte(levels)
	}
	return (float64(code)*2)/float64(levels) - 1
}
