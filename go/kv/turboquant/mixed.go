// SPDX-Licence-Identifier: EUPL-1.2

package turboquant

import (
	"math"
	"sort"
)

const (
	mixedSeedOutlier uint64 = 3 // distinct from seedPurposeRotation/seedPurposeQJL
	mixedSeedBase    uint64 = 4
)

// MixedSplitDescriptor is the practical KV mode's per-head channel split: a
// tiny, explicit record of which channels are outliers (quantised at
// OutlierBits) versus the rest (quantised at BaseBits). OutlierMask is a
// ceil(D/8)-byte bitmask — fixed-size and unambiguous, cheaper than a
// variable-length index list at the split sizes this package targets (e.g.
// 32-of-128 outliers: 16 mask bytes vs 32 index bytes).
type MixedSplitDescriptor struct {
	D           int
	OutlierMask []byte
	BaseBits    int
	OutlierBits int
}

// NewMixedSplitDescriptor builds a descriptor marking outlierChannels (each
// in [0,d)) as outliers.
//
//	split := NewMixedSplitDescriptor(128, []int{3, 17, 100}, 2, 3)
func NewMixedSplitDescriptor(d int, outlierChannels []int, baseBits, outlierBits int) MixedSplitDescriptor {
	mask := make([]byte, (d+7)/8)
	for _, ch := range outlierChannels {
		if ch < 0 || ch >= d {
			continue
		}
		mask[ch/8] |= 1 << uint(ch%8)
	}
	return MixedSplitDescriptor{D: d, OutlierMask: mask, BaseBits: baseBits, OutlierBits: outlierBits}
}

// CalibrateMixedSplit selects the k channels with the highest running mean
// |amplitude| over samples (each of dimension d) as outliers — the paper's
// "practical KV mode" calibration: e.g. 32-of-128 channels at 3 bits, the
// remaining 96 at 2 bits, for a 2.5-bit effective rate.
//
//	split := CalibrateMixedSplit(calibrationRows, 128, 32, 2, 3) // 2.5-bit effective
func CalibrateMixedSplit(samples [][]float32, d, k, baseBits, outlierBits int) MixedSplitDescriptor {
	amp := make([]float64, d)
	for _, row := range samples {
		for i := 0; i < d && i < len(row); i++ {
			amp[i] += math.Abs(float64(row[i]))
		}
	}
	idx := make([]int, d)
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool {
		if amp[idx[a]] != amp[idx[b]] {
			return amp[idx[a]] > amp[idx[b]]
		}
		return idx[a] < idx[b] // stable tie-break: deterministic across platforms
	})
	if k > d {
		k = d
	}
	if k < 0 {
		k = 0
	}
	return NewMixedSplitDescriptor(d, idx[:k], baseBits, outlierBits)
}

// isOutlier reports whether channel ch is in the outlier set.
func (s MixedSplitDescriptor) isOutlier(ch int) bool {
	return s.OutlierMask[ch/8]&(1<<uint(ch%8)) != 0
}

// outlierCount returns the number of channels marked as outliers (the
// dimension of the outlier sub-vector Encode/Decode operate on).
//
//	NewMixedSplitDescriptor(8, []int{1, 3}, 2, 3).outlierCount() // 2
func (s MixedSplitDescriptor) outlierCount() int {
	n := 0
	for ch := 0; ch < s.D; ch++ {
		if s.isOutlier(ch) {
			n++
		}
	}
	return n
}

// split partitions x into its outlier and base sub-rows, each preserving
// ascending channel order.
func (s MixedSplitDescriptor) split(x []float32) (outlier, base []float32) {
	for i, v := range x {
		if i >= s.D {
			break
		}
		if s.isOutlier(i) {
			outlier = append(outlier, v)
		} else {
			base = append(base, v)
		}
	}
	return outlier, base
}

// MixedEncoded is one row's mixed-bit payload: the split descriptor plus two
// independent Q_mse sub-encodings, each with its own row norm and rotation —
// the outlier sub-vector at Split.OutlierBits, the rest at Split.BaseBits.
type MixedEncoded struct {
	Split   MixedSplitDescriptor
	Outlier QMSEEncoded
	Base    QMSEEncoded
}

// EncodeMixed quantises row x under split: the outlier channels and the base
// channels are extracted into two sub-rows and each Q_mse-encoded
// independently at its own bit width. seed is derived into two decorrelated
// sub-seeds so the outlier and base rotations are independent of each other.
//
//	split := NewMixedSplitDescriptor(4, []int{0}, 1, 2) // channel 0 at 2 bits, rest at 1
//	e := EncodeMixed([]float32{9, 1, 1, 1}, split, 42)
//	x := DecodeMixed(e, 42)
func EncodeMixed(x []float32, split MixedSplitDescriptor, seed uint64) MixedEncoded {
	outlierRow, baseRow := split.split(x)
	return MixedEncoded{
		Split:   split,
		Outlier: EncodeQMSE(outlierRow, split.OutlierBits, deriveSeed(seed, mixedSeedOutlier)),
		Base:    EncodeQMSE(baseRow, split.BaseBits, deriveSeed(seed, mixedSeedBase)),
	}
}

// DecodeMixed reverses EncodeMixed: decodes the two sub-rows and
// interleaves them back into channel order per e.Split's mask. seed must
// match the seed Encode used.
//
//	split := NewMixedSplitDescriptor(4, []int{0}, 1, 2)
//	e := EncodeMixed([]float32{9, 1, 1, 1}, split, 7)
//	x := DecodeMixed(e, 7)
func DecodeMixed(e MixedEncoded, seed uint64) []float32 {
	outlierVals := DecodeQMSE(e.Outlier, deriveSeed(seed, mixedSeedOutlier))
	baseVals := DecodeQMSE(e.Base, deriveSeed(seed, mixedSeedBase))
	out := make([]float32, e.Split.D)
	oi, bi := 0, 0
	for i := 0; i < e.Split.D; i++ {
		if e.Split.isOutlier(i) {
			out[i] = outlierVals[oi]
			oi++
		} else {
			out[i] = baseVals[bi]
			bi++
		}
	}
	return out
}

// MarshalMixed serialises e to its wire form: the outlier mask verbatim,
// followed by MarshalQMSE(Outlier), followed by MarshalQMSE(Base).
//
//	data := MarshalMixed(EncodeMixed(row, split, 42))
func MarshalMixed(e MixedEncoded) []byte {
	outlierData := MarshalQMSE(e.Outlier)
	baseData := MarshalQMSE(e.Base)
	out := make([]byte, len(e.Split.OutlierMask)+len(outlierData)+len(baseData))
	off := copy(out, e.Split.OutlierMask)
	off += copy(out[off:], outlierData)
	copy(out[off:], baseData)
	return out
}

// UnmarshalMixed reverses MarshalMixed, given the row dimension d and the
// bit widths the encoder used (baseBits, outlierBits — the per-codec-
// instance constants; see MixedCodec).
//
//	e := UnmarshalMixed(data, 128, 2, 3)
func UnmarshalMixed(data []byte, d, baseBits, outlierBits int) MixedEncoded {
	maskLen := (d + 7) / 8
	empty := MixedEncoded{Split: MixedSplitDescriptor{D: d, BaseBits: baseBits, OutlierBits: outlierBits}}
	if len(data) < maskLen {
		return empty
	}
	mask := append([]byte(nil), data[:maskLen]...)
	split := MixedSplitDescriptor{D: d, OutlierMask: mask, BaseBits: baseBits, OutlierBits: outlierBits}
	k := split.outlierCount()
	off := maskLen
	outlierLen := 4 + packedByteLen(k, outlierBits)
	baseLen := 4 + packedByteLen(d-k, baseBits)
	if len(data) < off+outlierLen+baseLen {
		return MixedEncoded{Split: split}
	}
	outlier := UnmarshalQMSE(data[off:off+outlierLen], k, outlierBits)
	off += outlierLen
	base := UnmarshalQMSE(data[off:off+baseLen], d-k, baseBits)
	return MixedEncoded{Split: split, Outlier: outlier, Base: base}
}
