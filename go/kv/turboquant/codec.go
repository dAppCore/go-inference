// SPDX-Licence-Identifier: EUPL-1.2

package turboquant

import core "dappco.re/go"

// Codec is the uniform interface MeasureCodecs drives across every KV
// quantisation scheme under test — TurboQuant's two variants, the mixed-bit
// split, and the plain group-quant baselines — so the instrument can loop
// over one []Codec rather than special-casing each algorithm.
type Codec interface {
	// Name identifies the codec in the instrument's report table.
	Name() string
	// Encode quantises one row to its wire-format bytes.
	Encode(row []float32) []byte
	// Decode reconstructs a row of dimension d from an encoded payload.
	Decode(data []byte, d int) []float32
	// BytesPerRow reports the encoded payload size for a row of dimension
	// d, every overhead counted (scales, γ, ρ, sign/index bits) — the value
	// len(Encode(row)) would produce for a row of that dimension.
	BytesPerRow(d int) int
}

// QMSECodec adapts EncodeQMSE/DecodeQMSE to Codec at a fixed bit width and
// seed.
type QMSECodec struct {
	Bits int
	Seed uint64
}

func (c QMSECodec) Name() string { return core.Sprintf("TurboQuant-Qmse-b%d", c.Bits) }
func (c QMSECodec) Encode(row []float32) []byte {
	return MarshalQMSE(EncodeQMSE(row, c.Bits, c.Seed))
}
func (c QMSECodec) Decode(data []byte, d int) []float32 {
	return DecodeQMSE(UnmarshalQMSE(data, d, c.Bits), c.Seed)
}
func (c QMSECodec) BytesPerRow(d int) int { return 4 + packedByteLen(d, c.Bits) }

// QProdCodec adapts EncodeQProd/DecodeQProd to Codec at a fixed total bit
// width and seed.
type QProdCodec struct {
	TotalBits int
	Seed      uint64
}

func (c QProdCodec) Name() string { return core.Sprintf("TurboQuant-Qprod-b%d", c.TotalBits) }
func (c QProdCodec) Encode(row []float32) []byte {
	return MarshalQProd(EncodeQProd(row, c.TotalBits, c.Seed))
}
func (c QProdCodec) Decode(data []byte, d int) []float32 {
	return DecodeQProd(UnmarshalQProd(data, d, c.TotalBits), c.Seed)
}
func (c QProdCodec) BytesPerRow(d int) int {
	return 4 + packedByteLen(d, c.TotalBits-1) + 4 + packedByteLen(d, 1)
}

// MixedCodec adapts EncodeMixed/DecodeMixed to Codec at a fixed calibrated
// split and seed.
type MixedCodec struct {
	Split MixedSplitDescriptor
	Seed  uint64
}

// Name reports the codec's TRUE measured effective bit rate — the
// channel-count-weighted average of OutlierBits and BaseBits — rather than a
// rounded label, so the report table never asserts a rate it did not
// measure.
func (c MixedCodec) Name() string {
	k := c.Split.outlierCount()
	if c.Split.D == 0 {
		return "TurboQuant-Mixed"
	}
	eff := (float64(k*c.Split.OutlierBits) + float64((c.Split.D-k)*c.Split.BaseBits)) / float64(c.Split.D)
	return core.Sprintf("TurboQuant-Mixed-%.2fbit", eff)
}
func (c MixedCodec) Encode(row []float32) []byte {
	return MarshalMixed(EncodeMixed(row, c.Split, c.Seed))
}
func (c MixedCodec) Decode(data []byte, d int) []float32 {
	return DecodeMixed(UnmarshalMixed(data, d, c.Split.BaseBits, c.Split.OutlierBits), c.Seed)
}
func (c MixedCodec) BytesPerRow(d int) int {
	k := c.Split.outlierCount()
	return len(c.Split.OutlierMask) + 4 + packedByteLen(k, c.Split.OutlierBits) + 4 + packedByteLen(d-k, c.Split.BaseBits)
}

// GroupQuantInt8Codec adapts the plain symmetric int8 group-quant baseline
// (g=64, mirroring engine/metal's paged KV q8 scheme) to Codec.
type GroupQuantInt8Codec struct{}

func (GroupQuantInt8Codec) Name() string { return "GroupQuant-Int8-g64" }
func (GroupQuantInt8Codec) Encode(row []float32) []byte {
	return MarshalGroupQuantInt8(EncodeGroupQuantInt8(row))
}
func (GroupQuantInt8Codec) Decode(data []byte, d int) []float32 {
	return DecodeGroupQuantInt8(UnmarshalGroupQuantInt8(data, d))
}
func (GroupQuantInt8Codec) BytesPerRow(d int) int {
	numGroups := (d + GroupSize - 1) / GroupSize
	return d + 4*numGroups
}

// GroupQuantInt4Codec adapts the plain symmetric int4 group-quant baseline
// (g=64) to Codec.
type GroupQuantInt4Codec struct{}

func (GroupQuantInt4Codec) Name() string { return "GroupQuant-Int4-g64" }
func (GroupQuantInt4Codec) Encode(row []float32) []byte {
	return MarshalGroupQuantInt4(EncodeGroupQuantInt4(row))
}
func (GroupQuantInt4Codec) Decode(data []byte, d int) []float32 {
	return DecodeGroupQuantInt4(UnmarshalGroupQuantInt4(data, d))
}
func (GroupQuantInt4Codec) BytesPerRow(d int) int {
	numGroups := (d + GroupSize - 1) / GroupSize
	return packedByteLen(d, 4) + 4*numGroups
}

// Codecs returns the full codec table MeasureCodecs drives: TurboQuant
// Q_mse and Q_prod at 2/3/4 bits, the mixed-bit outlier split at
// mixedSplit's calibrated bit widths, and the plain int8/int4 group-quant
// baselines.
//
//	split := CalibrateMixedSplit(calibrationRows, 128, 32, 2, 3)
//	codecs := Codecs(42, split)
func Codecs(seed uint64, mixedSplit MixedSplitDescriptor) []Codec {
	return []Codec{
		QMSECodec{Bits: 2, Seed: seed},
		QMSECodec{Bits: 3, Seed: seed},
		QMSECodec{Bits: 4, Seed: seed},
		QProdCodec{TotalBits: 2, Seed: seed},
		QProdCodec{TotalBits: 3, Seed: seed},
		QProdCodec{TotalBits: 4, Seed: seed},
		MixedCodec{Split: mixedSplit, Seed: seed},
		GroupQuantInt8Codec{},
		GroupQuantInt4Codec{},
	}
}
