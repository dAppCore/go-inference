// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import core "dappco.re/go"

// Quantize packs float32 values into one GGUF quantised payload in the
// given format — the exported entry to the nine kernels, so engines
// stream through ONE shared implementation instead of carrying
// byte-identical private copies. values length must be a multiple of
// the format's block size (32 for the _0 family, 256 for the K
// family).
//
//	packed, err := gguf.Quantize(gguf.QuantizeQ4_K, block)
func Quantize(format QuantizeFormat, values []float32) ([]byte, error) {
	return AppendQuantize(format, nil, values)
}

// AppendQuantize appends the quantised payload for values onto dst and
// returns the extended slice — the streaming-writer shape: callers
// quantise chunk by chunk into one growing buffer without re-copying.
//
//	for _, block := range chunks {
//		out, err = gguf.AppendQuantize(gguf.QuantizeQ8_0, out, block)
//		if err != nil { return err }
//	}
func AppendQuantize(format QuantizeFormat, dst []byte, values []float32) ([]byte, error) {
	_, blockSize, _, err := ggufQuantizeLayout(format)
	if err != nil {
		return nil, err
	}
	if len(values)%blockSize != 0 {
		return nil, core.NewError(core.Sprintf("gguf: %d values not divisible by %s block size %d", len(values), format, blockSize))
	}
	switch format {
	case QuantizeQ8_0:
		return appendQuantizeQ8_0(dst, values), nil
	case QuantizeQ4_0:
		return appendQuantizeQ4_0(dst, values), nil
	case QuantizeQ5_0:
		return appendQuantizeQ5_0(dst, values), nil
	case QuantizeQ4_K:
		return appendQuantizeQ4_K(dst, values), nil
	case QuantizeQ5_K:
		return appendQuantizeQ5_K(dst, values), nil
	case QuantizeQ6_K:
		return appendQuantizeQ6_K(dst, values), nil
	case QuantizeQ8_K:
		return appendQuantizeQ8_K(dst, values), nil
	case QuantizeQ3_K:
		return appendQuantizeQ3_K(dst, values), nil
	case QuantizeQ2_K:
		return appendQuantizeQ2_K(dst, values), nil
	}
	return nil, core.NewError(core.Sprintf("gguf: no kernel for quantise format %q", format))
}
