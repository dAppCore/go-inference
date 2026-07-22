// SPDX-Licence-Identifier: EUPL-1.2

package turboquant

import core "dappco.re/go"

// real.go is the RFC #41 slice S3 real-KV instrument: it runs TurboQuant's
// Q_mse codec over ACTUAL captured K/V rows — engine/metal's
// (*ArchSession).DumpKVRows, post-RoPE, exactly what the live cache holds —
// rather than MeasureCodecs' synthetic Gaussian/sphere-uniform draws. It
// reuses MeasureCodecs' own per-row measurement (measureOne) for every
// reported bit width, so a real-kv row is driven through byte-identical
// codec + attention-error machinery to the synthetic sweep; only the data
// source differs. This package stays host-only/engine-neutral (doc.go) —
// callers capture rows with the metal-side tap and pass the resulting
// [][]float32 in.

// RealMixedKeyBits and RealMixedValueBits are the live cache's asymmetric
// K4/V3 default under test: keys quantised one bit richer than values, on
// the working hypothesis that a key error perturbs every downstream softmax
// weight while a value error only perturbs its own weighted contribution.
// RealMixedNominalBits is their plain average — the "3.5-bit" figure the
// live default is named after.
const (
	RealMixedKeyBits     = 4
	RealMixedValueBits   = 3
	RealMixedNominalBits = float64(RealMixedKeyBits+RealMixedValueBits) / 2
)

// RealMeasurement is MeasureReal's report: one CodecReport per (bit width,
// K|V side) — via the SAME measureOne machinery MeasureCodecs drives — plus
// Mixed, the live RealMixedKeyBits/RealMixedValueBits configuration's
// relative MSE pooled across both sides.
type RealMeasurement struct {
	D       int
	Reports []CodecReport
	Mixed   MixedRealReport
}

// MixedRealReport is the pooled RealMixedKeyBits/RealMixedValueBits relative
// MSE: keys quantised at RealMixedKeyBits, values at RealMixedValueBits, one
// row-MSE pooled across every row of both sides (ΣΣ‖x-x̃‖² / ΣΣ‖x‖² over the
// K rows then the V rows) — the whole cache's blended distortion at its live
// byte budget, not a per-side figure.
type MixedRealReport struct {
	RowMSE         float64
	RowMSERelative float64
	Samples        int
}

// MeasureReal measures TurboQuant's Q_mse codec's REAL-KV distortion at
// every width in bits (default {2, 3, 4} when bits is empty) plus the live
// RealMixedKeyBits/RealMixedValueBits configuration, separately for the
// K-side and V-side rows engine/metal's DumpKVRows captured — one row per
// (attention head, cached token), post-RoPE. seed selects the codec's
// rotation, matching MeasureCodecs' threading; the same seed reproduces the
// same measurement over the same rows.
//
//	keys, values, err := session.DumpKVRows(4)
//	if err != nil { ... }
//	result, err := turboquant.MeasureReal(keys, values, 42)
//	if err != nil { ... }
//	core.Println(turboquant.FormatRealReport(result))
func MeasureReal(keys, values [][]float32, seed uint64, bits ...int) (RealMeasurement, error) {
	if len(keys) == 0 {
		return RealMeasurement{}, core.NewError("turboquant.MeasureReal: no key rows")
	}
	if len(values) == 0 {
		return RealMeasurement{}, core.NewError("turboquant.MeasureReal: no value rows")
	}
	d := len(keys[0])
	if d == 0 {
		return RealMeasurement{}, core.NewError("turboquant.MeasureReal: zero-dimension rows")
	}
	if err := validateRealRows(keys, d); err != nil {
		return RealMeasurement{}, core.E("turboquant.MeasureReal", "key rows", err)
	}
	if err := validateRealRows(values, d); err != nil {
		return RealMeasurement{}, core.E("turboquant.MeasureReal", "value rows", err)
	}
	if len(bits) == 0 {
		bits = []int{2, 3, 4}
	}

	result := RealMeasurement{D: d}
	for _, b := range bits {
		codec := QMSECodec{Bits: b, Seed: seed}
		result.Reports = append(result.Reports,
			measureOne(codec, dataSourceEntry{name: "real-kv-K", rows: keys}, d, seed),
			measureOne(codec, dataSourceEntry{name: "real-kv-V", rows: values}, d, seed),
		)
	}

	keyCodec := QMSECodec{Bits: RealMixedKeyBits, Seed: seed}
	valueCodec := QMSECodec{Bits: RealMixedValueBits, Seed: seed}
	result.Reports = append(result.Reports,
		measureOne(keyCodec, dataSourceEntry{name: "real-kv-K-mixed", rows: keys}, d, seed),
		measureOne(valueCodec, dataSourceEntry{name: "real-kv-V-mixed", rows: values}, d, seed),
	)
	rowMSE, rowMSERelative, samples := pooledRelativeMSE(keyCodec, keys, valueCodec, values)
	result.Mixed = MixedRealReport{RowMSE: rowMSE, RowMSERelative: rowMSERelative, Samples: samples}

	return result, nil
}

// validateRealRows checks every row in rows has exactly d elements — a
// captured K/V set must be rectangular (one dimension per attention head),
// unlike the synthetic generators which build it that way by construction.
func validateRealRows(rows [][]float32, d int) error {
	for i, row := range rows {
		if len(row) != d {
			return core.NewError(core.Sprintf("row %d has %d elements, want %d", i, len(row), d))
		}
	}
	return nil
}

// pooledRelativeMSE runs keyCodec over keys and valueCodec over values and
// pools BOTH sides' squared error and squared norm into one relative-MSE
// figure. measureOne reports one codec against one source, so a blended
// K-codec+V-codec figure needs its own small loop; it reuses the same
// squaredL2Diff/squaredL2 primitives measureOne itself is built from
// (instrument.go).
func pooledRelativeMSE(keyCodec Codec, keys [][]float32, valueCodec Codec, values [][]float32) (rowMSE, rowMSERelative float64, samples int) {
	var sumSq, sumDenom float64
	accumulate := func(codec Codec, rows [][]float32) {
		for _, row := range rows {
			recon := codec.Decode(codec.Encode(row), len(row))
			sumSq += squaredL2Diff(row, recon)
			sumDenom += squaredL2(row)
			samples++
		}
	}
	accumulate(keyCodec, keys)
	accumulate(valueCodec, values)
	if samples > 0 {
		rowMSE = sumSq / float64(samples)
	}
	if sumDenom > 0 {
		rowMSERelative = sumSq / sumDenom
	}
	return rowMSE, rowMSERelative, samples
}

// FormatRealReport renders result as a fixed-width, human-readable table
// alongside the blended mixed-configuration line — the form the receipt
// pastes into its distortion table.
//
//	result, _ := turboquant.MeasureReal(keys, values, 42)
//	core.Println(turboquant.FormatRealReport(result))
func FormatRealReport(result RealMeasurement) string {
	out := core.Sprintf("TurboQuant real-KV distortion — d=%d\n", result.D)
	out += core.Sprintf("%-26s %-16s %6s %12s %12s %12s %12s %6s %6s\n",
		"codec", "source", "bytes", "row-mse", "rel-mse", "max-dSM", "out-rel-err", "n", "attn-n")
	for _, r := range result.Reports {
		out += core.Sprintf("%-26s %-16s %6d %12.4g %12.4g %12.4g %12.4g %6d %6d\n",
			r.Codec, r.DataSource, r.BytesPerRow, r.RowMSE, r.RowMSERelative,
			r.MaxSoftmaxDelta, r.OutputRelError, r.Samples, r.AttentionRows)
	}
	out += core.Sprintf("%-26s %-16s %6s %12.4g %12.4g %12s %12s %6d %6s\n",
		core.Sprintf("Mixed-K%d/V%d(%.1fb)", RealMixedKeyBits, RealMixedValueBits, RealMixedNominalBits),
		"real-kv-pooled", "-", result.Mixed.RowMSE, result.Mixed.RowMSERelative, "-", "-", result.Mixed.Samples, "-")
	return out
}

// SaveRealKVRows writes rows (dimension d = len(rows[0])) to path in
// LoadRealKVRows' wire format: a little-endian uint32 dimension header
// followed by consecutive f32-LE rows — LoadRealKVRows' write-side
// companion. Capture tooling (the live capture harness that populates a
// testdata fixture, or MeasureCodecs' optional real-KV source) writes
// through this rather than a bespoke format.
//
//	err := SaveRealKVRows("testdata/real_kv_keys.bin", keys)
func SaveRealKVRows(path string, rows [][]float32) error {
	if len(rows) == 0 {
		return core.NewError("turboquant.SaveRealKVRows: no rows")
	}
	d := len(rows[0])
	if d == 0 {
		return core.NewError("turboquant.SaveRealKVRows: zero-dimension rows")
	}
	if err := validateRealRows(rows, d); err != nil {
		return core.E("turboquant.SaveRealKVRows", "rows", err)
	}
	data := make([]byte, 4+4*d*len(rows))
	putUint32LE(data, uint32(d))
	off := 4
	for _, row := range rows {
		for _, v := range row {
			putFloat32LE(data[off:], v)
			off += 4
		}
	}
	if r := core.WriteFile(path, data, 0o644); !r.OK {
		cause, _ := r.Value.(error)
		return core.E("turboquant.SaveRealKVRows", "write "+path, cause)
	}
	return nil
}
