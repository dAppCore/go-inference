// SPDX-Licence-Identifier: EUPL-1.2

package turboquant

import (
	"math"
	"math/rand/v2"

	core "dappco.re/go"
)

// DefaultRealKVFixturePath is the conventional location for the optional
// real-KV measurement fixture (RFC #41 slice S1). It is not part of this
// commit — the orchestrator captures real rows at merge — so
// MeasureCodecs's default call legitimately runs without it.
const DefaultRealKVFixturePath = "testdata/real_kv_rows.bin"

// attentionN is the causal-attention window MeasureCodecs simulates: one
// exact query attending over up to this many candidate K/V rows, the decode-
// step shape a live KV cache serves.
const attentionN = 512

// LoadRealKVRows reads the real-KV measurement fixture at path: a
// little-endian binary file whose first 4 bytes are a uint32 row dimension
// d, followed by consecutive f32-LE rows of width d.
//
// ok is false with err == nil when path simply does not exist — a fresh
// checkout has no captured fixture yet, and that is not a failure, per the
// task's "skip that source cleanly when absent". err is non-nil only for a
// file that exists but is malformed (too short, non-positive dimension,
// payload not a multiple of the row width) — that IS a real problem worth
// surfacing.
//
//	rows, d, ok, err := LoadRealKVRows(DefaultRealKVFixturePath)
//	if err != nil { t.Fatal(err) }
//	if !ok { t.Skip("no real-KV fixture captured yet") }
func LoadRealKVRows(path string) (rows [][]float32, d int, ok bool, err error) {
	read := core.ReadFile(path)
	if !read.OK {
		return nil, 0, false, nil
	}
	data := read.Bytes()
	if len(data) < 4 {
		return nil, 0, false, core.E("turboquant.LoadRealKVRows", "file shorter than the 4-byte dimension header", nil)
	}
	d = int(getUint32LE(data))
	if d <= 0 {
		return nil, 0, false, core.E("turboquant.LoadRealKVRows", "row dimension header must be positive", nil)
	}
	payload := data[4:]
	rowBytes := d * 4
	if len(payload)%rowBytes != 0 {
		return nil, 0, false, core.E("turboquant.LoadRealKVRows", "payload length is not a multiple of the row byte width", nil)
	}
	n := len(payload) / rowBytes
	rows = make([][]float32, n)
	for i := 0; i < n; i++ {
		row := make([]float32, d)
		base := i * rowBytes
		for j := 0; j < d; j++ {
			row[j] = getFloat32LE(payload[base+j*4:])
		}
		rows[i] = row
	}
	return rows, d, true, nil
}

// generateGaussianRows fills n rows of dimension d with i.i.d. N(0,1)
// coordinates — arbitrary-norm synthetic activations.
func generateGaussianRows(rng *rand.Rand, n, d int) [][]float32 {
	out := make([][]float32, n)
	for i := range out {
		row := make([]float32, d)
		for j := range row {
			row[j] = float32(rng.NormFloat64())
		}
		out[i] = row
	}
	return out
}

// generateSphereUniformRows fills n rows of dimension d, each a uniform
// random direction on the unit (d-1)-sphere — the distribution the
// TurboQuant Lloyd-Max centroids are calibrated against, and the
// distortion-oracle tests' data source.
func generateSphereUniformRows(rng *rand.Rand, n, d int) [][]float32 {
	out := make([][]float32, n)
	for i := range out {
		g := make([]float64, d)
		var normSq float64
		for j := range g {
			g[j] = rng.NormFloat64()
			normSq += g[j] * g[j]
		}
		norm := math.Sqrt(normSq)
		row := make([]float32, d)
		for j := range g {
			row[j] = float32(g[j] / norm)
		}
		out[i] = row
	}
	return out
}

// dataSourceEntry is one named pool of rows MeasureCodecs measures every
// codec against.
type dataSourceEntry struct {
	name string
	rows [][]float32
}

// squaredL2Diff returns ||a-b||² accumulated in float64.
func squaredL2Diff(a, b []float32) float64 {
	var sum float64
	for i := range a {
		d := float64(a[i]) - float64(b[i])
		sum += d * d
	}
	return sum
}

// squaredL2 returns ||a||² accumulated in float64.
func squaredL2(a []float32) float64 {
	var sum float64
	for _, v := range a {
		sum += float64(v) * float64(v)
	}
	return sum
}

// attnOutput returns Σ weights[i]·rows[i] — the attention-weighted sum of
// value rows, accumulated in float64.
func attnOutput(weights []float64, rows [][]float32) []float64 {
	d := len(rows[0])
	out := make([]float64, d)
	for i, w := range weights {
		row := rows[i]
		for j := 0; j < d; j++ {
			out[j] += w * float64(row[j])
		}
	}
	return out
}

// measureAttention simulates one decode-step causal-attention query against
// up to attentionN rows drawn from rows (both K and V roles — a live cache
// keeps separate K/V tensors, but reusing one pool here is a fair stand-in
// for measuring quantisation error's effect on softmax weights and the
// attention output, and keeps the real-KV fixture's minimum size to
// attentionN rows rather than 2×attentionN). The query, when rows has a
// spare row beyond the K/V pool, is drawn from the SAME distribution as the
// source (rows[n]); otherwise it falls back to a fresh Gaussian draw.
//
// Returns the max absolute delta between the exact and candidate softmax
// weights, and the candidate output vector's relative L2 error against the
// exact output — plus n, the actual window size used (< attentionN if rows
// is short).
func measureAttention(codec Codec, rows [][]float32, d int, rng *rand.Rand) (maxSoftmaxDelta, outputRelError float64, n int) {
	n = min(len(rows), attentionN)
	if n == 0 {
		return 0, 0, 0
	}
	keys := rows[:n]

	var q []float64
	if len(rows) > n {
		q = toFloat64(rows[n])
	} else {
		qRow := make([]float64, d)
		for i := range qRow {
			qRow[i] = rng.NormFloat64()
		}
		q = qRow
	}
	scale := 1 / math.Sqrt(float64(d))

	scoresExact := make([]float64, n)
	for i, k := range keys {
		scoresExact[i] = dot(q, toFloat64(k)) * scale
	}
	weightsExact := softmax(scoresExact)
	outExact := attnOutput(weightsExact, keys)

	scoresCand := make([]float64, n)
	decodedV := make([][]float32, n)
	for i, k := range keys {
		kHat := codec.Decode(codec.Encode(k), d)
		scoresCand[i] = dot(q, toFloat64(kHat)) * scale
		decodedV[i] = kHat // same pool serves K and V roles; see doc comment
	}
	weightsCand := softmax(scoresCand)
	outCand := attnOutput(weightsCand, decodedV)

	for i := range weightsExact {
		if delta := math.Abs(weightsExact[i] - weightsCand[i]); delta > maxSoftmaxDelta {
			maxSoftmaxDelta = delta
		}
	}
	denom := l2Norm(outExact)
	if denom > 0 {
		outputRelError = l2Norm(subtract(outExact, outCand)) / denom
	}
	return maxSoftmaxDelta, outputRelError, n
}

// CodecReport is one (codec, data source) row of MeasureCodecs' report.
type CodecReport struct {
	Codec           string
	DataSource      string
	BytesPerRow     int
	RowMSE          float64 // mean ||x - x̃||² over the source's rows
	RowMSERelative  float64 // RowMSE's numerator over Σ||x||² — comparable across sources of different scale
	MaxSoftmaxDelta float64
	OutputRelError  float64
	Samples         int // rows the row-MSE columns were measured over
	AttentionRows   int // rows the attention columns were measured over (<= attentionN)
}

// measureOne runs codec against every row in src, returning the aggregate
// report row.
func measureOne(codec Codec, src dataSourceEntry, d int, seed uint64) CodecReport {
	n := len(src.rows)
	report := CodecReport{Codec: codec.Name(), DataSource: src.name, BytesPerRow: codec.BytesPerRow(d), Samples: n}
	if n == 0 {
		return report
	}
	var sumSq, sumDenom float64
	for _, row := range src.rows {
		recon := codec.Decode(codec.Encode(row), d)
		sumSq += squaredL2Diff(row, recon)
		sumDenom += squaredL2(row)
	}
	report.RowMSE = sumSq / float64(n)
	if sumDenom > 0 {
		report.RowMSERelative = sumSq / sumDenom
	}

	rng := rand.New(rand.NewPCG(deriveSeed(seed, attnSeedSalt), splitmix64(seed)))
	maxDelta, outRelErr, attnN := measureAttention(codec, src.rows, d, rng)
	report.MaxSoftmaxDelta = maxDelta
	report.OutputRelError = outRelErr
	report.AttentionRows = attnN
	return report
}

const attnSeedSalt uint64 = 5

// MixedSplitOutlierFraction is the fraction of channels the mixed-bit
// baseline promotes to OutlierBits — the paper's practical example promotes
// 32 of 128 channels (1/4), generalised here to any dimension.
const MixedSplitOutlierFraction = 0.25

// MeasureCodecs is the RFC #41 slice S1 instrument: it runs every codec in
// Codecs (TurboQuant Q_mse/Q_prod at 2-4 bits, the mixed-bit split, and the
// int8/int4 group-quant baselines) against every available data source at
// dimension d — synthetic Gaussian and sphere-uniform rows always, plus the
// real-KV fixture at realKVPath when present and dimension-matched — and
// reports bytes-per-row, row MSE, and attention-level error for each.
//
//	result := MeasureCodecs(128, 42, 4000, DefaultRealKVFixturePath)
//	core.Println(FormatReport(result))
func MeasureCodecs(d int, seed uint64, samplesPerSource int, realKVPath string) MeasureCodecsResult {
	rng := rand.New(rand.NewPCG(seed, splitmix64(seed)))
	// The attention window draws one extra row (see measureAttention) beyond
	// attentionN for the query, when available.
	poolSize := samplesPerSource
	if poolSize < attentionN+1 {
		poolSize = attentionN + 1
	}
	gaussian := generateGaussianRows(rng, poolSize, d)
	sphere := generateSphereUniformRows(rng, poolSize, d)

	k := int(math.Round(MixedSplitOutlierFraction * float64(d)))
	mixedSplit := CalibrateMixedSplit(gaussian, d, k, 2, 3)
	codecs := Codecs(seed, mixedSplit)

	sources := []dataSourceEntry{
		{name: "gaussian", rows: gaussian},
		{name: "sphere-uniform", rows: sphere},
	}
	if rows, fileD, ok, err := LoadRealKVRows(realKVPath); ok && err == nil {
		if fileD == d {
			sources = append(sources, dataSourceEntry{name: "real-kv", rows: rows})
		}
		// A fixture that exists but doesn't match d is silently skipped
		// rather than erroring — MeasureCodecs is called at one fixed
		// dimension per run, and a dimension-mismatched fixture is not this
		// call's data to use.
	}

	result := MeasureCodecsResult{D: d}
	for _, codec := range codecs {
		for _, src := range sources {
			result.Reports = append(result.Reports, measureOne(codec, src, d, seed))
		}
	}
	return result
}

// MeasureCodecsResult is MeasureCodecs' full report: one CodecReport per
// (codec, data source) combination actually measured.
type MeasureCodecsResult struct {
	D       int
	Reports []CodecReport
}

// FormatReport renders result as a fixed-width, human-readable table — the
// form the orchestrator pastes into the tracker.
//
//	result := MeasureCodecs(128, 42, 4000, DefaultRealKVFixturePath)
//	core.Println(FormatReport(result))
func FormatReport(result MeasureCodecsResult) string {
	out := core.Sprintf("TurboQuant KV codec measurement — d=%d\n", result.D)
	out += core.Sprintf("%-26s %-15s %6s %12s %12s %12s %12s %6s %6s\n",
		"codec", "source", "bytes", "row-mse", "rel-mse", "max-dSM", "out-rel-err", "n", "attn-n")
	for _, r := range result.Reports {
		out += core.Sprintf("%-26s %-15s %6d %12.4g %12.4g %12.4g %12.4g %6d %6d\n",
			r.Codec, r.DataSource, r.BytesPerRow, r.RowMSE, r.RowMSERelative,
			r.MaxSoftmaxDelta, r.OutputRelError, r.Samples, r.AttentionRows)
	}
	return out
}
