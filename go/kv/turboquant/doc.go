// SPDX-Licence-Identifier: EUPL-1.2

// Package turboquant is a host-only, engine-neutral reference implementation
// and measurement instrument for TurboQuant KV-cache quantisation
// (arxiv.org/abs/2504.19874, RFC #41 slice S1). It has no engine/metal
// dependency, no build tags, no GPU — this is the correctness oracle and
// decision instrument other slices build the kernel work against, not the
// kernel work itself.
//
// Two codecs, both operating per row (one K or V head-row of dimension d,
// 64-256 in practice):
//
//   - Q_mse — the MSE-optimal codec (EncodeQMSE/DecodeQMSE): normalise by the
//     row's L2 norm, rotate by a seeded random orthogonal matrix, quantise
//     each rotated coordinate independently against Lloyd-Max centroids
//     solved for that coordinate's sphere-marginal density.
//   - Q_prod — the inner-product-preserving codec (EncodeQProd/DecodeQProd):
//     Q_mse at (b-1) bits, plus a 1-bit QJL sign sketch of the residual, so a
//     downstream attention dot product stays (near-)unbiased.
//
// Plus the paper's practical mixed-bit outlier split (EncodeMixed/
// DecodeMixed, calibrated by CalibrateMixedSplit) and two plain baselines to
// measure against — symmetric group-quant int8/int4 at group size 64
// (EncodeGroupQuantInt8/EncodeGroupQuantInt4), reimplemented host-side from
// engine/metal's paged KV q8 scheme rather than imported (this package must
// build without the darwin/arm64 Metal engine).
//
// MeasureCodecs is the decision instrument: it runs every codec against
// synthetic Gaussian and sphere-uniform rows (always) and an optional
// real-KV fixture (testdata/real_kv_rows.bin, captured by the orchestrator
// at merge — absent in a fresh checkout, skipped cleanly), reporting
// bytes-per-row, row MSE, and attention-level error for each.
//
//	result := turboquant.MeasureCodecs(128, 42, 4000, turboquant.DefaultRealKVFixturePath)
//	core.Println(turboquant.FormatReport(result))
package turboquant
