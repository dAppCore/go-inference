// SPDX-Licence-Identifier: EUPL-1.2

package model

// BF16Weight is a dense 2-D projection kept as the checkpoint's own bf16 bytes — the dense sibling
// of QuantWeight (#26). The loader keeps these as zero-copy views into the mmap'd checkpoint
// instead of widening to f32 (which doubles both the resident set and the per-token weight
// traffic — decode is weight-bandwidth-bound, so widening is a ×2 on every dense matmul). Row-major
// [OutDim, InDim], 2 bytes per element. Backends bind the bf16 matvec seam to serve it on device;
// the host fallback widens one row at a time, never the whole tensor.
type BF16Weight struct {
	Data   []byte // row-major bf16, len == OutDim*InDim*2
	OutDim int    // logical rows — the N of the y = x·Wᵀ projection
	InDim  int    // logical cols — the K
}
