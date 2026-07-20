// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sort"
	"testing"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// TestKernelPSOStats is the .metal-source analogue of Go's allocs/op sweep (#393 / the
// AX-11 method): the Metal compiler caps each pipeline's occupancy by its register and
// threadgroup-memory appetite, and that cap is queryable — MaxTotalThreadsPerThreadgroup
// at 1024 means register-clean; anything lower means the compiler register-limited the
// kernel (the GPU-side equivalent of a hot allocation). This test builds every function
// in OUR custom metallib plus the MLX kernels the engine actually dispatches, and logs
// the table ranked worst-first — the work list for the kernel-source pass. It fails on
// nothing (the receipts bench prices any fix); it exists so the ranking is one
// `-run TestKernelPSOStats -v` away.
//
// VERDICT (M3 Ultra, metallib @ mlx v0.32.0 + lthn): 95 kernels surveyed,
// ZERO register-limited — every plainly-instantiable kernel in both libraries compiles
// at the full 1024-thread cap. Threadgroup memory tops out at 18.5KB
// (lthn_logits_sample_bf16) and 16.4KB (the paged-SDPA p1 tiles) — reduction kernels
// whose tile IS the algorithm, not accidents. Cross-checked with the receipts bench
// (fat kernels at 567–807 GB/s ≈ roofline), the .metal source is already at the bar:
// the GPU analogues of Go's allocs/op waste (register spills, occupancy caps, bad
// access patterns) are absent. Remaining wins live at dispatch shape (#392's levers),
// not kernel source; per-line ALU nits would need the Xcode shader profiler on a
// captured trace, with expected diminishing returns.
func TestKernelPSOStats(t *testing.T) {
	requireNativeRuntime(t)

	type row struct {
		name       string
		maxThreads uint
		execWidth  uint
		staticTG   uint
		lib        string
	}
	var rows []row
	var skipped []string

	// Some kernels ABORT plain PSO creation with an uncatchable ObjC exception (SIGABRT):
	// function-constant kernels and the steel-attn probe family. Skip them by prefix —
	// the receipts bench covers instantiable ones at their real shapes.
	skipPrefixes := []string{"lthn_attn_splitd", "lthn_qmv_rows", "lthn_gather_qmv", "lthn_sdpa_vector_2pass_1_q8"}
	build := func(libName, name string) (pso metal.MTLComputePipelineState) {
		defer func() { _ = recover() }()
		for _, pre := range skipPrefixes {
			if len(name) >= len(pre) && name[:len(pre)] == pre {
				return nil
			}
		}
		if libName == "lthn" { // pipelineFor resolves the MAIN library only
			fn := customLibrary.NewFunctionWithName(name)
			if fn != nil && fn.GetID() != 0 {
				if fn.FunctionConstantsDictionary().Count() > 0 {
					return nil // requires function constants — plain PSO creation aborts
				}
				pso, _ = device.NewComputePipelineStateWithFunctionError(fn)
			}
			return pso
		}
		if p, err := pipelineFor(name); err == nil {
			pso = p
		}
		return pso
	}
	addFromLibrary := func(libName string, names []string) {
		for _, name := range names {
			pso := build(libName, name)
			if pso == nil {
				skipped = append(skipped, name+" ("+libName+")")
				continue
			}
			rows = append(rows, row{
				name:       name,
				maxThreads: pso.MaxTotalThreadsPerThreadgroup(),
				execWidth:  pso.ThreadExecutionWidth(),
				staticTG:   pso.StaticThreadgroupMemoryLength(),
				lib:        libName,
			})
		}
	}

	// Our custom library: every function it ships (function-constant kernels fail plain
	// NewFunctionWithName and land in skipped — lthn_qmv_rows etc. are covered by the
	// receipts bench at their instantiated shapes instead).
	if customLibrary != nil && customLibrary.GetID() != 0 {
		addFromLibrary("lthn", customLibrary.FunctionNames())
	} else {
		t.Log("custom metallib not loaded — lthn kernels not surveyed")
	}

	// The MLX kernels the engine dispatches (the receipts' working set — extend as
	// dispatch sites grow; enumerating ALL of MLX's instantiations is noise).
	addFromLibrary("mlx", []string{
		"affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0",
		"affine_qmv_bfloat16_t_gs_64_b_4_batch_0",
		"affine_qmv_wide_bfloat16_t_gs_64_b_4_nv_2_kl_8_batch_0",
		"affine_qmm_t_bfloat16_t_gs_64_b_4_alN_true_batch_0",
		"affine_gather_qmv_bfloat16_t_gs_64_b_4",
		"affine_gather_qmv_fast_bfloat16_t_gs_64_b_4",
		gemvKernelName("bfloat16", 4, 8, 4, 1, 4, 4),
		rmsKernelBF16(2816),
		rmsKernelBF16(5376),
		"rmsbfloat16",
		"vv_Addbfloat16",
		"vv_Multiplybfloat16",
	})
	// MLX's sdpa family needs function constants — build through the engine's own
	// constructors (which supply them) and append the rows directly.
	if pso, err := sdpaVectorPipelineForHeadDim(256); err == nil {
		rows = append(rows, row{name: "sdpa_vector_bfloat16_t_256_256", lib: "mlx",
			maxThreads: pso.MaxTotalThreadsPerThreadgroup(), execWidth: pso.ThreadExecutionWidth(), staticTG: pso.StaticThreadgroupMemoryLength()})
	}
	if pso, err := sdpaVector2Pass1Pipeline(core.Sprintf("sdpa_vector_2pass_1_bfloat16_t_%d_%d", 256, 256), 128); err == nil {
		rows = append(rows, row{name: "sdpa_vector_2pass_1_bfloat16_t_256_256", lib: "mlx",
			maxThreads: pso.MaxTotalThreadsPerThreadgroup(), execWidth: pso.ThreadExecutionWidth(), staticTG: pso.StaticThreadgroupMemoryLength()})
	}
	if pso, err := sdpaVector2Pass2Pipeline(core.Sprintf("sdpa_vector_2pass_2_bfloat16_t_%d", 256)); err == nil {
		rows = append(rows, row{name: "sdpa_vector_2pass_2_bfloat16_t_256", lib: "mlx",
			maxThreads: pso.MaxTotalThreadsPerThreadgroup(), execWidth: pso.ThreadExecutionWidth(), staticTG: pso.StaticThreadgroupMemoryLength()})
	}

	sort.Slice(rows, func(i, j int) bool {
		if rows[i].maxThreads != rows[j].maxThreads {
			return rows[i].maxThreads < rows[j].maxThreads
		}
		return rows[i].staticTG > rows[j].staticTG
	})

	flagged := 0
	t.Logf("%-64s %-5s %10s %6s %9s", "kernel", "lib", "maxThreads", "width", "staticTG")
	for _, r := range rows {
		mark := "  "
		if r.maxThreads < 1024 {
			mark = "⚠ " // register-limited: the compiler capped occupancy below a full threadgroup
			flagged++
		}
		t.Logf("%s%-62s %-5s %10d %6d %9d", mark, r.name, r.lib, r.maxThreads, r.execWidth, r.staticTG)
	}
	t.Logf("%d kernels surveyed, %d register-limited (maxThreads < 1024), %d skipped (function constants / absent)", len(rows), flagged, len(skipped))
	for _, s := range skipped {
		t.Logf("  skipped: %s", s)
	}
}
