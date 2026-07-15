# Coverage-spec findings

The coverage-spec tests characterize the current contracts of
`speculative_model.go` and `flash_prompt.go` without changing production code.

- The speculative adapter reports pair-validation errors through `Err`, records
  prompt/generated counts, and exposes unsupported `Classify` and
  `BatchGenerate` operations as failed `core.Result` values.
- The flash pipeline getters cache unavailable-kernel results. The tests use a
  loaded but wrong metallib to pin those error contracts, while GPU-only tests
  compare successful split-D, q8, sliding-window, and prompt flash outputs to
  the existing multi-query reference lanes.
- No production bug was found in this write-blind pass. GPU parity and actual
  coverage percentages remain orchestrator-owned because this sandbox has no
  Metal device; the parity tests skip when their required compiled lane is not
  present.

Functions intentionally not reached by a successful GPU path in this sandbox:
the orchestrator must run the GPU-gated tests to determine whether the optional
split-D, q8, and sliding-window kernels are installed. Their wrong-metallib
error paths are covered here in source and will be executed on a configured
orchestrator.
