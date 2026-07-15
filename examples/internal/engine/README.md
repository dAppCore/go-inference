<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# internal/engine — the platform composition

One blank import selects the GPU engine for the running platform and registers
every built-in model architecture:

```go
import _ "dappco.re/go/inference/examples/internal/engine"
```

darwin/arm64 → `engine/metal` · linux/amd64 → `engine/hip` (amd/cuda/cpu
lanes) · elsewhere → no engine (LoadModel reports "no backends available").
Copy these three build-tagged files into your own app as the starting point —
see [../../pkg/README.md](../../pkg/README.md) for the per-backend builds.
