# TODO.md — go-inference Task Queue

Dispatched from core/go orchestration. This package is minimal by design.

---

## Phase 1: Foundation — `d76448d` (Charon)

- [x] **Add tests for option application** — Verify GenerateConfig defaults, all With* options, ApplyGenerateOpts/ApplyLoadOpts behaviour. Comprehensive API tests (1,074 LOC).
- [x] **Add tests for backend registry** — Register, Get, List, Default priority order, LoadModel routing.
- [x] **Add tests for Default() platform preference** — Verify metal > rocm > llama_cpp ordering.

## Phase 2: Integration — COMPLETE

- [x] **go-mlx migration** — `register_metal.go` implements `inference.Backend` via `metalBackend{}` + `metalAdapter{}` wrapping `internal/metal.Model`. Auto-registers via `inference.Register()` in `init()`. Build-tagged `darwin && arm64`. Full TextModel coverage: Generate, Chat, Classify, BatchGenerate, Info, Metrics, Err, Close.
- [x] **go-rocm implementation** — `register_rocm.go` implements `inference.Backend` + `inference.TextModel` via llama-server subprocess. Auto-registers via `inference.Register(&rocmBackend{})`. Phase 4 complete (5,794 LOC by Charon).
- [x] **go-ml migration** — `adapter.go` bridges `inference.TextModel` → `ml.Backend/StreamingBackend` (118 LOC, 13 tests). `backend_mlx.go` collapsed from 253 to 35 LOC using `inference.LoadModel`. `backend_http_textmodel.go` provides reverse wrappers (135 LOC, 19 tests).

## Phase 3: Extended Interfaces (when needed)

- [ ] **BatchModel interface** — When go-i18n needs 5K sentences/sec, add: `type BatchModel interface { TextModel; BatchGenerate(ctx, []string, ...GenerateOption) iter.Seq2[int, Token] }`. Not before it's needed.
- [ ] **Stats interface** — When LEM Lab dashboard needs metrics: `type StatsModel interface { TextModel; Stats() GenerateStats }` with tokens/sec, peak memory, GPU util.

---

## Design Principles

1. **Minimal interface** — Only add methods when 2+ consumers need them
2. **Zero dependencies** — stdlib only, compiles everywhere
3. **Backwards compatible** — New interfaces extend, never modify existing ones
4. **Platform agnostic** — No build tags, no CGO, no OS-specific code

## Workflow

1. Virgil in core/go manages this package directly (too small for a dedicated Claude)
2. Changes here are coordinated with go-mlx and go-rocm Claudes via their TODO.md
3. New interface methods require Virgil approval before adding
