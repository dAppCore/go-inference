# TODO.md — go-inference Task Queue

Dispatched from core/go orchestration. This package is minimal by design.

---

## Phase 1: Foundation

- [ ] **Add tests for option application** — Verify GenerateConfig defaults, all With* options, ApplyGenerateOpts/ApplyLoadOpts behaviour.
- [ ] **Add tests for backend registry** — Register, Get, List, Default priority order, LoadModel routing.
- [ ] **Add tests for Default() platform preference** — Verify metal > rocm > llama_cpp ordering.

## Phase 2: Integration

- [ ] **go-mlx migration** — go-mlx Phase 4 backend abstraction should import go-inference instead of defining its own TextModel/Backend. Update go-mlx's design doc and plan to reference this package.
- [ ] **go-rocm implementation** — go-rocm implements inference.Backend + inference.TextModel.
- [ ] **go-ml migration** — go-ml's Backend/StreamingBackend should align with or wrap inference.TextModel. The go-ml Backend adds context.Context + non-streaming helpers on top.

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
