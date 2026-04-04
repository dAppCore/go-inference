# RFC-025: Agent Experience (AX) Design Principles

- **Status:** Draft
- **Authors:** Snider, Cladius
- **Date:** 2026-03-19
- **Applies to:** All Core ecosystem packages (CoreGO, CorePHP, CoreTS, core-agent)

## Abstract

Agent Experience (AX) is a design paradigm for software systems where the primary code consumer is an AI agent, not a human developer. AX sits alongside User Experience (UX) and Developer Experience (DX) as the third era of interface design.

This RFC establishes AX as a formal design principle for the Core ecosystem and defines the conventions that follow from it.

## Motivation

As of early 2026, AI agents write, review, and maintain the majority of code in the Core ecosystem. The original author has not manually edited code (outside of Core struct design) since October 2025. Code is processed semantically — agents reason about intent, not characters.

Design patterns inherited from the human-developer era optimise for the wrong consumer:

- **Short names** save keystrokes but increase semantic ambiguity
- **Functional option chains** are fluent for humans but opaque for agents tracing configuration
- **Error-at-every-call-site** produces 50% boilerplate that obscures intent
- **Generic type parameters** force agents to carry type context that the runtime already has
- **Panic-hiding conventions** (`Must*`) create implicit control flow that agents must special-case

AX acknowledges this shift and provides principles for designing code, APIs, file structures, and conventions that serve AI agents as first-class consumers.

## The Three Eras

| Era | Primary Consumer | Optimises For | Key Metric |
|-----|-----------------|---------------|------------|
| UX | End users | Discoverability, forgiveness, visual clarity | Task completion time |
| DX | Developers | Typing speed, IDE support, convention familiarity | Time to first commit |
| AX | AI agents | Predictability, composability, semantic navigation | Correct-on-first-pass rate |

AX does not replace UX or DX. End users still need good UX. Developers still need good DX. But when the primary code author and maintainer is an AI agent, the codebase should be designed for that consumer first.

## Principles

### 1. Predictable Names Over Short Names

Names are tokens that agents pattern-match across languages and contexts. Abbreviations introduce mapping overhead.

```
Config    not  Cfg
Service   not  Srv
Embed     not  Emb
Error     not  Err (as a subsystem name; err for local variables is fine)
Options   not  Opts
```

**Rule:** If a name would require a comment to explain, it is too short.

**Exception:** Industry-standard abbreviations that are universally understood (`HTTP`, `URL`, `ID`, `IPC`, `I18n`) are acceptable. The test: would an agent trained on any mainstream language recognise it without context?

### 2. Comments as Usage Examples

The function signature tells WHAT. The comment shows HOW with real values.

```go
// Detect the project type from files present
setup.Detect("/path/to/project")

// Set up a workspace with auto-detected template
setup.Run(setup.Options{Path: ".", Template: "auto"})

// Scaffold a PHP module workspace
setup.Run(setup.Options{Path: "./my-module", Template: "php"})
```

**Rule:** If a comment restates what the type signature already says, delete it. If a comment shows a concrete usage with realistic values, keep it.

**Rationale:** Agents learn from examples more effectively than from descriptions. A comment like "Run executes the setup process" adds zero information. A comment like `setup.Run(setup.Options{Path: ".", Template: "auto"})` teaches an agent exactly how to call the function.

### 3. Path Is Documentation

File and directory paths should be self-describing. An agent navigating the filesystem should understand what it is looking at without reading a README.

```
flow/deploy/to/homelab.yaml    — deploy TO the homelab
flow/deploy/from/github.yaml   — deploy FROM GitHub
flow/code/review.yaml           — code review flow
template/file/go/struct.go.tmpl — Go struct file template
template/dir/workspace/php/     — PHP workspace scaffold
```

**Rule:** If an agent needs to read a file to understand what a directory contains, the directory naming has failed.

**Corollary:** The unified path convention (folder structure = HTTP route = CLI command = test path) is AX-native. One path, every surface.

### 4. Templates Over Freeform

When an agent generates code from a template, the output is constrained to known-good shapes. When an agent writes freeform, the output varies.

```go
// Template-driven — consistent output
lib.RenderFile("php/action", data)
lib.ExtractDir("php", targetDir, data)

// Freeform — variance in output
"write a PHP action class that..."
```

**Rule:** For any code pattern that recurs, provide a template. Templates are guardrails for agents.

**Scope:** Templates apply to file generation, workspace scaffolding, config generation, and commit messages. They do NOT apply to novel logic — agents should write business logic freeform with the domain knowledge available.

### 5. Declarative Over Imperative

Agents reason better about declarations of intent than sequences of operations.

```yaml
# Declarative — agent sees what should happen
steps:
  - name: build
    flow: tools/docker-build
    with:
      context: "{{ .app_dir }}"
      image_name: "{{ .image_name }}"

  - name: deploy
    flow: deploy/with/docker
    with:
      host: "{{ .host }}"
```

```go
// Imperative — agent must trace execution
cmd := exec.Command("docker", "build", "--platform", "linux/amd64", "-t", imageName, ".")
cmd.Dir = appDir
if err := cmd.Run(); err != nil {
    return fmt.Errorf("docker build: %w", err)
}
```

**Rule:** Orchestration, configuration, and pipeline logic should be declarative (YAML/JSON). Implementation logic should be imperative (Go/PHP/TS). The boundary is: if an agent needs to compose or modify the logic, make it declarative.

### 6. Universal Types (Core Primitives)

Every component in the ecosystem accepts and returns the same primitive types. An agent processing any level of the tree sees identical shapes.

```go
// Universal contract
setup.Run(core.Options{Path: ".", Template: "auto"})
brain.New(core.Options{Name: "openbrain"})
deploy.Run(core.Options{Flow: "deploy/to/homelab"})

// Fractal — Core itself is a Service
core.New(core.Options{
    Services: []core.Service{
        process.New(core.Options{Name: "process"}),
        brain.New(core.Options{Name: "brain"}),
    },
})
```

**Core primitive types:**

| Type | Purpose |
|------|---------|
| `core.Options` | Input configuration (what you want) |
| `core.Config` | Runtime settings (what is active) |
| `core.Data` | Embedded or stored content |
| `core.Service` | A managed component with lifecycle |
| `core.Result[T]` | Return value with OK/fail state |

**What this replaces:**

| Go Convention | Core AX | Why |
|--------------|---------|-----|
| `func With*(v) Option` | `core.Options{Field: v}` | Struct literal is parseable; option chain requires tracing |
| `func Must*(v) T` | `core.Result[T]` | No hidden panics; errors flow through Core |
| `func *For[T](c) T` | `c.Service("name")` | String lookup is greppable; generics require type context |
| `val, err :=` everywhere | Single return via `core.Result` | Intent not obscured by error handling |
| `_ = err` | Never needed | Core handles all errors internally |

### 7. Directory as Semantics

The directory structure tells an agent the intent before it reads a word. Top-level directories are semantic categories, not organisational bins.

```
plans/
├── code/       # Pure primitives — read for WHAT exists
├── project/    # Products — read for WHAT we're building and WHY
└── rfc/        # Contracts — read for constraints and rules
```

**Rule:** An agent should know what kind of document it's reading from the path alone. `code/core/go/io/RFC.md` = a lib primitive spec. `project/ofm/RFC.md` = a product spec that cross-references code/. `rfc/snider/borg/RFC-BORG-006-SMSG-FORMAT.md` = an immutable contract for the Borg SMSG protocol.

**Corollary:** The three-way split (code/project/rfc) extends principle 3 (Path Is Documentation) from files to entire subtrees. The path IS the metadata.

### 8. Lib Never Imports Consumer

Dependency flows one direction. Libraries define primitives. Consumers compose from them. A new feature in a consumer can never break a library.

```
code/core/go/*     → lib tier (stable foundation)
code/core/agent/   → consumer tier (composes from go/*)
code/core/cli/     → consumer tier (composes from go/*)
code/core/gui/     → consumer tier (composes from go/*)
```

**Rule:** If package A is in `go/` and package B is in the consumer tier, B may import A but A must never import B. The repo naming convention enforces this: `go-{name}` = lib, bare `{name}` = consumer.

**Why this matters for agents:** When an agent is dispatched to implement a feature in `core/agent`, it can freely import from `go-io`, `go-scm`, `go-process`. But if an agent is dispatched to `go-io`, it knows its changes are foundational — every consumer depends on it, so the contract must not break.

### 9. Issues Are N+(rounds) Deep

Problems in code and specs are layered. Surface issues mask deeper issues. Fixing the surface reveals the next layer. This is not a failure mode — it is the discovery process.

```
Pass 1: Find 16 issues (surface — naming, imports, obvious errors)
Pass 2: Find 11 issues (structural — contradictions, missing types)
Pass 3: Find 5 issues (architectural — signature mismatches, registration gaps)
Pass 4: Find 4 issues (contract — cross-spec API mismatches)
Pass 5: Find 2 issues (mechanical — path format, nil safety)
Pass N: Findings are trivial → spec/code is complete
```

**Rule:** Iteration is required, not a failure. Each pass sees what the previous pass could not, because the context changed. An agent dispatched with the same task on the same repo will find different things each time — this is correct behaviour.

**Corollary:** The cheapest model should do the most passes (surface work). The frontier model should arrive last, when only deep issues remain. Tiered iteration: grunt model grinds → mid model pre-warms → frontier model polishes.

**Anti-pattern:** One-shot generation expecting valid output. No model, no human, produces correct-on-first-pass for non-trivial work. Expecting it wastes the first pass on surface issues that a cheaper pass would have caught.

### 10. CLI Tests as Artifact Validation

Unit tests verify the code. CLI tests verify the binary. The directory structure IS the command structure — path maps to command, Taskfile runs the test.

```
tests/cli/
├── core/
│   └── lint/
│       ├── Taskfile.yaml          ← test `core-lint` (root)
│       ├── run/
│       │   ├── Taskfile.yaml      ← test `core-lint run`
│       │   └── fixtures/
│       ├── go/
│       │   ├── Taskfile.yaml      ← test `core-lint go`
│       │   └── fixtures/
│       └── security/
│           ├── Taskfile.yaml      ← test `core-lint security`
│           └── fixtures/
```

**Rule:** Every CLI command has a matching `tests/cli/{path}/Taskfile.yaml`. The Taskfile runs the compiled binary against fixtures with known inputs and validates the output. If the CLI test passes, the underlying actions work — because CLI commands call actions, MCP tools call actions, API endpoints call actions. Test the CLI, trust the rest.

**Pattern:**

```yaml
# tests/cli/core/lint/go/Taskfile.yaml
version: '3'
tasks:
  test:
    cmds:
      - core-lint go --output json fixtures/ > /tmp/result.json
      - jq -e '.findings | length > 0' /tmp/result.json
      - jq -e '.summary.passed == false' /tmp/result.json
```

**Why this matters for agents:** An agent can validate its own work by running `task test` in the matching `tests/cli/` directory. No test framework, no mocking, no setup — just the binary, fixtures, and `jq` assertions. The agent builds the binary, runs the test, sees the result. If it fails, the agent can read the fixture, read the output, and fix the code.

**Corollary:** Fixtures are planted bugs. Each fixture file has a known issue that the linter must find. If the linter doesn't find it, the test fails. Fixtures are the spec for what the tool must detect — they ARE the test cases, not descriptions of test cases.

## Applying AX to Existing Patterns

### File Structure

```
# AX-native: path describes content
core/agent/
├── go/                    # Go source
├── php/                   # PHP source
├── ui/                    # Frontend source
├── claude/                # Claude Code plugin
└── codex/                 # Codex plugin

# Not AX: generic names requiring README
src/
├── lib/
├── utils/
└── helpers/
```

### Error Handling

```go
// AX-native: errors are infrastructure, not application logic
svc := c.Service("brain")
cfg := c.Config().Get("database.host")
// Errors logged by Core. Code reads like a spec.

// Not AX: errors dominate the code
svc, err := c.ServiceFor[brain.Service]()
if err != nil {
    return fmt.Errorf("get brain service: %w", err)
}
cfg, err := c.Config().Get("database.host")
if err != nil {
    _ = err // silenced because "it'll be fine"
}
```

### API Design

```go
// AX-native: one shape, every surface
core.New(core.Options{
    Name: "my-app",
    Services: []core.Service{...},
    Config: core.Config{...},
})

// Not AX: multiple patterns for the same thing
core.New(
    core.WithName("my-app"),
    core.WithService(factory1),
    core.WithService(factory2),
    core.WithConfig(cfg),
)
```

## The Plans Convention — AX Development Lifecycle

The `plans/` directory structure encodes a development methodology designed for how generative AI actually works: iterative refinement across structured phases, not one-shot generation.

### The Three-Way Split

```
plans/
├── project/    # 1. WHAT and WHY — start here
├── rfc/        # 2. CONSTRAINTS — immutable contracts
└── code/       # 3. HOW — implementation specs
```

Each directory is a phase. Work flows from project → rfc → code. Each transition forces a refinement pass — you cannot write a code spec without discovering gaps in the project spec, and you cannot write an RFC without discovering assumptions in both.

**Three places for data that can't be written simultaneously = three guaranteed iterations of "actually, this needs changing."** Refinement is baked into the structure, not bolted on as a review step.

### Phase 1: Project (Vision)

Start with `project/`. No code exists yet. Define:
- What the product IS and who it serves
- What existing primitives it consumes (cross-ref to `code/`)
- What constraints it operates under (cross-ref to `rfc/`)

This is where creativity lives. Map features to building blocks. Connect systems. The project spec is integrative — it references everything else.

### Phase 2: RFC (Contracts)

Extract the immutable rules into `rfc/`. These are constraints that don't change with implementation:
- Wire formats, protocols, hash algorithms
- Security properties that must hold
- Compatibility guarantees

RFCs are numbered per component (`RFC-BORG-006-SMSG-FORMAT.md`) and never modified after acceptance. If the contract changes, write a new RFC.

### Phase 3: Code (Implementation Specs)

Define the implementation in `code/`. Each component gets an RFC.md that an agent can implement from:
- Struct definitions (the DTOs — see principle 6)
- Method signatures and behaviour
- Error conditions and edge cases
- Cross-references to other code/ specs

The code spec IS the product. Write the spec → dispatch to an agent → review output → iterate.

### Pre-Launch: Alignment Protocol

Before dispatching for implementation, verify spec-model alignment:

```
1. REVIEW — The implementation model (Codex/Jules) reads the spec
   and reports missing elements. This surfaces the delta between
   the model's training and the spec's assumptions.

   "I need X, Y, Z to implement this" is the model saying
   "I hear you but I'm missing context" — without asking.

2. ADJUST — Update the spec to close the gaps. Add examples,
   clarify ambiguities, provide the context the model needs.
   This is shared alignment, not compromise.

3. VERIFY — A different model (or sub-agent) reviews the adjusted
   spec without the planner's bias. Fresh eyes on the contract.
   "Does this make sense to someone who wasn't in the room?"

4. READY — When the review findings are trivial or deployment-
   related (not architectural), the spec is ready to dispatch.
```

### Implementation: Iterative Dispatch

Same prompt, multiple runs. Each pass sees deeper because the context evolved:

```
Round 1: Build features (the obvious gaps)
Round 2: Write tests (verify what was built)
Round 3: Harden security (what can go wrong?)
Round 4: Next RFC section (what's still missing?)
Round N: Findings are trivial → implementation is complete
```

Re-running is not failure. It is the process. Each pass changes the codebase, which changes what the next pass can see. The iteration IS the refinement.

### Post-Implementation: Auto-Documentation

The QA/verify chain produces artefacts that feed forward:
- Test results document the contract (what works, what doesn't)
- Coverage reports surface untested paths
- Diff summaries prep the changelog for the next release
- Doc site updates from the spec (the spec IS the documentation)

The output of one cycle is the input to the next. The plans repo stays current because the specs drive the code, not the other way round.

## Compatibility

AX conventions are valid, idiomatic Go/PHP/TS. They do not require language extensions, code generation, or non-standard tooling. An AX-designed codebase compiles, tests, and deploys with standard toolchains.

The conventions diverge from community patterns (functional options, Must/For, etc.) but do not violate language specifications. This is a style choice, not a fork.

## Adoption

AX applies to all new code in the Core ecosystem. Existing code migrates incrementally as it is touched — no big-bang rewrite.

Priority order:
1. **Public APIs** (package-level functions, struct constructors)
2. **File structure** (path naming, template locations)
3. **Internal fields** (struct field names, local variables)

## References

- dAppServer unified path convention (2024)
- CoreGO DTO pattern refactor (2026-03-18)
- Core primitives design (2026-03-19)
- Go Proverbs, Rob Pike (2015) — AX provides an updated lens

## Changelog

- 2026-03-19: Initial draft
