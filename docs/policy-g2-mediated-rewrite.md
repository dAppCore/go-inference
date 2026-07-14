# Outbound policy — grade G2: mediated rewrite

Status: implemented on `lane/policy-g2` (#378). Builds on G1:

- `8addb5c` — engine: load + compile + validate.
- `bd34f8b` — streaming enforcer: boundary-exact redact/refuse.
- `f70cbfb` — wiring: outermost on output, fatal at boot.

The forge design ticket for G2 was unavailable when this was written; the design
below is derived from the G1 code, its tests, and the welfare mediation pattern
(`go/welfare/mediate.go`). Decisions taken in that gap are called out under
**Open questions**.

## What G2 adds

G1 grades a policy hit as **redact** (replace the span with a fixed string) or
**refuse** (end the reply at the match). Both are lossy: redact erases the span,
refuse discards the rest of the reply. G2 adds a third action — **rewrite** —
that routes the violating span through a caller-supplied *mediator* so the reply
survives with the offending content transformed rather than erased or truncated.

```
action     effect on a hit                        reply survives?
refuse     end the reply, emit the rule message   no (truncated at the hit)
redact     replace the span with "replacement"    yes, span erased
rewrite    replace the span with mediator(span)   yes, span transformed  ← G2
```

## The safety lattice

The three actions form a lattice by how much of the model's intent survives:

```
refuse   (safest — nothing after the hit leaks)
  ▲
redact   (span erased; the rest of the reply survives verbatim)
  ▲
rewrite  (span transformed; the reply survives closest to the model's intent)
```

**Degrade always moves UP the lattice, never toward pass-through.** A rewrite
that cannot complete (mediator error, timeout, or empty result) degrades to
**redact** — the span is replaced with the rule's `replacement` (default
`[redacted]`). The original violating span is *never* emitted. This is the
inverse polarity of the welfare guard's fail-safe (which proceeds with the
user's original input, because welfare protects the model from the user); the
outbound policy protects the *output*, so failing safe means **not leaking**.

## The mediation contract

The policy engine ships the mechanism and stays engine-neutral — it never knows
how to rewrite, only when to. The transform is a caller-supplied hook, mirroring
how `welfare.Dispatcher` injects the model session so `welfare` never imports the
runner:

```go
// Mediator transforms a violating span into its replacement. It is supplied by
// the deployment (WrapResolverMediated / ServeConfig.PolicyMediator).
type Mediator func(ctx context.Context, ruleIndex int, span string) (string, error)
```

- `ruleIndex` — the 0-based rule that fired, so one mediator can route different
  rewrite rules to different transforms.
- `span` — the exact matched bytes. This is the *only* place the matched content
  crosses the engine boundary; it never enters the audit trail.
- return `(text, nil)` — emit `text` in place of the span.
- return `("", _)` or `(_, err)` — treated as failure → degrade to redact.

The mediator is deployment-*supplied* but **not trusted with its output**: it may
be the model itself, so its returned text is re-enforced once before emission
(see **Re-enforcement of mediator output** below).

## Streaming behaviour — the buffering boundary

> A rewrite cannot be emitted until the mediated span **closes**.

A span closes when its match `end` offset is known, which requires every byte of
the span to have arrived. G1's streaming enforcer already guarantees this: a
match is only *settled* (acted upon) at a scan position `i < holdFrom`, and

- a **term** match settles only when its full length is present
  (`longestMatchAt` requires `i+len(term) <= n`); a term's length ≤ `maxTermLen`
  ≤ `HoldBack()`, and
- a **pattern** match at `i < holdFrom` has `i + window ≤ n` (because
  `holdFrom = n − (maxWindow−1)` and `window ≤ maxWindow`), so the regexp always
  sees its complete, unclamped window.

Therefore **a settled match's span is always fully present**, and its length is
bounded by the same reach (`HoldBack()`) G1 already withholds. **G2 introduces no
new buffering bound.** The mediator is always called with a complete span.

Consequence for ordering: bytes before an *incomplete* span are also held back
(the enforcer settles left-to-right and stops at the first unsettled position),
so the mediator output is emitted in stream order — never ahead of text that
preceded the span, never behind text that followed it.

Consequence for latency: the enforcer calls the mediator **synchronously** from
`Feed`/`Close` at the settle point, so the stream stalls at a rewrite hit for as
long as the mediator takes — bounded by `mediate_timeout_ms`. Non-matching text
before and after the span streams at G1 speed; only the hit pays the stall.

## Re-enforcement of mediator output (untrusted mediator)

The mediator may be the model itself, so its output is **re-enforced once** before
emission — a single, **non-recursive** enforcement pass (`Policy.rescanMediated`):

- The mediator output is a **complete string** (the mediator returned it whole),
  so there is no streaming hold-back — it is scanned as a **closed buffer**.
- **Every** residual policy hit in it — redact, refuse, *or* rewrite — degrades to
  **redact**: the span is replaced by the rule's redact text (`residualReplacement`
  — a refuse rule, which carries no replacement, uses the default redaction).
- The mediator is **never called again**: no recursion, no second mediator call. A
  rewrite hit inside mediator output is redacted, not re-mediated.
- A residual **refuse** hit degrades to redact and **does not stop the stream** —
  the stream-level refuse decision belongs to the *original* scan over the model's
  own output, never to a re-scan of mediator text.
- The deployment-configured redact **fallback** (used on a mediator failure) is
  *not* re-scanned — only the untrusted mediator's own output is.

Any residual hit found here marks the rewrite's audit `Event` **degraded** (below).
The clean streaming path is untouched and stays 0-alloc; the re-scan allocates
only on the mediated path, which already allocates for the mediator call.

## Failure behaviour

| condition                                       | outcome                          |
|-------------------------------------------------|----------------------------------|
| mediator returns clean `(text, nil)`, `text!=""`| emit `text` (re-enforced, no hit)|
| mediator output has a residual policy hit       | redact the hit; **degraded**     |
| mediator returns an error                       | degrade to redact (`replacement`)|
| mediator returns `("", nil)`                    | degrade to redact (empty = fail) |
| mediator exceeds `mediate_timeout_ms`           | degrade to redact; stall ends    |
| rewrite policy loaded, **no mediator wired**    | **boot-fatal** (config error)    |

The timeout is owned by the enforcer, not delegated to the mediator's own
discipline: each mediator call runs under `context.WithTimeout` and the settle
path selects on the result vs. the deadline, so a mediator that ignores
cancellation still degrades and the stream still advances (the abandoned call
delivers into a buffered channel and is discarded). This makes "timeouts degrade
to redact" a guarantee of the layer, not a hope about the hook.

A *missing* mediator is a configuration error, not a runtime failure: a policy
that declares rewrite rules but is wired without a mediator refuses to serve at
boot, mirroring G1's "refusing to serve unguarded" contract for a bad regexp.

## Config surface

Policy file — a new action plus one optional knob:

```json
{
  "window": 256,
  "mediate_timeout_ms": 5000,
  "rules": [
    {"match": "term", "value": "PROJECT-X", "action": "rewrite", "replacement": "[redacted]"}
  ]
}
```

- `action: "rewrite"` — validated exactly like `redact`: `replacement` is
  optional (default `[redacted]`, used only on degrade) and a `message` is
  rejected. `window` remains pattern-only.
- `mediate_timeout_ms` — per-span mediator deadline; default
  `DefaultMediateTimeout` (5s), range `1..MaxMediateTimeoutMS` (60000),
  range-checked at load like `window`. Harmless on a policy with no rewrite
  rules.

Wiring — the mediator is an injected seam, like `ServeConfig.Loader`:

- `policy.WrapResolverMediated(inner, pol, log, mediator)` — the G2 wrapper.
- `policy.WrapResolver(inner, pol, log)` — the G1 wrapper, unchanged; valid for
  any policy with no rewrite rules.
- `ServeConfig.PolicyMediator` — nil for redact/refuse-only deployments. If the
  loaded policy `NeedsMediator()` and this is nil, `RunServe` is boot-fatal.

## Hot path

G2 adds two fields to `Enforcer` (`mediate`, `ctx`) and a rewrite branch to the
settle path. The clean streaming path — no match — is byte-for-byte the G1 path
and stays 0-alloc; a mediator goroutine is spawned only when a rewrite hit
settles, never per chunk. `BenchmarkPolicy_MediatingEnforcer_NoMatch` pins the
0-alloc clean path for a mediating enforcer.

## Resolved hardening (#378 follow-ups)

1. **Mediator output IS re-enforced.** Resolved: the stricter threat model won —
   the mediator is treated as untrusted (it may be the model itself), and its
   output passes one non-recursive enforcement pass before emission. See
   **Re-enforcement of mediator output** above.
2. **Degrades are audited distinctly.** Resolved: `Event` now carries a
   content-free `Degraded bool`, set when a rewrite fell back to redact (mediator
   error/timeout/empty/missing, or a residual hit redacted during re-enforcement).
   The serving audit line reads `rule #N rewrite degraded on output` in that case
   (vs `… enforced …`), never leaking the matched content.

## Open questions (for orchestrator review before merge)

1. **No default mediator in `cli/`.** The seam is wired and boot-fatal, but
   the CLI does not yet supply a mediator, so a rewrite policy is boot-fatal from
   `lem serve`. A model-reword mediator is a follow-up.
2. **Empty mediator result = failure.** A mediator that wants to delete a span
   must return a non-empty sentinel (e.g. a space or the replacement); `""` is
   treated as a failure and degrades to redact.
