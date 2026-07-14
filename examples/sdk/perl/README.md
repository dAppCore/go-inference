# Use Gemma 4 from Perl — local OpenAI-compatible API (lem / go-inference)

sdk/perl — gemma4 via the generated Perl client (`sdk-config/perl.yaml`,
`generatorName: perl`).

```bash
task sdk                                        # once: generate build/sdk/perl (needs Java; see repo root README)
cd examples/sdk/perl
cpanm -l local --notest Log::Any URI::Query     # once: the two CPAN deps system Perl doesn't ship (see Friction)
perl main.pl                                    # against a running lem serve (LEM_BASE_URL overrides)
```

The client is `LemSDK`, generated from the OpenAPI spec — Class::Accessor
model objects, LWP::UserAgent transport, typed request/response shapes
(`choices[].message.{content,thought}`), no hand-written HTTP.
`main.pl` goes past hello-world:

1. lists the served models (`GET /v1/models`);
2. holds a two-turn conversation, resending turn 1's history on turn 2 to
   prove the model actually used it rather than re-guessing;
3. runs the same prompt with the thinking channel on (the default) and off
   (`chat_template_kwargs => { enable_thinking => JSON::false }`);
4. prints prompt/completion token usage after every call;
5. turns a connection refusal into "start a serve first: `lem serve`"
   instead of a raw LWP stack trace.

## Friction

This lane exists partly to measure Perl/CPAN pain honestly. What actually
happened, on macOS system Perl 5.34.1:

- **cpanminus wasn't installed** — `brew install cpanminus` (few seconds,
  no sudo).
- **Of the 11 modules in the generated `cpanfile`, only 2 were missing**:
  `Log::Any` and `URI::Query`. System Perl already ships `Moose`, `JSON`,
  `LWP::UserAgent`, `Class::Accessor`, `Class::Data::Inheritable`,
  `Module::Runtime`, `Module::Find`, `DateTime`, `Test::Exception` and the
  `HTTP::*` stack — this was less painful than a fresh-CPAN-environment
  worst case would suggest. `cpanm -l local --notest Log::Any URI::Query`
  installed both (pure-Perl, no compiled C, ~2s including the network
  fetch) into `examples/sdk/perl/local/lib/perl5`, which `main.pl` adds to
  `@INC` itself via `FindBin` — no global `PERL5LIB` or `~/perl5` needed,
  and no sudo anywhere in this lane.
- **The generated client's `croak` on a non-2xx/connect failure carries a
  full LWP stack trace.** `main.pl` wraps every SDK call and pattern-matches
  the connection-refused case so the user sees an instruction, not a trace.

**A real bug this lane surfaced (not a Perl problem — flagging, not
fixing, here):** the exported OpenAPI spec declares `thought` nested under
`choices[].message.thought`, and every generated client (Perl included)
deserialises strictly against that declared shape. But the live server
(`lem serve`, gemma-4-E2B, this session) actually serialises `thought` as
a **top-level sibling** of `choices`/`usage`, e.g.:

```json
{"choices":[{"message":{"role":"assistant","content":"..."}}],
 "usage":{...},
 "thought":"...the actual reasoning text..."}
```

Because the field never appears where the schema says it will, every
generated client — Go, Python, Rust, TypeScript, Perl, all five — silently
drops it: `choices[0].message.thought` is always empty, in every language,
regardless of `enable_thinking`. The four pre-existing examples never hit
this because each disables thinking for its one demo call; this lane's
requirement to exercise thinking on *and* off is what surfaced it. `thought`
below and in `main.pl`'s output is genuinely `(no thought field returned)`
on both requests — that's the honest typed-client result, not a bug in this
example. Worth a go-inference fix (move the field in the response
serialiser, or in the spec, so they agree) as a follow-up, not attempted
here — out of scope for a worktree scoped to the Perl SDK example.
- **Also worth knowing (not a bug, an operational trap):** with thinking
  left on and a modest `max_tokens`, the model can spend the *entire*
  budget on the (invisible-to-the-typed-client) reasoning pass before
  writing any reply — a plain conversational prompt came back with an
  empty `content` and `completion_tokens` pinned at the budget ceiling
  until `enable_thinking` was turned off for the memory demo. Budget
  accordingly.

None of the above came close to the 20-minute timebox — cpanminus install,
the two missing deps, and a clean live run took a few minutes end to end.
