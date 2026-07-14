# Use Gemma 4 from Ruby — local OpenAI-compatible API (lem / go-inference)

```bash
task sdk                                       # once: generate build/sdk/ruby
cd examples/sdk/ruby
bundle install --path vendor/bundle            # system ruby needs a local gem path, see Friction
bundle exec ruby main.rb                       # against a running lem serve (LEM_BASE_URL overrides)
```

The client is the `lem_sdk` gem (Faraday), generated from the OpenAPI
spec — typed request builders, typed
`choices[].message.{content,thought}` response, no hand-written HTTP.

`main.rb` goes past hello-world:

1. lists the served models (`GET /v1/models`);
2. a two-turn conversation that proves the model remembers turn 1 (turn 2
   resends the full history and asks for the fact back);
3. a thinking-channel demo: one call with defaults, one with
   `chat_template_kwargs: {"enable_thinking" => false}`;
4. prompt/completion token usage printed after every call;
5. a clear "start `lem serve`" message on connection refused instead of a
   raw Faraday backtrace.

## Friction

Honest rough edges hit while building this, worst first.

**The `thought` field is unreachable through the typed client — a spec bug,
not a Ruby problem.** The OpenAPI spec declares `thought` as a property of
`choices[].message`, so the generated `PostV1ChatCompletions200ResponseChoicesInnerMessage`
model has a `thought` accessor. But the live server puts `thought` at the
**top level** of the response object, as a sibling of `choices` — outside
any schema the response model declares:

```json
{
  "choices": [{"message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"}],
  "usage": {...},
  "thought": "1. Analyze the Request: ..."
}
```

The strict generated deserialiser only reads keys it has a schema property
for, so this top-level `thought` is silently dropped — `choices[0].message.thought`
is always `nil`, live, regardless of what the model actually reasoned.
Confirmed with a raw `curl` against the same endpoint (see the response
above) — this isn't a Ruby-client quirk, it would misfire identically in
every generated language here; the other four examples simply never read
`.thought` so they never noticed. Fix belongs in `cli spec` (move `thought`
to the top-level response schema, matching the wire format) or in the
server (nest it under `message` to match the spec) — not in this example.

**`chat_template_kwargs: {"enable_thinking": false}` did not reliably
suppress reasoning for gemma-4-E2B on this serve.** Across repeated calls
with the same prompt, the model sometimes emitted ~300-390 tokens of visible
chain-of-thought before its `content` **with the flag set to `false`**, and
sometimes answered directly with the flag left at its default (`true`).
`main.rb` still sends both variants as asked, and both print whatever
`thought` came back (always `nil` — see above) — but don't read anything
into the flag having "worked" just because a given run looked short.

**A reasoning pass can consume the whole token budget before any `content`
is written**, leaving `content: ""` and `finish_reason: "length"` — not an
error, just an exhausted budget. `max_tokens: 64` (fine for the go/python/ts
examples' single non-reasoning question) truncated every reply here. Budgets
below were picked from observed worst cases (memory turns: 400; the "why
does X matter" thinking-demo question: 700) — `main.rb` prints the
`finish_reason` whenever `content` comes back empty so a truncation is never
mistaken for the model forgetting or the API being broken.

**System Ruby is 2.6.10 (EOL, ships with macOS) — the generator's default
`gemRequiredRubyVersion` (`>= 2.7`) blocks Bundler outright** on an
otherwise-fine machine. `sdk-config/ruby.yaml` pins `gemRequiredRubyVersion: ">= 2.6"`
to match; the actual runtime dependency (Faraday) still resolves fine on
2.6.10 once that gate is out of the way. If your `ruby -v` is >= 2.7 this
line is a no-op.

**Bundler 1.17.2 (the version system Ruby ships) doesn't understand
Bundler-2-style config** — `bundle config set --local path vendor/bundle`
does nothing on this version and Bundler falls through to installing into
`/Library/Ruby/Gems/2.6.0`, which then demands `sudo`. Use the Bundler-1
syntax instead: `bundle install --path vendor/bundle` (shown above). No
`sudo`, no system gems touched.

**Generator's default `library` is `faraday`** (over `typhoeus` and
`httpx`) — pure Ruby, no libcurl FFI binding to compile against an old
system Ruby's headers, which is exactly the kind of native-extension
friction this environment already has enough of.
