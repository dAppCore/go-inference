# Use Gemma 4 from Java — local OpenAI-compatible API (lem / go-inference)

A Maven project using the GENERATED Java client (`sdk-config/java.yaml` →
`build/sdk/java`, okhttp-gson) to drive a local `lem serve` — the OpenAI
standard doing the client work, no hand-written HTTP.

Past hello-world, `Main.java`:

1. lists the served models (`GET /v1/models`, typed);
2. runs a two-turn conversation where turn 2 resends turn 1 in full and
   answers from history — proof the model remembers, not a trick;
3. probes the reasoning channel: one call with the model's own defaults, one
   with `chat_template_kwargs: {"enable_thinking": false}` — prints the
   typed `choices[0].message.thought` field the spec declares AND the raw
   top-level `thought` this server actually sends (see Friction);
4. prints prompt/completion token usage after every call;
5. tells you plainly to `lem serve` first if the connection is refused,
   instead of a stack trace.

## Build + run

```bash
# once, from the repo root: generate every SDK, including this one
task sdk                       # -> build/sdk/openapi.json, build/sdk/{go,java,python,rust,typescript}

# once: the generated client isn't published anywhere — install it into the
# local Maven repo so this example can depend on its coordinates like any
# other artefact
export JAVA_HOME="$(brew --prefix openjdk)"    # brew's openjdk is keg-only
export PATH="$JAVA_HOME/bin:$PATH"
mvn -q -f build/sdk/java install -DskipTests   # -> ~/.m2/repository/re/dappco/lem-sdk/0.1.0

# run the example against a running lem serve (LEM_BASE_URL overrides)
cd examples/sdk/java
mvn -q exec:java
```

`pom.xml` depends on `re.dappco:lem-sdk:0.1.0` as a normal Maven coordinate
— nothing path- or system-scoped. Re-run the `mvn install` step whenever the
spec changes and you regenerate.

## Example output

Against a live `gemma-4-E2B` serve:

```
serving: 42f62737af7a9fd8c1d55d79666c1a217be4e2e2
turn 1: Okay, I've noted that! **Teal** is your favorite color. 😊

I'll remember that for when we talk about it later.
usage: 20 prompt + 33 completion tokens
turn 2 (proves memory): Your favorite color is **teal**.
usage: 68 prompt + 8 completion tokens
thinking (defaults) -- typed message.thought: (absent)
thinking (defaults) -- raw top-level thought: 1.  **Analyze the Request:** The user wants a single sentence explaining the importance of "local inference." ...
thinking (defaults) -- answer: Local inference matters because it enables low-latency, private, and reliable processing by keeping data on the device rather than sending it to remote servers.
usage: 26 prompt + 387 completion tokens
thinking (enable_thinking=false) -- typed message.thought: (absent)
thinking (enable_thinking=false) -- raw top-level thought: (absent)
thinking (enable_thinking=false) -- answer: Local inference matters because it allows for rapid, efficient decision-making by processing data directly on the device rather than relying on distant cloud servers.
usage: 19 prompt + 29 completion tokens
```

Connection refused (no serve running):

```
$ LEM_BASE_URL=http://localhost:1 mvn -q exec:java
Could not reach http://localhost:1 -- start a serve first: lem serve --model <path-to-model>
```
(exits 1, no stack trace)

## Friction

Java was the roughest of the five SDK lanes so far — three real findings:

- **`thought` breaks the typed client outright, not just silently.** The
  exported spec nests `thought` under `choices[0].message` (matching the
  other four language clients' assumption), but this serve emits it as a
  **top-level sibling of `choices`**. The Java generator's default,
  `disallowAdditionalPropertiesIfNotPresent: true`, makes the generated
  model's Gson adapter *throw* (`JsonSyntaxException`: "The field `thought`
  ... is not defined") the moment a response carries a field the schema
  didn't declare on that object — every response where the model actually
  reasons. Go/Python/Rust/TS use looser deserialisation and just silently
  drop the field, which is why this didn't surface there; Java's stricter
  default turned a spec/implementation mismatch into a hard failure. Fixed
  in `sdk-config/java.yaml` by setting
  `disallowAdditionalPropertiesIfNotPresent: false` — which is also the
  option the generator's own docs call the OAS/JSON-Schema-*compliant*
  behaviour, not a workaround bolted on. The deeper bug is still upstream:
  go-inference's OpenAPI export and its actual HTTP response disagree on
  where `thought` lives, and that's worth a fix in the spec/handler, not
  just in every client's tolerance setting.
- **Even fixed, the typed field is still unreachable.** `disallowAdditionalPropertiesIfNotPresent: false`
  stops the crash but doesn't relocate the field — `choices[0].message.getThought()`
  is `null` on every real response from this serve, because the model class
  matches the spec's (wrong) nesting. `Main.chat()` executes the SDK's own
  `postV1ChatCompletionsCall(...)` directly and parses the response body
  twice from the one string — once through the SDK's typed Gson (proves the
  spec's shape is unreachable), once as a raw `JsonObject` (recovers the
  actual top-level value) — one network round trip either way. This is an
  escape hatch on top of the generated client, not a replacement for it.
- **`chat_template_kwargs.enable_thinking` did not read as fixed on/off during ad-hoc exploration.**
  Early manual `curl` probing (before this example existed) saw a request
  with `enable_thinking: false` still return a full `thought` block; the
  shipped example's own run above shows the flag behaving as documented
  (defaults → reasons, `false` → no `thought`, roughly 13x fewer completion
  tokens). Whether that's request-content-dependent, budget-dependent, or
  genuinely non-deterministic wasn't pinned down here — flagging it rather
  than asserting either way.
- No dependency or `openapi-generator` gotchas beyond the above — okhttp-gson,
  the fluent builder setters, and Maven's local-install-then-depend flow all
  worked exactly as documented once the config above was in place.
