# Use Gemma 4 from Swift — local OpenAI-compatible API (lem / go-inference)

```bash
task sdk                          # once: generate build/sdk/swift (needs openapi-generator-cli + a JRE)
cd examples/sdk/swift
swift run                         # against a running lem serve (LEM_BASE_URL overrides)
```

The client is the generated `LemSDK` Swift package (swift6 generator,
URLSession + async/await), depended on by relative path via a `GeneratedSDK`
symlink (see Friction #1). Typed request/response models
(`choices[].message.{content,thought}`), no hand-written HTTP.

This demo goes past hello-world:

1. lists the models the driver has loaded (`GET /v1/models`);
2. a two-turn conversation that resends the full message history, so the
   second answer proves gemma4 actually used turn one's content rather than
   coincidentally guessing;
3. a "thinking" demo — one request with defaults, one with
   `chatTemplateKwargs: ["enable_thinking": false]` — reading the typed
   `thought` field both times (see Friction #3 for why it prints empty
   either way, and what the usage numbers show instead);
4. prompt + completion token usage printed after every call;
5. a plain "start `lem serve`" message on connection refusal, instead of a
   raw `URLError` dump.

## Friction

Honest rough edges hit building this — nothing below was worked around by
hiding it.

**1. SwiftPM local-path package identity collides on the directory name,
not the package name.** `sdk-config/swift.yaml` uses `useSPMFileStructure`,
so the generator writes a normal SwiftPM package to `build/sdk/swift/`
(`Package(name: "LemSDK", ...)`). This demo package also lives in a
directory literally named `swift/` (`examples/sdk/swift/`, matching the
other three languages' layout). SwiftPM derives a *local path* dependency's
graph identity from the directory's basename — not from the `name:` in its
manifest — so `.package(path: "../../../build/sdk/swift")` collides with
this package's own identity and fails with a genuinely confusing error:

```
error: 'swift': product 'LemSDK' required by package 'swift' target 'LemSwiftExample' not found in package 'swift'.
```

The (also-deprecated) `.package(name: "X", path: ...)` override does
*not* fix this — the name argument doesn't affect graph identity either.
The actual fix: add a symlink with a distinct name and point SwiftPM at
that instead — `examples/sdk/swift/GeneratedSDK -> ../../../build/sdk/swift`,
referenced as `.package(path: "GeneratedSDK")`. No content is duplicated;
the symlink is the only extra file this demo needed to commit.

**2. The `thought` field is a spec/runtime mismatch, not a Swift problem.**
`cli spec`'s exported OpenAPI document declares `thought` nested under
`choices[].message` (matching the other three languages' READMEs, which
all describe `choices[].message.{content,thought}`). The live server does
not put it there. A direct request against the running serve with no
`chat_template_kwargs` returns:

```json
{
  "choices": [{ "index": 0, "message": { "role": "assistant", "content": "Yes." }, "finish_reason": "stop" }],
  "usage": { "prompt_tokens": 26, "completion_tokens": 173, "total_tokens": 199 },
  "thought": "\nThinking Process:\n\n1.  **Analyze the Request:** ..."
}
```

`thought` is a sibling of `choices`, not a property of `message` — confirmed
against `go/serving/provider/openai/openai.go`'s `ChatCompletionResponse`
(`Thought *string` lives on the response struct, not on `ChatMessage`). The
generated Swift (and, by the same schema, the go/python/rust/typescript
clients) has no typed path to the real value at all — the root response
model has no `thought` property to bind it to. This demo still prints the
*typed* `choices[0].message.thought` exactly as asked, honestly, as
`(none returned)`, rather than reaching past the generated types to parse
raw JSON to make the demo look more finished than the SDK actually is.
The `rust/README.md` for this same example set already anticipated this
("a mis-declared spec fails deserialisation here first") — this is the
first of the five language demos to actually exercise the field and land
on it. Fixing `cli spec`'s handler-side schema declaration (or the
runtime's response shape) is a go-inference change, out of scope here.

The token usage is the honest substitute receipt: 173 completion tokens
for a two-word visible answer (defaults) vs 3 tokens for the same question
with `enable_thinking: false` — thinking is demonstrably happening, the
generated client just can't type its way to the text.

**3. `chat_template_kwargs` and message `content` are untyped in the spec**
(`type: object` / no schema at all), so the generator emits both as the
generic `JSONValue` enum rather than `[String: Bool]` or `String`. Every
call site wraps plain values — `JSONValue("...")`,
`JSONValue(["enable_thinking": JSONValue(false)])` — instead of using Swift
literals directly. Mechanical, not a bug, but worth knowing before writing
the first line of client code.

**4. `swift6` needed `responseAs: AsyncAwait` set explicitly** — its
default (`library: urlsession`) generates completion-handler APIs, not
`async throws`. Setting it in `sdk-config/swift.yaml` was a deliberate,
one-line pick, not a fallback.
