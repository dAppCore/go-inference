# Use Gemma 4 from Kotlin — local OpenAI-compatible API (lem / go-inference)

`lem spec` exports go-inference's HTTP surface as an OpenAPI 3.1 document, and
[openapi-generator](https://openapi-generator.tech/docs/generators/kotlin)
turns that into a typed Kotlin (`jvm-okhttp4` + Moshi) client. This example
goes past hello-world: it lists the served models, runs a two-turn
conversation that proves the model remembers turn 1, and probes the thinking
channel — including the one place the generated client actually falls short
(see Friction).

```bash
task sdk                       # once, from the repo root: generate build/sdk/kotlin
```

## Build + run

The generated client needs Java to compile with (openapi-generator itself
also needs a JRE — see the repo-root Taskfile). Two Java versions are in
play here; see Friction #1 for why:

```bash
export JAVA_HOME="/opt/homebrew/opt/openjdk@21"   # or wherever your JDK 21 lives
export PATH="$JAVA_HOME/bin:$PATH"

cd examples/sdk/kotlin
gradle run                     # against a running lem serve (LEM_BASE_URL overrides)
```

This is a **Gradle composite build**: `settings.gradle.kts` points
`includeBuild("../../../build/sdk/kotlin")` at the generated (gitignored)
client and substitutes it for the `re.dappco:lem-sdk` coordinate declared in
this project's `build.gradle.kts` — no `publishToMavenLocal` step, no
published version to keep in sync. A published SDK would just be a normal
`implementation("re.dappco:lem-sdk:x.y.z")` dependency and this file would
lose its `settings.gradle.kts`.

## What it does

1. **List models** — `GET /v1/models` via `InferenceApi.getV1Models()`.
2. **Two-turn conversation** — tells the model a favourite number, resends
   the full history (including the model's own turn-1 reply) on turn 2, and
   asks for the number back. The transcript and a pass/fail line proving
   memory are printed.
3. **Thinking demo** — one call with defaults, one with
   `chatTemplateKwargs = mapOf("enable_thinking" to false)`. Both print the
   typed `choices[0].message.thought` field — and both print `null`, because
   of Friction #2 below. The default-request branch then makes one **raw**
   HTTP call (reusing the SDK's own `Serializer.moshi` instance, not the
   typed client) to show the actual thought text the server produced.
4. **Usage tokens** — printed after every chat call.
5. **Connection-refused handling** — the whole flow runs inside one
   `try/catch (ConnectException)` that prints a "start `lem serve`" message
   and exits 1.

## Friction

Friction is the product — here's everything that didn't just work:

1. **JDK 26 (this machine's default) fails the generated build outright.**
   The generated `build.gradle` applies the classic `apply plugin: 'kotlin'`
   (Kotlin 2.2.20 bundled via `buildscript`), and Gradle's Java plugin
   defaults `compileJava`'s target to the running JVM (26). Kotlin 2.2.20
   doesn't support a JVM 26 target and silently falls back to 24, and Gradle
   then refuses the mismatch outright:
   ```
   ⛔ Inconsistent JVM Target Compatibility Between Java and Kotlin Tasks
   Inconsistent JVM-target compatibility detected for tasks 'compileJava' (26) and 'compileKotlin' (24).
   ```
   Fix: build with a JDK the Kotlin toolchain actually supports —
   `openjdk@21` in this repo's brew environment. Neither `sdk-config/kotlin.yaml`
   nor this example's `build.gradle.kts` sets a `jvmToolchain`, so it inherits
   whatever `JAVA_HOME` is active — worth a toolchain block if this trips
   people up in practice.

2. **The live `thought` field doesn't live where the OpenAPI spec says it
   does — this is a genuine spec/implementation mismatch, not a generator
   bug.** `go/engine/driver/inference.go`'s hand-built schema declares
   `thought` nested under `choices[].message.thought`, and every generated
   SDK (Kotlin included) models it there. But the actual server response
   (`go/serving/provider/openai/openai.go`'s `ChatCompletionResponse`) puts
   `Thought *string \`json:"thought,omitempty"\`` on the **top-level**
   response object, as a sibling of `choices` and `usage` — never inside the
   message. Confirmed straight off the wire:
   ```json
   {
     "choices": [{"message": {"role": "assistant", "content": "Hello!"}, ...}],
     "usage": {...},
     "thought": "\nThinking Process:\n\n1.  **Analyze the Request:** ..."
   }
   ```
   Consequence: `choices[0].message.thought` is **always null** through every
   generated SDK, for every language, thinking on or off — the field the
   spec models simply never appears on the wire. Worse, the *top-level*
   `thought` isn't modelled anywhere in the spec either, so Moshi (default
   `failOnUnknownProperties = false`) silently drops it — a typed client
   cannot recover the model's reasoning at all as things stand. This example
   demonstrates the gap honestly: it prints the typed field (`null`), then
   falls back to one raw OkHttp call + the SDK's own `Serializer.moshi` to
   show the real value exists. The four other language examples
   (`sdk/{go,python,rust,typescript}`) describe `choices[].message.thought`
   as "typed" in their READMEs but never actually print it — this is the
   first example in the set to exercise it, which is how the mismatch
   surfaced. **Fix belongs in `go/engine/driver/inference.go`'s spec schema
   (move `thought` to the top level) or in the server (nest it under
   message) — not in any generated client.**

3. **`gradle` gets silently rewritten to a non-executable `./gradlew`.**
   This machine's shell hook intercepts bare `gradle` invocations in a
   directory that has a `gradlew`, and the *generated* `build/sdk/kotlin`
   ships one at `644` (not executable) — so the rewritten invocation fails
   with `Permission denied`, not a Gradle error. Either `chmod +x gradlew`
   in the generated tree, or call the full path
   (`/opt/homebrew/bin/gradle`) to bypass the rewrite. This example's own
   directory has no `gradlew` committed (system Gradle 9.6.1 is assumed —
   see prerequisites), so it isn't affected, but `build/sdk/kotlin` is.

4. **The generated client hides its own transport from consumers.**
   `build/sdk/kotlin/build.gradle` declares `okhttp` and `moshi-kotlin` as
   `implementation` dependencies, not `api` — correct Gradle hygiene for a
   normal consumer, but this example needs raw `OkHttpClient`/`Request` (for
   the connection-refused path and generous timeouts) and the SDK's own
   `Serializer.moshi` (for Friction #2's workaround). Both had to be
   redeclared explicitly in this project's `build.gradle.kts`, version-pinned
   to match the generated `build.gradle` by hand — there's no single source
   of truth for those versions across the composite build boundary.

5. **`content` is `kotlin.Any?`, not `kotlin.String`.** The spec never
   declares a `type` for `messages[].content` (it's meant to accept a plain
   string or an array of typed content parts), so the Kotlin generator
   — like every other language here — falls back to an untyped field. Same
   shape as the Go/Python/Rust/TypeScript examples; not new, but worth
   repeating since it's the one place "typed client" doesn't quite hold.

6. **A connection-refused exit prints its message correctly, then Gradle
   also reports the *task* as failed.** `gradle run` treats the wrapped
   JVM's non-zero exit as a build failure and prints its own
   `BUILD FAILED`/`Process ... finished with non-zero exit value 1` banner
   underneath the actual error line. The real message is the one starting
   `cannot reach ... — start \`lem serve\` first` — don't mistake the
   Gradle banner for a build/compile problem.
