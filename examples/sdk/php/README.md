# Use Gemma 4 from PHP — local OpenAI-compatible API (lem / go-inference)

```bash
task sdk                       # once: generate build/sdk/php from lem's OpenAPI spec
cd examples/sdk/php
composer install               # once: installs the path-repo SDK + Guzzle
php main.php                   # against a running lem serve (LEM_BASE_URL overrides)
```

The client is `lethean/lem-sdk` (php-nextgen generator, Guzzle transport),
generated from the OpenAPI spec — typed request/response models
(`choices[].message.{content,thought}`), no hand-written HTTP.

This example goes past hello-world:

1. lists the models the driver is currently serving,
2. runs a two-turn conversation where turn 2 proves the model remembered
   turn 1 (the client resends the full history — the server is stateless),
3. demonstrates the `thought` field twice: once with defaults (thinking on)
   and once with `chat_template_kwargs: {enable_thinking: false}`,
4. prints prompt/completion token usage after every call,
5. gives a clear "start `lem serve`" message on connection failure instead of
   a raw Guzzle stack trace.

## Friction

**`php` vs `php-nextgen`: picked `php-nextgen`.** Both generators currently
pin the same `"php": "^8.1"` floor in their `composer.mustache` (checked the
generator source directly — the "nextgen = newer PHP floor" assumption is
wrong today), so PHP-version support isn't the differentiator it sounds
like. The real differences: `php-nextgen` is the OpenAPI Generator project's
declared forward replacement for `php` (BETA vs STABLE, but that's a
project-maturity label, not a code-quality one for this size of API surface),
and it drops `guzzlehttp/psr7` v1 support entirely (`^2.0` only) while pinning
a newer Guzzle floor (`^7.4.5` vs `^7.3`). Generation was clean either way —
no errors, no manual template overrides needed. Given the task asked for
"suits modern PHP (8.x) best," `php-nextgen` is the more honest answer even
though the version-floor argument doesn't hold up under inspection.

**Real spec bug, not a client bug: `thought` isn't where the spec says it
is.** `build/sdk/openapi.json` declares `thought` nested under
`choices[0].message` (`go/engine/driver/inference.go:138`), which is where
the generated PHP model exposes it, and where the sibling
go/python/rust/typescript examples all read it. But the live wire response
puts `thought` at the **top level** of the response body —
`go/serving/provider/openai/openai.go:133`,
`ChatCompletionResponse.Thought *string \`json:"thought,omitempty"\``, sibling
to `Choices`, not inside `ChatMessage`. Confirmed against the running serve:

```json
{
  "choices": [{"message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"}],
  "usage": {...},
  "thought": "\nThinking Process:\n1. ..."
}
```

The generated client's typed `getThought()` on `choices[0].message` reliably
returns `null` even with thinking on — it isn't wrong, the spec told it to
look in the wrong place. `main.php` proves the model actually reasoned by
decoding the raw response body alongside the typed call for that one
request; every other call in the example uses only the typed client. Fix
belongs in the OpenAPI exporter (`go/engine/driver/inference.go`), not here.

**The generated client swallows `ConnectException` before your code ever
sees it.** `InferenceApi`'s HTTP methods explicitly catch
`GuzzleHttp\Exception\ConnectException` and rewrap it as the SDK's own
`ApiException` (`build/sdk/php/src/Api/InferenceApi.php`) — catching
`ConnectException` in application code, as the Go/Python/Rust examples'
error-handling idiom would suggest, never fires. A connection failure is
instead an `ApiException` with `getCode() === 0` (curl errors aren't HTTP
statuses) and `getResponseHeaders() === null` (no HTTP response was ever
received) — that's the signal `main.php` checks to print the "start `lem
serve`" message instead of a raw exception.

**`chat_template_kwargs` wants a PHP `object`, not an array.** The generated
setter is `setChatTemplateKwargs(?object $chat_template_kwargs)` — passing
the natural `['enable_thinking' => false]` array is a `TypeError`. Needs an
explicit `(object)` cast, or a `new stdClass` with dynamic properties.

**`composer.lock` pins a content hash of the gitignored `build/sdk/php`
directory — committed anyway, but only after checking it's stable.**
Composer's `path` repository type records a `dist.reference` that's a hash
of the dependency's file contents, unlike Cargo.lock/package-lock.json's
local path/file deps (no content hash at all — version + declared deps
only). That sounded like guaranteed lock churn on every regeneration. Tested
it directly: regenerated the OpenAPI spec and the PHP SDK from scratch twice
and re-ran `composer update` — the reference hash
(`a3a99089251ba32f2b4853f4d4f96cf919f486b6`) came back byte-identical both
times (`hideGenerationTimestamp` defaults to `true`, so there's no embedded
"generated at" timestamp to cause drift). Committed `composer.lock`, matching
the Rust/TypeScript examples' convention — `vendor/` is gitignored, the lock
file isn't.

**A "thinking" call can burn its whole `max_tokens` budget on the reasoning
trace and return empty `content`.** With defaults (thinking on) and
`max_tokens: 256`, "what is 17 times 24" routinely finishes with
`finish_reason: "length"` and `content: ""` — all 256 tokens went into
`thought`, none into the final answer. That's real model/budget behaviour,
not a bug in the example; raise `max_tokens` (1024 reliably leaves room for
both) if you want to see a completed answer alongside the thought.

## Actually running it

```
$ php main.php
serving: 42f62737af7a9fd8c1d55d79666c1a217be4e2e2
turn 1: Got it! Teal is your favorite color. 💚
usage: 18 prompt + 12 completion tokens
turn 2 (should recall teal): Teal
usage: 49 prompt + 3 completion tokens
thought via typed client: (null -- see Friction, it is really top-level)
thought via raw decode:
Thinking Process:
1.  **Identify the core request:** The user wants to know the product of 17 and 24 (17 * 24).
...
answer:
usage: 30 prompt + 256 completion tokens
thought (enable_thinking=false): (none, as expected)
answer: To find the result of 17 times 24, you can use a couple of methods...
usage: 23 prompt + 256 completion tokens
```

(Truncated for the README; the full run is real, captured against a live
`lem serve` on `localhost:36911` serving gemma-4-E2B. `LEM_BASE_URL` against
a closed port produces `Could not reach http://localhost:19999 -- start a
serve first: lem serve --model <path>` and exit code 1, not a stack trace.)
