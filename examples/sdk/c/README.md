# Use Gemma 4 from C — local OpenAI-compatible API (lem / go-inference)

A CMake + libcurl client for `lem serve`, generated from go-inference's
OpenAPI 3.1 spec by [openapi-generator](https://openapi-generator.tech/docs/generators/c)'s
`c` generator, then patched (see [Friction](#friction) — the generator's
output does not compile as-is). Lists the served model, holds a two-turn
conversation that proves the model remembers turn 1, and probes the
thinking channel two ways.

## Build

```bash
# 1. Generate the client (from the repo root; regenerates every language)
task sdk

# 2. Patch the C output — required every time it's regenerated, see Friction
bash examples/sdk/c/postgen-fix.sh build/sdk/c

# 3. Configure + build this example (links the generated client as a CMake subdirectory)
cd examples/sdk/c
cmake -S . -B cmake-build -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build -j
```

Needs `clang`/`cc`, `cmake`, and libcurl (macOS ships it; `brew install curl`
if headers are missing).

## Run

```bash
# once: start a serve (from the repo root)
task build && ./bin/lem serve --model ~/.cache/huggingface/hub/models--mlx-community--gemma-4-E2B-it-qat-4bit/snapshots/<hash>

# then, from examples/sdk/c
./cmake-build/lem_c_example              # LEM_BASE_URL overrides the default http://localhost:36911
```

If `lem serve` isn't running, the client fails fast with a clear message
instead of libcurl's raw connection-refused error.

## Friction

The `c` generator is by far the roughest of the five languages this repo
targets (go/python/rust/typescript all "just work" — see `../README.md`).
Every item below was found by actually building and running this example,
not by reading docs.

### 1. The generated library does not compile — `any_type_t` is a stub

Any schema field with **no declared type** — lem's spec has three:
`messages[].content` (string, or an array of typed parts), the freeform
`chat_template_kwargs`, and the generic envelope's `error.details` /
`response.data` — is modelled by the `c` generator as `any_type_t`. But the
generator ships `model/any_type.h` as an 81-byte placeholder:

```c
/*
 * any_type.h
 *
 * A placeholder for now, this type isn't really needed.
 */
```

No typedef, no header guard, no `.c` file. Worse, the three affected model
files (`error.c`, `response.c`,
`post_v1_chat_completions_request_messages_inner.c`) call the type's helper
functions with an **empty prefix** instead of `any_type_`:

```c
_t *content_local_nonprim = NULL;
content_local_nonprim = _parseFromJSON(content); //custom
...
_free(post_v1_chat_completions_request_messages_inner->content);
```

`_t`, `_free`, `_parseFromJSON`, `_convertToJSON` are not valid symbols —
`cmake --build` fails with "unknown type name 'any_type_t'" followed by a
cascade of "call to undeclared function" errors. This is a template bug in
openapi-generator 7.22.0's C generator, confirmed by reading the emitted
source (it isn't a spec problem — the struct field declarations correctly
say `any_type_t`; only the helper-function call sites lost the prefix).

**Fix applied:** [`postgen-fix.sh`](postgen-fix.sh) gives `any_type_t` a
real definition (a `cJSON*` wrapper, matching the pattern the generator
already uses for the plain `object_t` type) and rewrites the three broken
call sites to use it. It's idempotent and must be re-run after every
`task sdk` regeneration — this is not committed generated code, it's a
patch script committed alongside the example. Without it, the C SDK cannot
be built at all, for *any* endpoint that touches a freeform field —
including the one every chat completion needs (`content`).

### 2. `chat_template_kwargs` has no typed setters

`object_t` (used for `chat_template_kwargs`, and the same shape `any_type_t`
should have had) is not a struct with fields — it's this:

```c
typedef struct object_t {
    void *temporary;
} object_t;
```

`object_convertToJSON` calls `cJSON_Parse(object->temporary)` — i.e.
`temporary` is expected to already be a **raw JSON string**. There is no
builder API. To send `{"enable_thinking": false}` you write:

```c
object_t *kwargs = object_create();
kwargs->temporary = strdup("{\"enable_thinking\": false}");
```

No validation, no escaping help, and a typo in the hand-written JSON fails
silently at `cJSON_Parse` (returns an empty object, not an error). This is
exactly the "cannot express the kwargs object cleanly" case anticipated
before this example was built — confirmed.

### 3. `thought` is nested in the spec; the live server puts it at the top level

The exported spec declares:

```
ChatCompletionResponse.choices[].message.thought   (string)
```

and the generated model matches that — `post_v1_chat_completions_200_response_choices_inner_message_t`
has a `thought` field. But the live server's actual response shape (confirmed
by curling `localhost:36911/v1/chat/completions` directly, bypassing every
generated client) is:

```json
{
  "id": "chatcmpl-...", "choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "length"}],
  "usage": {...},
  "thought": "\n*   **Question:** Is 17 prime?\n*   **Definition..."
}
```

`thought` is a **sibling of `choices`**, not nested under `message` — matching
`go/serving/provider/openai/openai.go`'s `ChatCompletionResponse.Thought
*string` field (`ChatMessage` itself has no `Thought` field at all). The
spec — and therefore every generated client in this repo, C included — is
looking in the wrong place. `choice->message->thought` is `NULL` on every
call this example makes, regardless of `chat_template_kwargs`; this isn't a
client bug, the data legitimately isn't at that path. **This is a spec/
implementation drift bug in go-inference's `cli spec` exporter, not
specific to the C generator** — it affects the go/python/rust/typescript
examples identically (worth a follow-up outside this task's scope).

### 4. The library's CMake target name is undiscoverable from `sdk-config/c.yaml`

`additionalProperties.projectName` (the option this example's
`sdk-config/c.yaml` sets, matching the other four configs' `packageName`/
`projectName`) is **not** what names the CMake target. The `c` generator
derives `pkgName` from the OpenAPI document's `info.title` instead —
`"Lethean lem API"` becomes `lethean_lem_api`. Changing `c.yaml`'s
`projectName` has no visible effect; renaming lem's spec `info.title` would
silently rename the library out from under every consumer's
`target_link_libraries`. This example's `CMakeLists.txt` hardcodes
`lethean_lem_api` and documents why in a comment.

### 5. Every `_create()` constructor is marked `deprecated`, with no alternative

All object constructors (`post_v1_chat_completions_request_messages_inner_create`,
etc.) carry `__attribute__((deprecated))`, but there is no non-deprecated
constructor — the only alternative is hand-building the struct (malloc +
field assignment + manually setting `_library_owned = 1`), which this
example does for exactly that reason. Using the SDK "as intended" emits
avoidable build noise.

## What worked cleanly

- The libcurl transport itself (`apiClient_invoke`), list/error plumbing,
  and every fully-typed model (`get_v1_models`, `usage`, the response
  envelope minus `thought`) generated and compiled without any patching.
- `apiClient_t.response_code` stays `0` on a curl-level failure (nothing
  overwrites it outside the success path) — a clean, reliable signal for
  "the transport failed" vs. "the server returned an error status",
  used for the connection-refused message.
