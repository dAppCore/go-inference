# Use Gemma 4 from TypeScript (axios) — local OpenAI-compatible API (lem / go-inference)

The axios sibling of [`examples/sdk/typescript`](../typescript/) (typescript-fetch).
Same OpenAPI spec, same `lem serve`, a different generator — and the two are
**not** drop-in equivalents (see Friction). This example exists to prove the
one thing fetch can't do out of the box: axios interceptors, demonstrated
here as per-request timing.

```bash
task sdk:spec                                                  # once: build/sdk/openapi.json
export JAVA_HOME="$(brew --prefix openjdk)" PATH="$JAVA_HOME/bin:$PATH"  # if java isn't already on PATH
openapi-generator-cli generate -i build/sdk/openapi.json \
  -c sdk-config/typescript-axios.yaml -o build/sdk/typescript-axios      # generate the client

cd examples/sdk/typescript-axios
npm --prefix ../../../build/sdk/typescript-axios install    # once: build the generated package
npm --prefix ../../../build/sdk/typescript-axios run build  # only needed if the line above's prepare gate blocked (see Friction)
npm install
npm start                    # against a running lem serve (LEM_BASE_URL overrides, default http://localhost:36911)
```

The client is `@lethean/lem-sdk-axios` (typescript-axios), generated from the
same OpenAPI spec as the fetch variant. `main.ts` goes past hello world:

1. lists the served models,
2. a two-turn conversation where turn 2 resends history and proves the model
   remembered turn 1 (the number 47),
3. a thinking demo — one call with defaults, one with
   `chat_template_kwargs: { enable_thinking: false }` — printing the
   `thought` text (see Friction: it's not where the OpenAPI schema says),
4. prints prompt/completion token usage after every call,
5. a request/response **interceptor pair logging wall-clock timing per
   call** — the axios-specific feature this variant exists to show off,
6. a clear `ECONNREFUSED` message telling you to start `lem serve` if
   nothing's listening.

## Friction

Real rough edges hit building this, kept honest rather than smoothed over:

- **`thought` is not where the OpenAPI spec (and therefore the generated
  types) say it is.** The spec nests it at `choices[0].message.thought`, and
  the generated `PostV1ChatCompletions200ResponseChoicesInnerMessage`
  interface agrees. The live serve actually returns it on the **response
  root** (`{ id, object, choices, usage, thought }`), sibling to `choices`,
  not inside the message. `choices[0].message.thought` is `undefined` on
  every real call; the text is only reachable by casting the response past
  its own type (`(body as unknown as { thought?: string }).thought`). This is
  a go-inference spec/implementation mismatch, not an SDK generation issue —
  worth a follow-up on the driver side, but out of scope here.

- **`enable_thinking: false` was not reliably honoured while testing this.**
  Identical requests (same prompt, same flag) suppressed `thought` on some
  calls and not on others across the session — it looked ordering/cache
  dependent rather than random (go-inference does prompt-prefix reuse). The
  captured run below happens to show it working; an earlier run in the same
  session, with a different prompt history, did not. Don't take one clean
  run as proof the flag is dependable — flag this to the driver owner before
  relying on it.

- **Every response is `AxiosResponse<T>`, not `T`.** The typescript-axios
  class methods return `AxiosPromise<T>` (`Promise<AxiosResponse<T>>`), so
  every call needs an extra `.data` unwrap that the fetch variant doesn't:
  `(await inference.postV1ChatCompletions(...)).data`. For `/v1/models` this
  produces the slightly confusing `models.data.data` — the first `.data` is
  axios's HTTP envelope, the second is the OpenAPI schema's own field
  (`GetV1Models200Response.data: Array<...>`). Same field name, two
  different layers.

- **Model property names are `snake_case`, not `camelCase`.** typescript-axios
  has no `modelPropertyNaming` option and emits properties exactly as they
  appear in the OpenAPI schema — `chat_template_kwargs`, `max_tokens`,
  `prompt_tokens`, `completion_tokens`. The fetch variant's generator
  produces camelCase for the same fields (`chatTemplateKwargs`, `maxTokens`,
  `promptTokens`). Copying a fetch-variant snippet verbatim into an axios
  project will not compile — same spec, genuinely different generated
  shapes. Operation *method* names stay camelCase in both (`getV1Models`,
  `postV1ChatCompletions`) — only the model/property naming differs.

- **Request shape is unwrapped, not wrapped.** The fetch client's methods
  take a single named-parameter object (`{ postV1ChatCompletionsRequest:
  {...} }`); the axios client's class methods take the request body
  directly as the first argument (`postV1ChatCompletions({ model,
  messages, ... })`). Another place a copy-pasted fetch snippet silently
  breaks.

- **The `npm --prefix ... install` prepare gate didn't actually block this
  time** — `install` ran the package's `prepare`/`build` script itself and
  `dist/` came out fully built. The explicit `npm run build` line above is
  still worth keeping for portability: npm's script-execution policy for
  nested/`file:` deps varies by npm version and any `ignore-scripts`/
  `allow-scripts` config in the environment, and the fetch lane hit exactly
  that block. If `dist/` is missing after `install`, run the explicit
  `build` line.

## Verified live capture

Against a real `lem serve` (gemma-4-E2B, `http://localhost:36911`):

```
$ npm start
  [axios] GET http://localhost:36911/v1/models — 11ms
serving: 42f62737af7a9fd8c1d55d79666c1a217be4e2e2
  [axios] POST http://localhost:36911/v1/chat/completions — 1505ms
turn 1: I remember the number 47.
  finish_reason: stop
  thought (typed choices[0].message.thought): (absent — see Friction)
  thought (actual, response body's top-level .thought):  Thinking Process: 1. **Analyze the Request:** The user wants me to remember the number 47 and ackno...
  usage: 32 prompt + 203 completion tokens
  [axios] POST http://localhost:36911/v1/chat/completions — 757ms
turn 2 (proves memory — should contain 47): 47
  finish_reason: stop
  thought (typed choices[0].message.thought): (absent — see Friction)
  thought (actual, response body's top-level .thought):  Thinking Process: 1. **Analyze the Request:** The user is asking "What number did I ask you to reme...
  usage: 65 prompt + 100 completion tokens
  [axios] POST http://localhost:36911/v1/chat/completions — 987ms
thinking (defaults): The capital of Spain is Madrid.
  finish_reason: stop
  thought (typed choices[0].message.thought): (absent — see Friction)
  thought (actual, response body's top-level .thought):  Thinking Process: 1. **Analyze the Request:** The user wants to know the capital of Spain, and the ...
  usage: 27 prompt + 133 completion tokens
  [axios] POST http://localhost:36911/v1/chat/completions — 110ms
thinking (chat_template_kwargs: { enable_thinking: false }): The capital of Portugal is Lisbon.
  finish_reason: stop
  thought (typed choices[0].message.thought): (absent — see Friction)
  thought (actual, response body's top-level .thought): (none)
  usage: 20 prompt + 8 completion tokens
```

And the connection-refused path (`LEM_BASE_URL` pointed at nothing listening):

```
$ LEM_BASE_URL=http://localhost:1 npm start

Could not reach http://localhost:1 — start a serve first: lem serve
```
