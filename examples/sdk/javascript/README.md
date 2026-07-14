# Use Gemma 4 from JavaScript (Node) — local OpenAI-compatible API (lem / go-inference)

```bash
task sdk                                                       # once: generate build/sdk/javascript
npm --prefix ../../../build/sdk/javascript install             # once: build the generated package (see Friction)
cd examples/sdk/javascript
npm install
npm start                                                      # against a running lem serve (LEM_BASE_URL overrides)
```

The client is `lem-sdk-js` (plain `javascript` generator, superagent +
Promises), generated from the OpenAPI spec — typed request/response classes,
no hand-written HTTP. It lists the served models, runs a two-turn
conversation that proves gemma4 remembers turn 1, then compares a
thinking-enabled and a thinking-disabled call, printing token usage after
every request.

## Friction

- **The generated package must be built BEFORE `npm install` here, not
  after.** `npm-generator`'s package.json ships a `prepare: npm run build`
  (babel-compiles `src/` to `dist/`, which is what `main` points at). npm
  *does* try to run `prepare` for a `file:` dependency — but it does so
  without first installing that package's own `devDependencies`
  (`@babel/core`, `@babel/preset-env`, the plugin set), so the build fails
  outright with `sh: babel: command not found` and `dist/` never appears.
  The fix is the two-step install above: build `build/sdk/javascript` in
  place first (a plain `npm install` there installs its real deps and runs
  `prepare` successfully), then install here. The TypeScript sibling example
  has the milder version of this same gotcha; here it's a hard failure, not
  a partial one.
- **Generated model properties are snake_case, not camelCase**, despite
  `modelPropertyNaming: camelCase` being the documented default for the
  `javascript` generator. We set it explicitly and regenerated to confirm —
  no effect. Every multi-word field on the wire comes through unchanged:
  `max_tokens`, `chat_template_kwargs`, `prompt_tokens`,
  `completion_tokens`. This is the opposite convention to the
  `typescript-fetch` sibling client (`maxTokens`, `chatTemplateKwargs`) — if
  you're porting code between the two examples, the field names don't just
  differ in casing, they're a different naming system entirely.
- **The typed `thought` field is unreachable — this is a real spec/server
  mismatch, not a generator quirk.** The OpenAPI document places `thought`
  on `choices[0].message`, and the generated `...ChoicesInnerMessage` class
  matches the spec exactly. But the live server puts `thought` at the TOP
  level of the response body, a sibling of `choices` and `usage`, not
  nested inside the message at all. Confirmed against the raw HTTP body via
  `postV1ChatCompletionsWithHttpInfo()`. Worse: even reaching for the raw
  top-level field on the *typed* response object doesn't help —
  `constructFromObject` only copies the property names it knows about, so
  an untyped top-level `thought` is silently dropped by the generated
  deserialiser too. The only way to read gemma4's thinking trace through
  this client is `response.body.thought` from the `...WithHttpInfo()`
  variant, which `main.js` does and labels explicitly. None of the other
  three sibling examples (go/python/typescript) exercise this path — they
  all call with `enable_thinking: false` and never print `.thought`, so
  this mismatch was still undiscovered before this example ran it live.
- **Thinking is on by default and spends its tokens from the same
  `max_tokens` budget as the answer.** At the sibling examples' usual
  96-token budget, a thinking-enabled call for a plain Q&A spends the whole
  budget reasoning and returns empty `content` (`finish_reason` never
  reached). The two-turn memory demo therefore sends
  `chat_template_kwargs: {enable_thinking: false}` — the thinking demo is
  the one place in this example that leaves thinking on, and it needs
  `max_tokens: 1024` to get a full reasoning trace AND a finished answer.
- **`npm start` runs plain `node main.js`** — no TypeScript, no build step
  of our own. The generator's own `dist/` is CommonJS (no `"type": "module"`
  in its package.json, and Babel's default target is CommonJS), so `main.js`
  uses `require`, matching what actually ships.
- On this machine, `npm install` prints an unrelated `npm warn
  allow-scripts` notice about the SDK's `prepare` script — that's a local
  npm plugin gate, not something this package or example controls; the
  install still succeeds.

## Live run (captured against `lem serve`, gemma-4-E2B)

```
serving: 42f62737af7a9fd8c1d55d79666c1a217be4e2e2
gemma4 (turn 1): That's an interesting combination! It sounds like you have a real passion for **Zig** [...]
usage: 25 prompt + 91 completion tokens
gemma4 (turn 2): Your pet quokka's name is **Bartholomew**.
usage: 135 prompt + 15 completion tokens
gemma4 (thinking, default): To find the product of 17 times 24, we can use the standard multiplication method [...] 17 times 24 is **408**.
thought (typed, choices[0].message.thought): (no thought field returned)
thought (raw response body, top level): Thinking Process: [...] (Self-Correction: Ensure the reasoning is explicit.)
usage: 30 prompt + 672 completion tokens
gemma4 (enable_thinking: false): To find the answer to $17 \times 24$ [...]
thought (typed, choices[0].message.thought): (no thought field returned)
thought (raw response body, top level): (no thought field returned)
usage: 23 prompt + 96 completion tokens
```
