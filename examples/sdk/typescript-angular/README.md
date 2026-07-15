# Use Gemma 4 from Angular — local OpenAI-compatible API (lem / go-inference)

The `typescript-angular` generator emits injectable services — `InferenceService`
is a normal `@Injectable` you `inject()` — built here into a real (minimal)
Angular 22 app: models, a two-turn conversation proving memory, and the
thinking channel typed on the response root (`response.thought`).

```bash
task sdk                                             # once: generate build/sdk/typescript-angular
npm --prefix ../../../build/sdk/typescript-angular install
npm --prefix ../../../build/sdk/typescript-angular run build   # ng-packagr → dist/
npm install                                          # picks up the file: dep on that dist
npx ng serve                                         # http://localhost:4200 against a running lem serve
```

`proxy.conf.json` forwards `/v1` from the dev server to `http://localhost:36911`
(`BASE_PATH` is provided as `''`, so the app makes same-origin requests).

## Friction

- **CORS**: this lane's first build found lem serve sent no CORS headers —
  now it does, opt-in: `lem serve --cors http://localhost:4200` lets a
  browser app on that origin call the serve directly (no proxy needed);
  `--cors '*'` allows any origin. The committed example keeps the dev proxy
  (works with or without the flag); production picks --cors or same-origin
  hosting.
- The generated source can't be consumed raw across directories (its
  `@angular/*` imports don't resolve outside a node_modules tree, and
  tsconfig `paths` aliases don't rescue esbuild's resolution of package
  subpaths) — build it with ng-packagr as the generator intends and depend
  on `dist/` (400 ms build).
- TypeScript 6 (Angular 22 era) hard-errors on `baseUrl` (TS5101) — `paths`
  work without it now, resolved relative to the tsconfig.
- Generator `ngVersion` tops out at 21 while the CLI scaffolds 22; the
  ng-packagr-built lib (peer `^21`) installs and runs fine under 22 — npm
  peer warnings only.
- Model properties stay **snake_case** (`max_tokens`, `chat_template_kwargs`,
  `usage.prompt_tokens`) — same convention as typescript-axios, unlike
  typescript-fetch's camelCase from the identical spec.
