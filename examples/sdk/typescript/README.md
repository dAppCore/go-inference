# sdk/typescript — gemma4 via the generated TypeScript client

```bash
task sdk                    # once: generate build/sdk/typescript
cd examples/sdk/typescript
npm --prefix ../../../build/sdk/typescript install   # once: build the generated package
npm --prefix ../../../build/sdk/typescript run build # (npm's script gate can block its prepare)
npm install
npm start                   # against a running lem serve (LEM_BASE_URL overrides)
```

The client is `@lethean/lem-sdk` (typescript-fetch), generated from the
OpenAPI spec — typed camelCase models (`choices[].message.{content,thought}`),
no hand-written HTTP.
