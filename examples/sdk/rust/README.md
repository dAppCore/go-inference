# sdk/rust — gemma4 via the generated Rust client

```bash
task sdk                # once: generate build/sdk/rust
cd examples/sdk/rust
cargo run               # against a running lem serve (LEM_BASE_URL overrides)
```

The client is the `lem-sdk` crate (reqwest), generated from the OpenAPI
spec — typed serde models (`choices[].message.{content,thought}`), no
hand-written HTTP. Strictly typed: this is the lane the honest response
schemas were built for (a mis-declared spec fails deserialisation here first).
