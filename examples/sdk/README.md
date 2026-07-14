# examples/sdk — gemma4 from any language, via the OpenAPI standard

`lem spec` exports go-inference's HTTP surface as an OpenAPI 3.1 document, and
[openapi-generator](https://openapi-generator.tech/docs/generators) turns that
document into a typed client in **any of its ~90 languages** — SDKs for free,
no hand-written client code. The seventeen lanes here are the proof: the same
API driven past hello world (models, a two-turn conversation that proves
memory, the thinking toggle, token usage, friendly errors) through one
generated client per language, each run LIVE against a local `lem serve`.

Tuned for **gemma4 / Lemma** — the engine this repo optimises (thinking
channel split into a typed `thought` field on the response root, MTP, prompt
reuse). Other models serve through the same API; results vary, it's OSS.

## Once: generate the clients

```bash
task sdk          # spec → build/sdk/openapi.json → build/sdk/<lang>/ for every sdk-config/*.yaml
```

(Needs `openapi-generator-cli` + a JRE — `brew install openapi-generator openjdk`.
Add a language by dropping a `sdk-config/<lang>.yaml` — each existing config
was picked deliberately from its generator's options page.)

## Once: start a serve

```bash
task build && ./bin/lem serve --model ~/.cache/huggingface/hub/models--mlx-community--gemma-4-E2B-it-qat-4bit/snapshots/<hash>
# or from the TUI: lem tui → Service tab → enter
```

Every example reads `LEM_BASE_URL` (default `http://localhost:36911`).

## The lanes

| Language | Directory | Run |
|----------|-----------|-----|
| Go | [`go/`](go/) | `GOWORK=off go run .` |
| Python | [`python/`](python/) | `pip install ../../../build/sdk/python && python3 main.py` |
| TypeScript (fetch) | [`typescript/`](typescript/) | `npm install && npm start` |
| TypeScript (axios) | [`typescript-axios/`](typescript-axios/) | `npm install && npm start` — interceptor timing demo |
| Angular | [`typescript-angular/`](typescript-angular/) | `npx ng serve` — injectable services, dev-proxy to the serve |
| JavaScript (Node) | [`javascript/`](javascript/) | `npm install && npm start` |
| Rust | [`rust/`](rust/) | `cargo run` |
| Java | [`java/`](java/) | `mvn -f ../../../build/sdk/java install -DskipTests && mvn exec:java` |
| Kotlin | [`kotlin/`](kotlin/) | `gradle run` (JDK 21) |
| Swift | [`swift/`](swift/) | `swift run` |
| PHP | [`php/`](php/) | `composer install && php main.php` |
| Ruby | [`ruby/`](ruby/) | `bundle install --path vendor/bundle && bundle exec ruby main.rb` |
| Perl | [`perl/`](perl/) | `cpanm -l local Log::Any URI::Query && perl main.pl` |
| C | [`c/`](c/) | see its README — the generator's output needs `postgen-fix.sh` first |
| C# | [`csharp/`](csharp/) | `DOTNET_ROLL_FORWARD=LatestMajor dotnet run` — typed `response.Thought` end to end |
| Bash | [`bash/`](bash/) | `./chat.sh` — the generated curl client, no runtime at all |
| IDE (.http) | [`http-client/`](http-client/) | open `lem.http` in an IntelliJ-family IDE, or `ijhttp` |

Each lane's README carries a **Friction** section — every rough edge hit
while building it, honestly. That's the point: the fleet that built these
found (and go-inference fixed, same day) four real defects before any user
could report them — the serial/batch scheduler dropping the thinking
override, `/v1/messages` rejecting Anthropic's string-content shorthand,
the spec mis-placing `thought` under `message` when the wire puts it on the
response root, and multi-tagged routes generating duplicate Go types.

All seventeen lanes are live — the C# lane joined once `dotnet-sdk`
was installed, and being generated after the spec fixes it was the first
with the thinking channel fully typed (`response.Thought`), no raw-JSON
workaround.
