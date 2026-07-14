# Use Gemma 4 from C# — local OpenAI-compatible API (lem / go-inference)

```bash
task sdk                    # once: generate build/sdk/csharp
cd examples/sdk/csharp
DOTNET_ROLL_FORWARD=LatestMajor dotnet run   # against a running lem serve (LEM_BASE_URL overrides)
```

The client is `Lethean.LemSdk` (httpclient library, net8.0), generated from
the OpenAPI spec — typed request builders and a typed response where the
gemma4 reasoning arrives as **`response.Thought`** on the response root, split
from the answer. This was the first lane generated after the spec's thought
placement was fixed, so the thinking channel is fully typed end to end — no
raw-JSON second parse like the earlier lanes needed.

## Friction

- The `csharp` generator's default library is now `generichost` (Generic
  Host + DI machinery) — deliberately overridden to `httpclient` for the
  classic `new Configuration { BasePath }` shape a console demo wants.
- Targeting `net8.0` (LTS) from a machine with only the .NET 10 runtime
  fails at LAUNCH (`Microsoft.NETCore.App 8.0.0 not found`) even though the
  SDK builds it fine — `DOTNET_ROLL_FORWARD=LatestMajor` runs it on the
  newer runtime. Machines with the 8.x runtime installed don't need it.
- `brew install --cask dotnet-sdk` needs sudo (an interactive installer) —
  the only lane in the fleet whose toolchain couldn't be provisioned
  unattended.
- Generated ctor parameters are lowerCamel while properties are TitleCase;
  the `object` schema field becomes `varObject` to dodge the keyword.
