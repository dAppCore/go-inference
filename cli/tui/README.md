# tui — `lem tui`

A Bubble Tea terminal UI over the go-inference library: pick a model, chat
with streaming + the thinking channel, and host the HTTP API — all in one
process, one copy of the weights.

## Tabs (`tab` / `shift+tab` to move)

| Tab | Does |
|-----|------|
| **Chat** | streaming transcript + input; `esc` cancels a reply, `ctrl+t` flips thinking |
| **Models** | discovered checkpoints (HF cache + `$LEM_MODELS_DIR`); `enter` loads with the Settings context length |
| **Service** | OpenAI/Anthropic/Ollama HTTP API for the **loaded** model — same weights, no second load. TUI turns and API requests share one serial scheduler lane, so nothing races the engine. Point opencode/codex at `http://localhost:36911/v1`, Anthropic/Ollama clients at the root |
| **Settings** | context length (applies at load), max tokens, thinking default/on/off |
| **Tools** | built-in demo tools; when armed, declarations ride the system turn, calls execute locally and feed back — the full agent loop in the terminal |
| **Modes** | sampling presets: Balanced (checkpoint defaults) · Greedy · Creative · Coder |

## Layout

| File | Concern |
|------|---------|
| `tui.go` | `Run` — flags, `-check` headless frame, program boot |
| `app.go` | the Elm update loop: keys, messages, tab routing, tool loop |
| `tabs.go` | tab bar (bubbletea tabs-example borders) |
| `stream.go` | generation goroutine → event channel → `waitEvent` bridge |
| `service.go` | the Service tab: serial scheduler + `serving.Serve` lifecycle |
| `picker.go` / `settings.go` / `modes.go` / `tools.go` | per-tab state + views |
| `style.go` | lipgloss palette (dark) |

Everything is testable headless: `app_test.go` drives `Update` directly, and
the `LTHN_PROBE_MODEL`-gated live tests load a real checkpoint, chat through
the real loop, and curl the Service tab's API.
