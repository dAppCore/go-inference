<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# chat/budget

Thinking budget: WithThinkingBudget caps how many tokens a reasoning model
(thinking ON, the Gemma 4 default) may spend inside its thought channel
before the backend forces the channel closed and moves to the visible
answer. GenerateMetrics.ThinkingBudgetForced reports whether that forced
close actually happened on this generation.

## Run

```sh
go run ./pkg/chat/budget -model ~/models/gemma-4-e2b-it-4bit
```

Flags and behaviour are documented in [main.go](main.go) — the code is the example.
