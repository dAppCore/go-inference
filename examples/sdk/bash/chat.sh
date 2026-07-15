#!/usr/bin/env bash
# SPDX-Licence-Identifier: EUPL-1.2
#
# gemma4 from plain bash via the GENERATED curl client (task sdk → the `bash`
# generator's build/sdk/bash/lem-cli.sh) — models, a two-turn conversation
# that proves memory, the thinking toggle, and usage, no SDK runtime at all.
set -euo pipefail

BASE="${LEM_BASE_URL:-http://localhost:36911}"
CLI="$(dirname "$0")/../../../build/sdk/bash/lem-cli.sh"
[ -f "$CLI" ] || { echo "generated client missing — run: task sdk" >&2; exit 1; }
command -v jq >/dev/null || { echo "this demo pretty-prints with jq — brew install jq" >&2; exit 1; }

call() { bash "$CLI" --host "$BASE" --content-type application/json "$@"; }

if ! out=$(call getV1Models 2>&1); then
  echo "cannot reach $BASE — start one with: lem serve --model <snapshot>" >&2
  exit 1
fi
echo "serving: $(jq -r '.data[].id' <<<"$out")"

ask() { # ask <messages-json> -> full response on stdout
  jq -cn --argjson messages "$1" \
    '{model:"gemma4", max_tokens:192, chat_template_kwargs:{enable_thinking:false}, messages:$messages}' \
    | call postV1ChatCompletions -
}

# Turn 1: state a fact.
h1='[{"role":"user","content":"My favourite colour is teal. Reply with one short sentence."}]'
r1=$(ask "$h1")
a1=$(jq -r '.choices[0].message.content' <<<"$r1")
echo "turn 1: $a1"
echo "  usage: $(jq -r '.usage | "\(.prompt_tokens) prompt + \(.completion_tokens) completion"' <<<"$r1")"

# Turn 2: the resent history is the memory — the model must recall the fact.
h2=$(jq -cn --argjson h "$h1" --arg a "$a1" \
  '$h + [{role:"assistant",content:$a},{role:"user",content:"What is my favourite colour? Answer with just the colour."}]')
r2=$(ask "$h2")
echo "turn 2: $(jq -r '.choices[0].message.content' <<<"$r2")"
echo "  usage: $(jq -r '.usage | "\(.prompt_tokens) prompt + \(.completion_tokens) completion"' <<<"$r2")"

# Thinking on (the gemma4 default): the reasoning arrives as a separate
# thought field, the answer stays clean.
rt=$(jq -cn '{model:"gemma4", max_tokens:512, messages:[{role:"user","content":"Is 17 prime? One word."}]}' \
  | call postV1ChatCompletions -)
echo "thinking demo: $(jq -r '.choices[0].message.content' <<<"$rt")"
echo "  thought (first 90 chars): $(jq -r '(.thought // .choices[0].message.thought // "-") | .[0:90]' <<<"$rt")"
