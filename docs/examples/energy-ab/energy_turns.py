#!/usr/bin/env python3
"""energy_turns.py <label> — 10-turn conversation, FULL history resent each
turn (standard client shape). Identical traffic both arms; the serve flag is
the only experimental variable. Markers land in pm_markers.txt."""
import json
import sys
import time
import os
import urllib.request

LABEL = sys.argv[1]
# Fixed, predictable location shared with pm_integrate.py and the manually-run
# `powermetrics` (see README) — NOT under /tmp, which is world-writable and
# subject to symlink/race attacks on a predictable path (sonar python:S5443).
# ~/Lethean/lem/<subdir> is this repo's established home for lem's own runtime
# state (models/, tuning/, welfare/, ai/ — see go/serving, go/agent/ai).
STATE_DIR = os.path.join(os.path.expanduser("~"), "Lethean", "lem", "bench", "energy-ab")
os.makedirs(STATE_DIR, exist_ok=True)
MARKERS = os.path.join(STATE_DIR, "pm_markers.txt")
TOPICS = [
    "the frozen ridge at dawn", "the crevasse field crossing",
    "the supply cache dilemma", "the aurora over base camp",
    "the dog team's endurance", "the whiteout navigation",
    "the rival expedition's tracks", "the summit weather window",
    "the descent in failing light", "the return and the tally",
]


def mark(label: str) -> None:
    with open(MARKERS, "a") as f:
        f.write(f"{time.time():.6f} {label}\n")


def turn(history: list) -> str:
    body = json.dumps({
        "model": "x", "max_tokens": 400,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": history,
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{os.environ.get('LEM_PORT', '11434')}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    return r["choices"][0]["message"]["content"]


history = []
mark(f"ARM_{LABEL}_START")
for i, topic in enumerate(TOPICS, 1):
    history.append({"role": "user", "content":
                    f"Write the next chapter of the expedition log, about {topic}. "
                    f"Keep it vivid and continue from everything so far."})
    mark(f"TURN_{LABEL}_{i}_START")
    reply = turn(history)
    mark(f"TURN_{LABEL}_{i}_END")
    history.append({"role": "assistant", "content": reply})
mark(f"ARM_{LABEL}_END")
words = sum(len(m["content"].split()) for m in history)
print(f"arm {LABEL} complete: {words} words of history across {len(history)} messages")
