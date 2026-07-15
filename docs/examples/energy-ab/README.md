# The energy A/B — replay vs `-state` continuity

Measures joules per conversation turn, same binary, same traffic, one flag.

1. `mkdir -p ~/Lethean/lem/bench/energy-ab` once, then in your own terminal
   (sudo needs a TTY):
   `sudo powermetrics --samplers cpu_power,gpu_power -i 500 -o ~/Lethean/lem/bench/energy-ab/pm.txt`
2. 60s idle window (markers via the driver's clock).
3. Arm A: `LTHN_PROMPT_REUSE=0 lem serve --model <snapshot>
   -state-conversations=false`, then
   `LEM_PORT=<port> python3 energy_turns.py REPLAY`.
   (Since 64688fb the stateless lane ships llama-parity prompt reuse ON
   and its per-turn wall is flat — the kill switch reproduces the
   cache-less replay shape this arm measures.)
4. Arm B: serve with the default (continuity on), `energy_turns.py STATE`.
5. `python3 pm_integrate.py` — integrates Combined Power over the marked
   windows, subtracts idle, reports J/turn and the per-turn series.

Receipt (2026-07-13, M3 Ultra, gemma-4 E2B 4-bit, 10 turns, ~340 words/turn,
idle 1.02W): replay 1,439 J net (climbing 83→233 J/turn as history grows);
continuity 771 J net (flat ~77 J/turn). Identical output both arms
(3,370 vs 3,364 words). Continuity: 54% of the energy, gap widening
every turn.

## Peer arm — llama-server (2026-07-13, same session)

Same driver, same 10 turns, llama-server at shipping defaults
(`-ngl 99 -c 8192`, its prefix cache active — confirmed by its flat
per-turn trend): **1,121 J net = 112.1 J/turn** (0.031 Wh/turn), output
3,223 words. Three-way: lem continuity 77.1 J/turn · llama-server
112.1 · pure replay 143.9. Per million turns: 21.4 / 31.1 / 40.0 kWh.
llama's cache is single-slot volatile RAM (lost on restart/eviction);
lem's state is durable per-conversation.
