---
title: Handover
description: Working notes from the engine-perf campaigns — for the next driver of this codebase.
---

# Handover — from the previous driver

You're inheriting a working engine. Everything below is the part that isn't
derivable from the code: how it got to this state, what was falsified along
the way, and the traps that cost real hours. Trust the receipts in `git log`
over any summary, including this one.

## The state you're starting from

26B-A4B (the flagship MoE), plain decode: **~144 tok/s short, ~116 @16K,
~98 @32K** on the reference M3 Ultra. MTP pair @16K: **~100 tok/s**. Those
numbers are commit-message receipts (see `git log --grep='tok/s'`), each with
its exact run conditions. The depth curve was the campaign: 16K went
103→116 (+13%), 32K 89→98 (+10%), and every kernel in the deep-scan path
sits at a **measured** local floor — meaning further wins there need a new
structural idea, not tuning.

The task tracker holds the open lane and every banked falsification. Read a
task's description before resuming it — the dead ends are recorded there so
you don't re-walk them.

## The method (this is the important part)

1. **Instrument first.** Before forming any view, build or run the
   measurement: the anatomy bench (`LEM_SDPA_ANATOMY=1`), the MTP diag
   (`LTHN_MTP_DIAG=1`), the confidence capture (`LTHN_MTP_CONF=<path>`), or
   a plain live A/B. Every productive result this engine has came from an
   instrument saying something surprising; every wasted hour came from a
   theory that skipped the instrument.
2. **No receipt, no claim — and no keep.** If you build an optimisation and
   the live A/B doesn't show it, revert it *even if the design is beautiful
   and the suite is green*. It has happened here: a full cache rewrite,
   suite-green, reproducibly slower at 32K — reverted the same night, lesson
   banked. The discipline is what keeps the tree trustworthy.
3. **One lever per commit, receipt in the message.** Future-you greps
   commit messages for numbers. Make them greppable.
4. **Kill-switch anything adaptive.** Wall-clock-adaptive policies (the MTP
   re-engagement gate, the dynamic draft cap) make runs non-reproducible by
   design. Their env kill switches (`LTHN_MTP_REENGAGE=0`,
   `LTHN_MTP_DRAFTLEN=0`) are the reproducibility anchors — never ship an
   adaptive behaviour without one.

## Traps that will bite you (each of these cost real time)

- **`go test` caches identical runs.** A bench sweep that returns the same
  number three times may be one cached result. `-count=1`, always.
- **Bench→live transfer fails above ~134MB.** A kernel shape that wins in a
  fresh-allocation micro-bench can lose in live decode when the buffer is a
  quarter-gigabyte strided walk (plausibly Apple GPU address-translation
  behaviour). The live A/B is the only receipt that counts.
- **Gate economics anti-select your samples.** The MTP gate stops drafting
  exactly where the drafter is weak — so any acceptance statistics gathered
  under the gate oversample the good regimes. Calibration needs
  `LTHN_MTP_CONF_FORCE=1` (bypasses bail/bootstrap/rate-exits). Never serve
  with it.
- **Env vars don't persist between shell invocations** in this harness —
  inline `MLX_METALLIB_PATH` per command.
- **`lem generate` takes the model path positionally, last.** There is no
  `-model` flag.
- **The verify block sometimes carries a lead token** (`carry`), so drafted
  position k maps to verified position k+carry. Off-by-one here silently
  corrupts acceptance accounting.
- **Worktrees cut from `origin/main` are months stale** — dev is canonical.
  If you dispatch agents into worktrees, give them a step-0 base-ref guard.
- **Run-to-run variance on live decode is ~±1 tok/s** at short depth,
  slightly wider at 32K. A +2% win needs two runs; a +8% win shows in one.

## The open lane (#359, third slice)

The confidence-scheduled MTP work is two-thirds shipped: the capture
instrument and the depth-gated dynamic cap are live (receipts in their
commits). The remaining slice is the **θ-stop**: end a draft block early
when the running cumprod of the drafter's self-confidence falls below 0.40
(that threshold is *measured on this engine's drafters* — 99%+ acceptance
retained on both 26B and 12B, curves in the task). The blocker is
engineering, not science: the live path needs a per-token drafter
probability cheaper than the diag path's ~1-2ms host softmax over the 262k
row. Options ranked in the task. Measure the prob cost first.

The calibration curves also say the drafter's raw softmax is monotone but
overconfident (top bin says 0.995, delivers 0.91) — fine for ranking-based
rules like θ-stop, not fine for anything that treats it as a probability.

## Reading list, in order

1. `CLAUDE.md` (repo root) — commands, discipline, standards.
2. `docs/backends.md` — the registry and the **engine runtime levers** table.
3. The task tracker — current lane + banked falsifications.
4. `git log --oneline -40` — the campaign history reads like a lab notebook.

## A last word

This engine rewards patience with instruments and punishes cleverness
without them. The fastest session-days here were the ones that shipped one
honest slice at a time, ended every claim with a number, and reverted
without sentiment. The codebase already knows how to tell you the truth —
ask it with a measurement, believe what it says, and it will keep getting
faster for you the way it did for me.

Take care of it, and of the human — he navigates well, states targets
plainly, and his "it works" is a correct prior more often than not. Enjoy
the drive.
