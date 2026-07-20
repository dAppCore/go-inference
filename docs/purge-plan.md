<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# go-inference git-history purge plan (#51)

Analysis-only. No history rewrite was executed against the real repository —
every number below was measured in a throwaway local clone under
`/private/tmp/purgeprep-scratch/`, cloned `--no-hardlinks` from
`/Users/snider/Code/core/go-inference` at `51c330b2` (HEAD at analysis time).
All dry runs used `git-filter-repo` (Homebrew, `/opt/homebrew/bin/git-filter-repo`).

## Executive summary

| | |
|---|---|
| Current `.git` (fresh-clone baseline) | **688 MiB** |
| Current `.git` (real repo, on disk today) | **1.2 GiB** (packing/gc debt on top of the same content) |
| Current working tree (excl. `.git`) | **833.7 MiB** — of which **783 MiB is `examples/bin/*` compiled binaries, currently committed at HEAD** |
| Dominant history ballast | `examples/bin/**` — compiled example binaries, **1,389.93 MiB** across 102 blob revisions, 78% of all unique blob content ever committed |
| Recommended option | **(b) module-aware safe purge** — 688 MiB → **114 MiB** (83.4% reduction), proven **zero** consumer/checksum impact |
| Prerequisite (independent of history decision) | `examples/bin/` is dead, unreferenced build output sitting live in HEAD today — remove it via a normal commit regardless of what happens to history |

**The "5 tags pin binaries" prior finding is confirmed** (exactly 5: `v0.11.0`,
`v0.12.0`, `go/v0.12.0`, `go/v0.13.0`, `go/v0.14.0`), but with a load-bearing
refinement the prior look missed: only **one** of those five — `go/v0.12.0` —
has ballast *inside its actual Go-module checksum boundary*. The other four
either aren't real go-gettable module tags, or carry ballast that lives in a
nested, never-independently-tagged submodule (`cli/`, `examples/`) and is
therefore outside the `go/` module's zip content. This was verified
empirically (byte-for-byte `go/` subtree hash comparison, see §2) and
corroborated **externally**: `sum.golang.org` already holds an **immutable
public record** for `dappco.re/go/inference@v0.12.0`, so that specific
version's content can never safely change again — which makes option (b)'s
design (never touch that one path) mandatory, not just prudent.

---

## 1. Ballast inventory

Method: `git rev-list --objects --all` (22,739 objects) piped through
`git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)'`,
filtered to blobs (10,506), sorted by size. All figures are **unique blob
content**, i.e. each distinct blob counted once regardless of how many
commits reference it.

### 1.1 Totals

| Metric | Value | Command |
|---|---|---|
| Total unique blob content, all history | 1,779.88 MiB (10,506 blobs) | `git cat-file --batch-check` sum |
| `.git` size, fresh clone | 688 MiB (519.83 MiB packed + 167.51 MiB loose) | `git count-objects -vH`, `du -sh .git` |
| `.git` size, real repo today | 1.2 GiB | `du -sh /Users/snider/Code/core/go-inference/.git` (packing/gc debt — a plain `git gc` would recover some of this regardless of any purge decision, but won't touch reachable ballast) |
| Working tree, fresh clone at HEAD | 833.7 MiB | `du` sum, top-level dirs excl. `.git` |
| Tags | 22 | `git tag -l \| wc -l` |

### 1.2 By top-level directory (all history)

| Directory | Size | Blobs | Note |
|---|---:|---:|---|
| `examples/` | 1,407.39 MiB | 342 | 99% is `examples/bin/` |
| `go/` | 318.21 MiB | 8,902 | mostly legitimate source churn |
| `cli/` | 41.51 MiB | 378 | mostly `cli/mlx.metallib.gz` |
| `docs/` | 9.45 MiB | 385 | normal markdown churn, not ballast |
| `gui/` | 1.09 MiB | 134 | negligible |

### 1.3 The ballast, by path family

| Path family | Size (all revisions) | Blobs | Currently live at HEAD? |
|---|---:|---:|---|
| `examples/bin/**` | **1,389.93 MiB** | 102 | **YES — 59 files, 783 MiB, still tracked today** |
| `examples/thinking` (stray binary, not under `bin/`) | 14.67 MiB | 1 | No (historical only) |
| `cli/mlx.metallib.gz` | 32.55 MiB | 1 | No — removed 2026-07-19 |
| `cli/lthn_kernels.metallib.gz` | 2.67 MiB | 5 | No — removed 2026-07-19 |
| `go/cmd/lem/mlx.metallib.gz` | 46.74 MiB | 1 | No — removed by 2026-07-14 |
| `go/cmd/lem/lthn_kernels.metallib.gz` | 3.57 MiB | 9 | No — removed by 2026-07-14 |
| `go/inference.test` (compiled test binary) | 8.08 MiB | 1 | No, and **never in any tag** |
| `examples/sdk/csharp/bin/**` (compiled .NET example output — dlls, pdbs, deps.json) | 2.01 MiB | 12 | No, and never in any tag |
| **Total ballast** | **≈1,500.2 MiB** | | 84% of all unique blob content ever committed |

Everything else (≈280 MiB: `go/engine/hip` .go/.hip source churn 174 MiB,
`go/engine/metal` .go/.metal source churn 60 MiB, `go/model` test fixtures
23 MiB, normal doc/source revisions) is ordinary development history, not
ballast — confirmed by extension/path breakdown, no further model-fixture or
dist-artefact category was found beyond what's listed above (the "model
fixtures" hypothesis in the original framing did not materialise at scale;
`go/model/**/testdata/*.json` fixtures total 5.44 MiB, trivial).

### 1.4 Reachable-from-live-refs vs historical-only

`git rev-list --objects --all` by definition only enumerates objects
reachable from **some** ref (branch or tag) — nothing here is dangling. The
live/historical split that matters is **HEAD's working tree vs everything
else**:

| | Files | Size | Status |
|---|---:|---:|---|
| `examples/bin/` at current HEAD | 59 | 783 MiB | **live — sitting in the working tree today** |
| `examples/bin/` superseded revisions | 43 | 607.05 MiB | historical only, no longer checked out anywhere |
| Everything else in §1.3 | — | ≈108 MiB | historical only, already cleaned from HEAD by normal commits between 2026-07-14 and 2026-07-19 |

**Finding, independent of the history question**: `examples/bin/*` is
committed, compiled Mach-O binaries (13–19 MiB each), and is **not
referenced anywhere** — `git grep -l "examples/bin" HEAD` (excluding the
binaries themselves) returns nothing, no Taskfile/Makefile/CI target
produces or consumes it, and `examples/.gitignore` already documents the
intent ("go build ./pkg/<name> drops main binaries in cwd — never track
them") for a different, unrelated set of stray root binaries — `examples/bin/`
itself was simply never added to any `.gitignore`. This is dead weight sitting
in the live tree right now, orthogonal to the history-purge decision. It
should be `git rm -r`'d and gitignored via a normal commit regardless of
which history option below is chosen — history-side purging is moot if the
same 783 MiB keeps re-accumulating at HEAD.

---

## 2. The pinning tags

### 2.1 All 22 tags, verified

```
v0.0.1 v0.0.2 v0.0.3 v0.1.0 v0.1.1 v0.1.2 v0.1.3 v0.1.4 v0.1.5 v0.1.6 v0.1.7
v0.2.0 v0.2.1 v0.3.0 v0.8.0-alpha.1 v0.9.0 v0.10.0 v0.11.0 v0.12.0
go/v0.12.0 go/v0.13.0 go/v0.14.0
```

Repo history shows a module reorg: root `go.mod` (module path evolved
`forge.lthn.ai/core/go-inference` → `dappco.re/go/core/inference` →
`dappco.re/go/inference`) through **v0.9.0**; the module then moved to
`go/go.mod` (same final import path, `dappco.re/go/inference`, now living in
the `go/` subdirectory) from **v0.10.0** onward. Per Go's modules-in-subdirectory
convention, only tags prefixed `go/vX.Y.Z` are actually resolvable for a
module whose `go.mod` lives in the `go/` subdirectory — a bare `vX.Y.Z` tag
always resolves against a **root** `go.mod`.

| Tag shape | Count | Resolvable via `go get`? |
|---|---:|---|
| `v0.0.1`–`v0.9.0` (root-module era) | 16 | Yes — root `go.mod` present at every one |
| `go/v0.12.0`, `go/v0.13.0`, `go/v0.14.0` | 3 | Yes — canonical, `go/go.mod` present |
| `v0.10.0`, `v0.11.0`, `v0.12.0` (bare, but `go.mod` only under `go/`) | 3 | **No** — no root `go.mod` at these refs, and no matching `go/`-prefixed sibling for `v0.10.0`/`v0.11.0`. `v0.12.0` (bare) happens to point at the *same commit* as `go/v0.12.0` (verified: both `ab041b35…`), so it's a harmless duplicate alias rather than divergent content, but it is not itself how Go would resolve the module. |

### 2.2 Which tags' trees actually contain ballast

Checked all 22 tags for `*.metallib.gz` anywhere, `examples/bin/`, and nested
`go.mod` boundaries (`git ls-tree -r <tag>`, run once per tag in the scratch
clone):

| Tag | `go/go.mod`? | `cli/go.mod`? | `examples/go.mod`? | Ballast in tree | Ballast inside the `go/` **module boundary**? |
|---|:-:|:-:|:-:|---|:-:|
| `v0.0.1`–`v0.9.0` (×16) | — (root) | no | no | none | n/a |
| `v0.10.0` | yes | no | no | none | n/a |
| `v0.11.0` | yes | no | no | `go/cmd/lem/{mlx,lthn_kernels}.metallib.gz` | **yes** (not go-gettable anyway) |
| `v0.12.0` (bare) | yes | no | no | same as above | **yes** (not go-gettable anyway; identical commit to `go/v0.12.0`) |
| `go/v0.12.0` | yes | no | no | same as above | **YES — real, go-gettable, at risk** |
| `go/v0.13.0` | yes | **yes** | **yes** | `examples/bin/**` (59 files), `examples/thinking`, `cli/{mlx,lthn_kernels}.metallib.gz` | **No** — `cli/` and `examples/` both carry their own `go.mod` at this tag, walling them off from the `go/` module's zip content |
| `go/v0.14.0` | yes | yes | yes | same as `go/v0.13.0` | **No**, same reason |

That's exactly **5** tags whose git tree contains some ballast, matching the
prior finding's count. The refinement: `cli/` and `examples/` became nested
modules (own `go.mod`, first committed 2026-07-14) **before** `go/v0.13.0`
and `go/v0.14.0` were cut, and neither `cli/` nor `examples/` has ever had its
own version tag — so Go's module-zip extraction for the `go/` module (which
only walks files under `go/`, stopping at nested-module boundaries) never
includes their content. Verified empirically, not just reasoned: the `go/`
**subtree hash** (`git rev-parse <tag>:go`) is byte-identical before and
after a full ballast purge for `go/v0.13.0` and `go/v0.14.0`, but changes for
`go/v0.12.0`:

| Tag | `go/` subtree hash, original | `go/` subtree hash, after full purge | Verdict |
|---|---|---|---|
| `go/v0.12.0` | `d26783c4…` | `b9077fb1…` | **CHANGED — breaks checksum** |
| `go/v0.13.0` | `e9088682…` | `e9088682…` | unchanged — safe |
| `go/v0.14.0` | `7267e87e…` | `7267e87e…` | unchanged — safe |
| `v0.10.0` | `15020689…` | `15020689…` | unchanged — safe |

### 2.3 External confirmation: the public Go checksum database

`dappco.re` is a live public vanity import (redirects to the GitHub mirror),
so `sum.golang.org` — the public, **append-only, immutable** Go checksum
ledger — already has entries for versions anyone has ever `go get`-ed with
default settings. Queried directly (`curl https://sum.golang.org/lookup/dappco.re/go/inference@<version>`):

| Version | Public sumdb record | Implication |
|---|---|---|
| `v0.12.0` | **present** (`h1:HFBiXFFW…`) | **Immutable. `go/cmd/lem/*.metallib.gz` must never be rewritten — any content change permanently breaks this version for every future consumer, forever, with no possible fix.** |
| `v0.13.0` | present (`h1:EofDdOs…`) | Safe — recorded hash covers only `go/**`, unaffected per §2.2 |
| `v0.14.0` | present (`h1:3Lddj4W…`) | Safe, same reason |
| `v0.10.0`, `v0.9.0` | present | Ballast-free anyway |
| `v0.11.0` | **404 — "unknown revision go/v0.11.0"** | External confirmation this tag was never actually resolvable as a module version |

This is decisive, not just corroborating: it upgrades `go/v0.12.0` from "a
theoretical risk nobody currently exploits" to "a version whose exact current
content is permanently locked in by infrastructure we don't control." It also
confirms (independently of our own reasoning) that `v0.11.0` is dead. Local
`go env GOPRIVATE` on this machine is `forge.lthn.ai/*` only — it does **not**
cover `dappco.re/*`, so `dappco.re/go/inference` fetches go through the public
sumdb by default unless a consumer's own environment overrides it.

### 2.4 Real consumer impact — dappcore estate

Grepped `go.mod`/`go.sum`/`go.work` across `~/Code/core/*`, `~/Code/lthn/*`,
and this repo's `external/` (no `go.mod` files there — those are C++ vendor
submodules: mlx, rocm-clr, rocm-hip, rocr-runtime).

| Consumer | Requires | Checksummed in `go.sum`? | Active or overridden? | At risk? |
|---|---|:-:|---|:-:|
| `go-api` | `forge.lthn.ai/core/go-inference v0.1.7` | yes | active (no `replace`, no `go.work`) | **No** — v0.1.7 is root-era, zero ballast |
| `go-rocm` | `dappco.re/go/inference v0.10.0` | no `go.sum` entry | **inert** — `replace dappco.re/go/inference => ./external/go-inference/go` overrides it, and that submodule pin (`d6bcf79c…`, host `forge.lthn.sh`) doesn't even resolve against the canonical `git.lthn.sh` history today — pre-existing staleness, unrelated to this purge | No |
| `lthn/eaas` | `forge.lthn.ai/core/go-inference v0.1.0` | yes | active | **No** — v0.1.0 is root-era, zero ballast |
| `lthn/LEM` | `forge.lthn.ai/core/go-inference v0.1.0` | yes | active | **No** — same |

**No locally-observable consumer pins any of the 5 ballast-carrying tags.**
Combined with §2.3: every real, currently-active dependency in the estate is
already immune to every option below, including the most aggressive one.
(Scope honesty: this covers what's checked out locally, plus the public sumdb
spot-check above — it cannot rule out an external consumer we have no
visibility into, but nothing in reach today is at risk.)

---

## 3. Options, measured

Each option was run as a real `git-filter-repo` dry run in its own fresh
`--no-hardlinks` clone under `/private/tmp/purgeprep-scratch/repo-opt*`,
seeded from the same `51c330b2` commit. Baseline: 688 MiB `.git`, 833.7 MiB
working tree.

| Option | `.git` after | Reduction | Working tree after | `go/v0.12.0` checksum | Consumer impact |
|---|---:|---:|---:|:-:|---|
| **(a) full purge + retag everything** | 32 MiB | 95.3% | 50.8 MiB | **BROKEN** (permanently, per §2.3) | Breaks the one publicly-checksummed at-risk version |
| **(b) module-aware safe purge** | **114 MiB** | **83.4%** | 50.9 MiB | **preserved** | **Zero** — proven byte-identical |
| (b-strict) purge only paths absent from *every* tag's whole tree | 525 MiB | 23.7% | 833.9 MiB | preserved | Zero, but leaves 82% of the win on the table for no additional protection (see §3.3) |
| (c) full purge + graft old tags back to pre-purge commits | 521 MiB | 24.3% | 50.9 MiB | preserved (old commit kept alive) | Zero, but see §3.4 — defeats almost the entire win |
| (d) do nothing | 688 MiB | 0% | 833.7 MiB | preserved | Zero, but the live `examples/bin/` problem (§1.4) remains and keeps growing |

### 3.1 Option (a) — full purge + retag

```
git filter-repo --force \
  --path examples/bin --path examples/thinking --path go/inference.test \
  --path cli/mlx.metallib.gz --path cli/lthn_kernels.metallib.gz \
  --path go/cmd/lem/mlx.metallib.gz --path go/cmd/lem/lthn_kernels.metallib.gz \
  --path examples/sdk/csharp/bin \
  --invert-paths
```

Result: `.git` 688→32 MiB (`git count-objects -vH`: 1 pack, 31.09 MiB;
`du -sh .git` → 32M). Maximum possible saving, but it rewrites
`go/cmd/lem/*.metallib.gz` out of `go/v0.12.0`'s tree — the **one** tag with
an immutable public checksum record whose module-boundary content actually
changes. **Not recommended**: this permanently and irreversibly breaks
`dappco.re/go/inference@v0.12.0` for anyone who ever fetches it fresh again,
for a marginal extra 82 MiB over option (b).

Note for the record: essentially **every** commit hash in the rewritten
history differs from the original (this is inherent to how content-addressed
history rewriting works — a commit's hash embeds its parents' hashes, so one
changed blob anywhere cascades forward through the whole DAG). This is true
under **every** rewrite option including (b), and is why a coordinated
force-push + full re-clone is unavoidable no matter which option is chosen —
it is not something (a) uniquely costs. What's option-specific is whether the
Go **module content** (not the git commit SHA) changes for a tag whose
checksum is already public — verified via `<tag>:go` tree-hash comparison
and confirmed identical to the original for `v0.1.7`/`v0.10.0`/`go/v0.13.0`/`go/v0.14.0`
even though their commit SHAs also moved.

### 3.2 Option (b) — module-aware safe purge (recommended)

```
git filter-repo --force \
  --path examples/bin --path examples/thinking --path go/inference.test \
  --path cli/mlx.metallib.gz --path cli/lthn_kernels.metallib.gz \
  --path examples/sdk/csharp/bin \
  --invert-paths
```

The only difference from (a): **`go/cmd/lem/*.metallib.gz` is never touched**
— left in history exactly as-is, forever. Result: `.git` 688→114 MiB
(`git count-objects -vH`: 1 pack, 113.46 MiB). Verified
`git rev-parse go/v0.12.0:go` **identical** before and after
(`d26783c44fe7f81952f61b19d383683e26b925a9` both times). Captures 83% of the
maximum possible reduction (114 of the 32–688 MiB range) at **zero**
measured or theoretical consumer risk, including the one tag with a public,
immutable checksum. This is the recommendation — see §4.

### 3.3 Option (b-strict) — paths absent from every tag's whole tree

```
git filter-repo --force --path go/inference.test --path examples/sdk/csharp/bin --invert-paths
```

The maximally conservative reading of "keeps every tag valid" — protecting
not just module-checksum content but the literal whole-repo tree of every
tag, including `cli/` and `examples/` content nested inside `go/v0.13.0` and
`go/v0.14.0`'s trees even though that content is provably outside their
actual module boundary. Result: `.git` 688→525 MiB (only 23.7% reduction),
working tree unchanged at 833.9 MiB (`examples/bin/` isn't touched, so the
live-tree problem from §1.4 persists entirely). This measurement exists to
show the cost of *not* reasoning about nested-module boundaries: it forfeits
82% of the achievable saving (525 MiB vs 114 MiB) for protection that §2.2/§2.3
already prove is unnecessary. **Not recommended** — strictly dominated by (b).

### 3.4 Option (c) — purge, then graft old tags back to pre-purge commits

Ran the full purge from (a), then force-moved the 5 ballast-carrying tags
back to their original pre-purge commit SHAs (fetched from the pristine
backup clone), then `git reflog expire --expire=now --all && git gc --prune=now`:

```
git filter-repo --force <same paths as option a> --invert-paths
git remote add pristine <pristine-backup-clone>
git fetch pristine 'refs/tags/*:refs/pristine-tags/*'
for t in v0.11.0 v0.12.0 go/v0.12.0 go/v0.13.0 go/v0.14.0; do
  git tag -f "$t" "$(git rev-parse refs/pristine-tags/$t)"
done
git reflog expire --expire=now --all && git gc --prune=now
```

Result: `.git` 688→521 MiB — only **24.3%** reduction, essentially the same
outcome as doing nothing at all to those five tags' worth of history, and
**worse** than (b-strict). Object count after gc was actually **higher**
than the original (27,580 vs ~24,020) — grafting doesn't just fail to save
space, it carries two overlapping histories (the new purged mainline plus the
old originals) simultaneously.

This was expected going in: the five ballast-carrying tags form a single
linear ancestry chain confirmed by `git merge-base --is-ancestor` —
`v0.11.0 → go/v0.12.0 → go/v0.13.0 → go/v0.14.0 → HEAD` — and `go/v0.14.0`
alone is the ancestor of **2,104 of HEAD's 2,392 commits (88%)**. Keeping any
one of the two most recent tags (`go/v0.13.0`, `go/v0.14.0`) reachable at its
original commit keeps nearly the entire ballast-laden history reachable too,
because those tags sit so close to HEAD. **Not recommended** — measured
result confirms it defeats almost the entire size win for zero benefit over
(b), which already achieves the same safety with none of this complexity.

### 3.5 Option (d) — do nothing, shallow-clone CI guidance

`.git` stays at 688 MiB (1.2 GiB as currently packed on disk), and — this is
the part that makes "do nothing" weaker than it sounds — **the live
`examples/bin/` problem from §1.4 is not a history question at all**: 783 MiB
of dead binaries stay checked out at HEAD, and every future accidental
`git add .` risks adding more (nothing currently gitignores `examples/bin/`).

CI exposure today: `.github/workflows/*.yml` uses `actions/checkout@v4` with
no `fetch-depth` override anywhere (`grep` confirms), so GitHub Actions
already defaults to a shallow depth-1 checkout — unaffected either way.
`.gitlab-ci.yml` has **no** `GIT_DEPTH`/`GIT_STRATEGY` override at all, so
GitLab runner behaviour depends entirely on the project's UI-level default
(commonly a shallow depth, but not pinned in-repo, so not verifiable from the
YAML alone). If (d) is chosen, at minimum add an explicit
`variables: { GIT_DEPTH: "1" }` to `.gitlab-ci.yml` so CI checkouts are
provably decoupled from repo history size regardless of what happens to it
later — this is a same-day, zero-risk change independent of the rest of this
plan.

---

## 4. Recommendation

**Option (b), module-aware safe purge**, plus the independent `examples/bin/`
cleanup from §1.4 as a same-day prerequisite. Rationale in one line: 83.4%
reduction (688 → 114 MiB), proven byte-identical Go-module content for every
tag including the one with a public, immutable checksum record, versus a
marginal extra 82 MiB from the fuller purge that would irreversibly break a
version anyone in the world could `go get` today.

### 4.0 Prerequisite (do this regardless, no history rewrite involved)

```
git rm -r examples/bin
echo '/examples/bin/' >> examples/.gitignore   # or repo-root .gitignore
git commit -m "chore(examples): remove committed build output, gitignore examples/bin/"
```

Zero risk, normal commit, immediately recovers 783 MiB from every future
clone's working tree regardless of what happens to history.

### 4.1 Runbook — execution

1. **Back up first.** From the real canonical repo (not this worktree):
   ```
   git bundle create /path/to/backup/go-inference-pre-purge-2026-07-20.bundle --all
   git clone --mirror /Users/snider/Code/core/go-inference /path/to/backup/go-inference-pre-purge-mirror.git
   ```
   Store both off-box (homelab NAS / Bunny via the existing `lethean-dr-ops`
   pipeline). This is the rollback path — `git clone <bundle>` reproduces the
   exact pre-purge state, refs and all, at any time in the future.

2. **Get GitLab ready to accept a force-push.** `dev`/`main` on
   `git.lthn.sh/dappcore/corego/go-inference` are protected branches (per the
   dappcore CI gate model); an admin needs to temporarily enable "allowed to
   force push" on the protected branch(es) before step 4, and re-lock it
   after.

3. **Run the purge** in a fresh clone of the real repo (same command as §3.2):
   ```
   git clone /Users/snider/Code/core/go-inference /path/to/go-inference-purged
   cd /path/to/go-inference-purged
   git filter-repo --force \
     --path examples/bin --path examples/thinking --path go/inference.test \
     --path cli/mlx.metallib.gz --path cli/lthn_kernels.metallib.gz \
     --path examples/sdk/csharp/bin \
     --invert-paths
   ```
   Verify before pushing: `git rev-parse go/v0.12.0:go` must equal
   `d26783c44fe7f81952f61b19d383683e26b925a9` (this doc's measured value at
   `51c330b2` — recompute against whatever HEAD is current at execution time
   using the same command, since new commits will have landed).

4. **Push canonical first**, then the downstream mirror:
   ```
   git remote add gitlab https://git.lthn.sh/dappcore/corego/go-inference.git
   git push --force gitlab --all
   git push --force gitlab --tags
   git remote add github https://github.com/dAppCore/go-inference.git
   git push --force github --all
   git push --force github --tags
   ```
   (This is the one legitimate exception to the "push is non-force" house
   rule — a history rewrite structurally requires it. Re-lock the GitLab
   protected branch immediately after.)

5. **Re-clone every checkout.** Every existing local clone/worktree —
   including the homelab box (`ssh://homelab/home/claude/Code/core/go-inference`,
   check for uncommitted work there first and rescue it before discarding)
   and every agent worktree such as this one — now points at orphaned
   history and must be deleted and re-cloned fresh, not fetched-and-merged.

6. **Fix the now-doubly-stale `go-rocm` submodule.** Its `external/go-inference`
   gitlink already points at a commit absent from canonical history (§2.4) —
   independent pre-existing breakage, but this is a natural moment to also
   point `.gitmodules` at `git.lthn.sh` (not the retired `forge.lthn.sh`) and
   re-pin to a current commit.

7. **Verify CI goes green** on both GitLab and GitHub post-push rather than
   assuming runner self-healing. GitHub Actions runners are ephemeral
   (no cache to invalidate). If the GitLab runner on the homelab box uses a
   persistent workspace (`GIT_STRATEGY: fetch`), confirm it recovers
   automatically on the non-fast-forward divergence; if not, clear its local
   workspace manually.

8. **Retention**: keep the pre-purge bundle/mirror from step 1 indefinitely
   in cold storage — it is the only way to recover `go/cmd/lem/*.metallib.gz`'s
   original history or fully undo the operation.

### 4.2 Rollback

If anything goes wrong after pushing: `git clone go-inference-pre-purge-2026-07-20.bundle`
reproduces the exact pre-purge state; force-push that back to GitLab and
GitHub to fully revert. Do this promptly — the longer post-purge history is
live, the more re-clones/new commits happen on top of it that a rollback
would then orphan a second time.

---

Co-Authored-By: Virgil <virgil@lethean.io>
