# Handover

## 2026-07-16 — codex welcome-back brief (stint salvaged while the box was unreachable)

Your 93-commit hip stint (box dev@553744b) was extracted and **merged to
origin dev** while your quota was out and the box's sshd was down — fast-forward
your dev from origin before doing anything else. The exact pre-merge tip is
preserved as branch `codex/hip-stint-2026-07-16`. Landing receipts: hip builds
clean on the box at the merged tip; vet carries exactly your 5 known
unsafe.Pointer findings; the 6 failing tests below reproduce at YOUR tip
553744b, so none of them are merge damage.

What moved under you while you were away (fixed on landing, but the follow-ups
are yours):

- **model/arch re-homing**: `model/mamba2` → `model/arch/mamba2`,
  `model/qwen3` → `model/arch/Qwen/qwen3`. Two of your imports were fixed on
  landing (`mamba2_runtime.go`, `composed_runtime_test.go`).
- **The vision payload types moved to `model/vision`** (`vision.Linear`,
  `vision.Layer`, `vision.Projector`, `vision.Config`, `vision.Loaded`,
  `vision.UnifiedConfig`, `vision.Unified` — field-for-field identical).
  Deprecated aliases in `go/model/loaded.go` keep your `model.LoadedVision*`
  spellings compiling. **Yours: the mechanical rename in engine/hip, then
  delete the alias block.**
- **Untagged-portable contract**: three pure-Go helpers you defined in
  `linux && amd64`-tagged files but called from UNTAGGED files broke the
  Mac-side "untagged vet 0" gate (`hipBFloat16ToFloat32`,
  `hipFloat32ToBFloat16`, `assertFloat32SlicesNear`). Relocated to untagged
  homes on landing (`hip_bf16_portable.go`, `hip_assert_portable_test.go`).
  Before handing a stint back, gate the untagged surface on darwin too
  (`GOOS=darwin go vet ./engine/hip/` from any tree).

Your lane's own debt, in priority order:

1. **6 failing tests at your tip** (`go test ./engine/hip/` on the box):
   - `TestHIPKernels_MLXAffineQ6ProjectionBatchRow16LaunchConfig_Good` —
     launch-config want/got drift after your row16/row32 widening commits.
   - `TestHIPAttentionHeadsChunkedEligible_BlockPagesGood` and
     `_Gemma4HeadDim512_Good` — eligibility predicates changed by your GQA
     scan-sharing work; the asserts still pin the old behaviour.
   - `TestNativeContract_CapabilityReportGenericReactiveRegistryLabels_Good`
     and `TestNativeContract_RocmGemma4Q6CapabilityLabels_Good` — registry
     label maps drifted from the asserted shapes.
   - `TestDecodeHelpers_Bad_PlanAttachedDrafterRejectsGemma4NonLinkedTargetPack`
     (`e4b_mxfp8_planned_only`) — asserts the not-linked refusal, gets the
     not-runnable-on-this-card refusal on the RX 7800 XT; make the assert
     hardware-honest.
   A test updated to match new behaviour must still TEST something — no
   want/got mirroring.
2. **5 unsafe.Pointer vet findings** in `hip_driver_cgo.go` (4 pre-existing,
   1 from the last stint).
3. `internal/gguf` watch item stands: engine-internal GGUF reading is
   acceptable, but any drift toward the shared gguf format home gets pulled
   back there.
