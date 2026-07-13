# W4 KV residue findings

No production defect was found in this write-blind pass.

The remaining decode-cadence arms cannot be exercised honestly without a live
Metal session fixture. `MTPDecodeEach` reaches them only after both prompt
prefills, target/draft `stepID` calls, and target verify have populated
resident state (`go/engine/metal/mtp.go:156-365`). The sampled sibling has the
same requirement and additionally needs live target head sampling
(`go/engine/metal/mtp.go:416-668`). A unit test that merely constructs
`ArchSession` values cannot determine acceptance, rejection, rollback, pool
fallback, or re-engagement correctly; it would bypass the decode contract.

The corresponding cache hit/miss and ring-rollback paths call prefill, retained
hidden/logit replay, and speculative-KV truncation (`go/engine/metal/prompt_cache.go:69-242`).
Their existing runtime tests must be run by the landing GPU suite. This pass
adds only direct, non-GPU invariants: raw-block metadata rejection and wrapped
write placement; MTP preflight capacity and draft-length guards; and retained
prompt-boundary eligibility.
