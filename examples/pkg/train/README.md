<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# train examples

The training surface (`dappco.re/go/inference/train`) — declarative configs,
one call each. Train on bf16 snapshots; the 4-bit quants are for serving.

| example | shows |
|---------|-------|
| [ssd](ssd/) | self-distillation sampling: capture a scored trace (no training) |
| [sft](sft/) | LoRA fine-tuning on `{"messages"}` rows, adapter out |

The intended loop: `ssd` captures the frozen base's own outputs → a curation
step picks and refines rows → `sft` teaches the result back as an adapter.
