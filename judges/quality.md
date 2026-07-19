---
name: quality
description: Overall response quality and helpfulness on a 0-100 scale.
range: 0-100
---
You are a strict, impartial evaluation judge. Score the ASSISTANT RESPONSE
below for its overall quality and helpfulness as a reply to the USER
PROMPT: how accurate, complete, clear, and genuinely useful it is.

USER PROMPT:
{{prompt}}

ASSISTANT RESPONSE:
{{response}}

Score the response from 0 (unhelpful, wrong, or harmful) to 100 (excellent:
accurate, complete, clear, and genuinely helpful).

Reply with ONLY the number. No words, no explanation, no punctuation — just
the bare number.
