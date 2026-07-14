#!/usr/bin/env bash
# SPDX-Licence-Identifier: EUPL-1.2
#
# Patches a known bug in openapi-generator's "c" generator (confirmed against
# openapi-generator-cli 7.22.0): any schema with NO declared `type` — lem's
# spec has three (chat message `content`, `chat_template_kwargs`, the generic
# envelope's `error.details`/`response.data`) — is modelled as `any_type_t`,
# but the generator never emits a definition for that type, and the
# generated .c files call it with an EMPTY helper prefix instead of
# `any_type_`: bare `_t`, `_free()`, `_convertToJSON()`, `_parseFromJSON()`.
# None of those symbols exist. The library does not compile until this is
# fixed. See README.md#friction for the full writeup and the upstream repro.
#
# Idempotent: safe to re-run after every `task sdk` regeneration.
set -euo pipefail

sdk_dir="${1:-build/sdk/c}"
if [ ! -f "$sdk_dir/CMakeLists.txt" ]; then
  echo "postgen-fix.sh: $sdk_dir does not look like a generated C SDK (no CMakeLists.txt) — run task sdk first" >&2
  exit 1
fi

# 1. Give any_type_t a real definition (the generated file is an 81-byte
#    placeholder comment: "A placeholder for now, this type isn't really
#    needed" — it is needed).
cat > "$sdk_dir/model/any_type.h" <<'EOF'
/*
 * any_type.h — patched by examples/sdk/c/postgen-fix.sh, see README.md#friction
 */

#ifndef _any_type_H_
#define _any_type_H_

#include "../external/cJSON.h"

typedef struct any_type_t {
    cJSON *json; // owned; NULL means "no value"
} any_type_t;

any_type_t *any_type_create(cJSON *json);
void any_type_free(any_type_t *any_type);
any_type_t *any_type_parseFromJSON(cJSON *json);
cJSON *any_type_convertToJSON(any_type_t *any_type);

#endif /* _any_type_H_ */
EOF

cat > "$sdk_dir/model/any_type.c" <<'EOF'
/*
 * any_type.c — patched by examples/sdk/c/postgen-fix.sh, see README.md#friction
 */
#include <stdlib.h>
#include "any_type.h"

any_type_t *any_type_create(cJSON *json) {
    any_type_t *any_type = malloc(sizeof(any_type_t));
    if (!any_type) {
        return NULL;
    }
    any_type->json = json;
    return any_type;
}

void any_type_free(any_type_t *any_type) {
    if (!any_type) {
        return;
    }
    if (any_type->json) {
        cJSON_Delete(any_type->json);
    }
    free(any_type);
}

any_type_t *any_type_parseFromJSON(cJSON *json) {
    if (!json) {
        return NULL;
    }
    return any_type_create(cJSON_Duplicate(json, 1));
}

cJSON *any_type_convertToJSON(any_type_t *any_type) {
    if (!any_type || !any_type->json) {
        return cJSON_CreateNull();
    }
    return cJSON_Duplicate(any_type->json, 1);
}
EOF

# 2. Wire the new source file into the build (only the header was listed).
if ! grep -q 'model/any_type\.c' "$sdk_dir/CMakeLists.txt"; then
  perl -0pi -e 's{( {4}model/any_type\.h\n)}{$1    model/any_type.c\n}' "$sdk_dir/CMakeLists.txt"
fi

# 3. Fix the three generated files that call the never-defined bare-prefix
#    helpers instead of any_type_*. Word-boundary matches only — this does
#    NOT touch listEntry_t, error_free, error_convertToJSON etc., which are
#    distinct (already-prefixed) identifiers.
for f in error.c response.c post_v1_chat_completions_request_messages_inner.c; do
  path="$sdk_dir/model/$f"
  [ -f "$path" ] || { echo "postgen-fix.sh: expected generated file missing: $path" >&2; exit 1; }
  perl -pi -e '
    s/\b_t\b/any_type_t/g;
    s/\b_free\b/any_type_free/g;
    s/\b_convertToJSON\b/any_type_convertToJSON/g;
    s/\b_parseFromJSON\b/any_type_parseFromJSON/g;
  ' "$path"
done

echo "postgen-fix.sh: patched any_type_t + $sdk_dir/model/{error,response,post_v1_chat_completions_request_messages_inner}.c"
