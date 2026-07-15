// SPDX-Licence-Identifier: EUPL-1.2

// A C app using the GENERATED lem SDK (task sdk -> build/sdk/c, then
// ./postgen-fix.sh — see README.md#friction for why that second step
// exists) to drive a local gemma4 serve through the OpenAPI-generated
// libcurl client: list models, hold a two-turn conversation that proves
// memory, and probe the thinking channel both with and without
// chat_template_kwargs.

#include <ctype.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "include/apiClient.h"
#include "include/list.h"
#include "api/InferenceAPI.h"
#include "model/get_v1_models_200_response.h"
#include "model/get_v1_models_200_response_data_inner.h"
#include "model/post_v1_chat_completions_request.h"
#include "model/post_v1_chat_completions_request_messages_inner.h"
#include "model/post_v1_chat_completions_200_response.h"
#include "model/post_v1_chat_completions_200_response_choices_inner.h"
#include "model/post_v1_chat_completions_200_response_choices_inner_message.h"
#include "model/post_v1_chat_completions_200_response_usage.h"
#include "model/any_type.h"
#include "model/object.h"
#include "external/cJSON.h"

// The serve queues requests through a single serial scheduler, so a request
// that's merely waiting its turn can sit for a while — 120s covers that.
// CONNECTTIMEOUT stays short so a dead serve (nothing listening) fails fast
// instead of hanging for the full 120s.
static void configure_timeouts(CURL *curl) {
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 120L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 5L);
}

// apiClient_invoke() never sets response_code on a curl-level failure (it
// only prints its own diagnostic to stderr) — response_code == 0 is the
// caller's only signal that the transport itself failed, connection-refused
// included.
static void die_if_unreachable(apiClient_t *client, const char *base) {
    if (client->response_code != 0) {
        return;
    }
    fflush(stdout); // keep stdout's progress in order ahead of this stderr message when piped
    fprintf(stderr,
        "error: could not reach %s\n"
        "       start a serve first — from the repo root:\n"
        "       task build && ./bin/lem serve --model <path-to-model>\n",
        base);
    exit(1);
}

static bool contains_ci(const char *haystack, const char *needle) {
    if (!haystack || !needle) {
        return false;
    }
    size_t hlen = strlen(haystack), nlen = strlen(needle);
    for (size_t i = 0; i + nlen <= hlen; i++) {
        size_t j = 0;
        for (; j < nlen; j++) {
            if (tolower((unsigned char)haystack[i + j]) != tolower((unsigned char)needle[j])) {
                break;
            }
        }
        if (j == nlen) {
            return true;
        }
    }
    return false;
}

static post_v1_chat_completions_request_messages_inner_t *make_message(const char *role, const char *text) {
    post_v1_chat_completions_request_messages_inner_t *msg = malloc(sizeof(*msg));
    msg->_library_owned = 1;
    msg->role = strdup(role);
    // content has no declared type in the spec (string, or an array of
    // typed parts) — the generated model is any_type_t, a cJSON passthrough.
    msg->content = any_type_create(cJSON_CreateString(text));
    return msg;
}

static void print_usage(post_v1_chat_completions_200_response_usage_t *usage) {
    if (!usage) {
        printf("  usage: (not reported)\n");
        return;
    }
    printf("  usage: %d prompt + %d completion = %d total tokens\n",
        usage->prompt_tokens ? *usage->prompt_tokens : 0,
        usage->completion_tokens ? *usage->completion_tokens : 0,
        usage->total_tokens ? *usage->total_tokens : 0);
}

// Sends `messages` (caller keeps ownership — the request's own copy of the
// pointer is detached before the request is freed), prints the reply, and
// returns a heap copy of the answer text so the caller can fold it back
// into the next turn's history.
static char *chat(apiClient_t *client, const char *base, list_t *messages, object_t *kwargs, int max_tokens) {
    post_v1_chat_completions_request_t *req = malloc(sizeof(*req));
    memset(req, 0, sizeof(*req));
    req->_library_owned = 1;
    req->model = strdup("gemma4"); // cosmetic on a single-model serve
    req->messages = messages;
    req->max_tokens = malloc(sizeof(int));
    *req->max_tokens = max_tokens;
    req->chat_template_kwargs = kwargs; // NULL is fine — means "use the template default"

    post_v1_chat_completions_200_response_t *resp = InferenceAPI_postV1ChatCompletions(client, req);
    die_if_unreachable(client, base);
    if (!resp) {
        fprintf(stderr, "error: chat completion failed, HTTP %ld\n", client->response_code);
        exit(1);
    }

    listEntry_t *choiceEntry = resp->choices ? resp->choices->firstEntry : NULL;
    if (!choiceEntry) {
        fprintf(stderr, "error: no choices in response\n");
        exit(1);
    }
    post_v1_chat_completions_200_response_choices_inner_t *choice = choiceEntry->data;
    const char *content = choice->message->content ? choice->message->content : "";
    printf("  gemma4: %s\n", content);
    // choice->message->thought is ALWAYS NULL here — not a client bug. The
    // live serve puts "thought" at the top level of the response (a sibling
    // of "choices"), but the exported spec — and so every generated client,
    // this one included — declares it nested under choices[].message. The
    // typed field genuinely cannot see the real value. See README.md#friction.
    printf("  thought: %s\n", choice->message->thought ? choice->message->thought : "(none)");
    print_usage(resp->usage);

    char *answer = strdup(content);
    post_v1_chat_completions_200_response_free(resp);

    req->messages = NULL; // history belongs to the caller, not this request
    post_v1_chat_completions_request_free(req);
    return answer;
}

int main(void) {
    const char *base = getenv("LEM_BASE_URL");
    if (!base || !*base) {
        base = "http://localhost:36911";
    }

    apiClient_t *client = apiClient_create_with_base_path(base, NULL);
    client->curl_pre_invoke_func = configure_timeouts;

    printf("== models ==\n");
    get_v1_models_200_response_t *models = InferenceAPI_getV1Models(client);
    die_if_unreachable(client, base);
    if (!models) {
        fprintf(stderr, "error: models request failed, HTTP %ld\n", client->response_code);
        return 1;
    }
    listEntry_t *modelEntry;
    list_ForEach(modelEntry, models->data) {
        get_v1_models_200_response_data_inner_t *m = modelEntry->data;
        printf("serving: %s\n", m->id);
    }
    get_v1_models_200_response_free(models);

    printf("\n== two-turn conversation (turn 2 proves turn 1 is remembered) ==\n");
    list_t *history = list_createList();
    list_addElement(history, make_message("user",
        "My favourite colour is teal. Just reply with a one-sentence acknowledgement."));
    char *turn1 = chat(client, base, history, NULL, 128);

    list_addElement(history, make_message("assistant", turn1));
    list_addElement(history, make_message("user", "What's my favourite colour? Answer in one word."));
    char *turn2 = chat(client, base, history, NULL, 128);
    printf("  memory check: %s (looked for \"teal\" in turn 2's answer)\n",
        contains_ci(turn2, "teal") ? "PASS" : "FAIL");
    free(turn1);
    free(turn2);

    listEntry_t *histEntry;
    list_ForEach(histEntry, history) {
        post_v1_chat_completions_request_messages_inner_free(histEntry->data);
    }
    list_freeList(history);

    // 512 tokens (vs 128 above) — a thinking model spends its budget on the
    // hidden reasoning FIRST, so a tight cap can exhaust it before any
    // visible content is emitted (finish_reason "length", content ""). See
    // README.md#friction for the run that happened to us at 128.
    printf("\n== thinking channel: defaults ==\n");
    list_t *thinkingOn = list_createList();
    list_addElement(thinkingOn, make_message("user", "Is 17 prime? Show your reasoning, then give the final answer."));
    char *withThought = chat(client, base, thinkingOn, NULL, 512);
    free(withThought);
    listEntry_t *e;
    list_ForEach(e, thinkingOn) {
        post_v1_chat_completions_request_messages_inner_free(e->data);
    }
    list_freeList(thinkingOn);

    printf("\n== thinking channel: chat_template_kwargs {\"enable_thinking\": false} ==\n");
    // object_t has no typed setters (see README.md#friction) — the only way
    // to populate a freeform object field is to hand it a raw JSON string.
    object_t *kwargs = object_create();
    kwargs->temporary = strdup("{\"enable_thinking\": false}");
    list_t *thinkingOff = list_createList();
    list_addElement(thinkingOff, make_message("user", "Is 17 prime? Show your reasoning, then give the final answer."));
    char *withoutThought = chat(client, base, thinkingOff, kwargs, 512); // kwargs is freed by chat()'s request_free
    free(withoutThought);
    list_ForEach(e, thinkingOff) {
        post_v1_chat_completions_request_messages_inner_free(e->data);
    }
    list_freeList(thinkingOff);

    apiClient_free(client);
    return 0;
}
