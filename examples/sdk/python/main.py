# SPDX-Licence-Identifier: EUPL-1.2
"""A minimal Python app using the GENERATED lem SDK (task sdk → build/sdk/python)
to chat with a local gemma4 serve — the OpenAPI standard doing the client work."""

import os
import sys

import lem_sdk

base = os.environ.get("LEM_BASE_URL", "http://localhost:36911")

with lem_sdk.ApiClient(lem_sdk.Configuration(host=base)) as api_client:
    inference = lem_sdk.InferenceApi(api_client)

    models = inference.get_v1_models()
    for m in models.data:
        print("serving:", m.id)

    request = lem_sdk.PostV1ChatCompletionsRequest(
        model="gemma4",
        messages=[
            lem_sdk.PostV1ChatCompletionsRequestMessagesInner(
                role="user",
                content="In one sentence, why does local inference matter?",
            )
        ],
        max_tokens=96,
        chat_template_kwargs={"enable_thinking": False},
    )
    response = inference.post_v1_chat_completions(request)
    if not response.choices:
        sys.exit("no choices in response")
    print("gemma4:", response.choices[0].message.content)
    if response.usage:
        print(f"usage: {response.usage.prompt_tokens} prompt + "
              f"{response.usage.completion_tokens} completion tokens")
