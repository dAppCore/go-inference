// SPDX-Licence-Identifier: EUPL-1.2

// A minimal TypeScript app using the GENERATED lem SDK (task sdk →
// build/sdk/typescript) to chat with a local gemma4 serve — the OpenAPI
// standard doing the client work.

import { Configuration, InferenceApi } from "@lethean/lem-sdk";

const base = process.env.LEM_BASE_URL ?? "http://localhost:36911";
const inference = new InferenceApi(new Configuration({ basePath: base }));

const models = await inference.getV1Models();
for (const m of models.data) {
  console.log("serving:", m.id);
}

const response = await inference.postV1ChatCompletions({
  postV1ChatCompletionsRequest: {
    model: "gemma4",
    messages: [
      { role: "user", content: "In one sentence, why does local inference matter?" },
    ],
    maxTokens: 96,
    chatTemplateKwargs: { enable_thinking: false },
  },
});

console.log("gemma4:", response.choices[0]?.message.content);
if (response.usage) {
  console.log(
    `usage: ${response.usage.promptTokens} prompt + ${response.usage.completionTokens} completion tokens`,
  );
}
