// SPDX-Licence-Identifier: EUPL-1.2

// A minimal TypeScript app using the GENERATED lem SDK (task sdk →
// build/sdk/typescript-axios) to chat with a local gemma4 serve — the
// OpenAPI standard doing the client work, this time through axios rather
// than fetch (examples/sdk/typescript). See README.md's "Friction" section
// for where the two TypeScript variants disagree — it's more than the
// import name.

import axios, { type InternalAxiosRequestConfig } from "axios";
import {
  Configuration,
  InferenceApi,
  type PostV1ChatCompletions200Response,
  type PostV1ChatCompletionsRequestMessagesInner,
} from "@lethean/lem-sdk-axios";

const base = process.env.LEM_BASE_URL ?? "http://localhost:36911";

// --- the axios-specific advantage: an interceptor pair timing every call.
// fetch has no request/response hook — you'd have to wrap each call site by
// hand. This is the reason the axios variant exists alongside the fetch one.
type TimedConfig = InternalAxiosRequestConfig & { __start?: number };
const http = axios.create();
http.interceptors.request.use((config: TimedConfig) => {
  config.__start = performance.now();
  return config;
});
http.interceptors.response.use((response) => {
  const start = (response.config as TimedConfig).__start;
  const ms = start !== undefined ? (performance.now() - start).toFixed(0) : "?";
  console.log(`  [axios] ${response.config.method?.toUpperCase()} ${response.config.url} — ${ms}ms`);
  return response;
});

const inference = new InferenceApi(new Configuration({ basePath: base }), base, http);

// Print the reply, its usage, and its `thought` — both the TYPED path
// (choices[0].message.thought, per the OpenAPI spec) and the ACTUAL wire
// path. See Friction: the live server puts `thought` on the response root,
// not on the message, so the typed field is always undefined in practice.
function report(label: string, body: PostV1ChatCompletions200Response): void {
  const choice = body.choices[0];
  console.log(`${label}: ${choice?.message.content || "(empty — see finish_reason below)"}`);
  console.log(`  finish_reason: ${choice?.finish_reason}`);
  console.log(
    `  thought (typed choices[0].message.thought): ${choice?.message.thought ?? "(absent — see Friction)"}`,
  );
  const actualThought = (body as unknown as { thought?: string }).thought;
  console.log(
    `  thought (actual, response body's top-level .thought): ${
      actualThought ? actualThought.replace(/\s+/g, " ").slice(0, 100) + "..." : "(none)"
    }`,
  );
  if (body.usage) {
    console.log(`  usage: ${body.usage.prompt_tokens} prompt + ${body.usage.completion_tokens} completion tokens`);
  }
}

async function main(): Promise<void> {
  const models = await inference.getV1Models();
  // NB: models is an AxiosResponse — .data unwraps the HTTP envelope, then
  // .data again is the OpenAPI schema's own field name. See Friction.
  for (const m of models.data.data) {
    console.log("serving:", m.id);
  }

  // --- two-turn conversation: the SDK is stateless HTTP, so turn 2 resends
  // the full history — that's what lets the model prove it remembered turn 1.
  const history: PostV1ChatCompletionsRequestMessagesInner[] = [
    { role: "user", content: "Remember the number 47. Acknowledge it in one short sentence." },
  ];
  const turn1 = (
    await inference.postV1ChatCompletions({ model: "gemma4", messages: history, max_tokens: 220 })
  ).data;
  report("turn 1", turn1);
  history.push({ role: "assistant", content: turn1.choices[0]?.message.content ?? "" });
  history.push({ role: "user", content: "What number did I ask you to remember? Reply with just the number." });

  const turn2 = (
    await inference.postV1ChatCompletions({ model: "gemma4", messages: history, max_tokens: 220 })
  ).data;
  report("turn 2 (proves memory — should contain 47)", turn2);

  // --- thinking demo: defaults vs an explicit enable_thinking: false.
  const thinkingDefault = (
    await inference.postV1ChatCompletions({
      model: "gemma4",
      messages: [{ role: "user", content: "In one short sentence, name the capital of Spain." }],
      max_tokens: 220,
    })
  ).data;
  report("thinking (defaults)", thinkingDefault);

  const thinkingOff = (
    await inference.postV1ChatCompletions({
      model: "gemma4",
      messages: [{ role: "user", content: "In one short sentence, name the capital of Portugal." }],
      max_tokens: 220,
      chat_template_kwargs: { enable_thinking: false },
    })
  ).data;
  report("thinking (chat_template_kwargs: { enable_thinking: false })", thinkingOff);
}

main().catch((err: unknown) => {
  if (axios.isAxiosError(err) && err.code === "ECONNREFUSED") {
    console.error(`\nCould not reach ${base} — start a serve first: lem serve\n`);
    process.exit(1);
  }
  throw err;
});
