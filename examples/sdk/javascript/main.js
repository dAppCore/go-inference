// SPDX-Licence-Identifier: EUPL-1.2

// A minimal Node.js app using the GENERATED lem SDK (task sdk -> build/sdk/javascript)
// to chat with a local gemma4 serve -- the OpenAPI standard doing the client work.
//
// Plain CommonJS to match what the `javascript` (superagent) generator's build
// step actually emits to dist/ (see README's Friction section for why).

const LemSdk = require("lem-sdk-js");

const base = process.env.LEM_BASE_URL ?? "http://localhost:36911";
const apiClient = new LemSdk.ApiClient(base);
// requests queue through lem's serial scheduler -- a slow neighbour can push
// a call well past superagent's 60s default, so give it room.
apiClient.timeout = 120000;
const inference = new LemSdk.InferenceApi(apiClient);

// The javascript generator's model classes carry the API's raw wire-format
// field names (snake_case: max_tokens, chat_template_kwargs, prompt_tokens --
// see README Friction), unlike the camelCase typescript-fetch client.
function userMessage(content) {
  const m = new LemSdk.PostV1ChatCompletionsRequestMessagesInner("user");
  m.content = content;
  return m;
}

function assistantMessage(content) {
  const m = new LemSdk.PostV1ChatCompletionsRequestMessagesInner("assistant");
  m.content = content;
  return m;
}

function logUsage(response) {
  if (response.usage) {
    console.log(
      `usage: ${response.usage.prompt_tokens} prompt + ${response.usage.completion_tokens} completion tokens`,
    );
  }
}

// superagent buries the real connection error under a nested `.error` --
// `err.message`/`err.code` are undefined at the top level (confirmed against
// a closed port; see README Friction).
function connectionHint(err) {
  const code = err && err.error && err.error.code;
  if (code === "ECONNREFUSED") {
    return `cannot reach lem at ${base} -- start it first: lem serve (or: lem tui -> Service tab -> enter)`;
  }
  return null;
}

async function chat(messages, maxTokens, chatTemplateKwargs) {
  const request = new LemSdk.PostV1ChatCompletionsRequest(messages, "gemma4");
  request.max_tokens = maxTokens;
  if (chatTemplateKwargs) {
    request.chat_template_kwargs = chatTemplateKwargs;
  }
  try {
    return await inference.postV1ChatCompletions(request);
  } catch (err) {
    console.error(connectionHint(err) ?? `chat: ${err.error ?? err}`);
    process.exit(1);
  }
}

// Same call, but keeps the raw superagent Response alongside the typed
// object -- see README Friction: the live server puts `thought` at the TOP
// of the response body, the spec (and so the generated model) puts it on
// `choices[0].message`, and the generator's allow-list constructFromObject
// drops fields it doesn't recognise -- so the typed field is unreachable and
// this raw body is the only place gemma4's thinking trace actually survives.
async function chatWithRawBody(messages, maxTokens, chatTemplateKwargs) {
  const request = new LemSdk.PostV1ChatCompletionsRequest(messages, "gemma4");
  request.max_tokens = maxTokens;
  if (chatTemplateKwargs) {
    request.chat_template_kwargs = chatTemplateKwargs;
  }
  try {
    const { data, response } = await inference.postV1ChatCompletionsWithHttpInfo(request);
    return { data, rawThought: response.body.thought };
  } catch (err) {
    console.error(connectionHint(err) ?? `chat: ${err.error ?? err}`);
    process.exit(1);
  }
}

async function main() {
  let models;
  try {
    models = await inference.getV1Models();
  } catch (err) {
    console.error(connectionHint(err) ?? `models: ${err.error ?? err}`);
    process.exit(1);
  }
  for (const m of models.data) {
    console.log("serving:", m.id);
  }

  // (b) two-turn conversation -- turn 2 resends the full history so gemma4
  // can only answer correctly if it actually used what turn 1 said. Thinking
  // is off here (see (c) below for why): with the default 96-token budget
  // gemma4 spends it all reasoning and content comes back empty.
  const noThinking = { enable_thinking: false };
  const history = [
    userMessage(
      "My favourite programming language is Zig, and my pet quokka is called Bartholomew.",
    ),
  ];
  const turn1 = await chat(history, 96, noThinking);
  const reply1 = turn1.choices[0].message.content;
  console.log("gemma4 (turn 1):", reply1);
  logUsage(turn1);

  history.push(assistantMessage(reply1));
  history.push(userMessage("What is my pet quokka's name?"));
  const turn2 = await chat(history, 96, noThinking);
  console.log("gemma4 (turn 2):", turn2.choices[0].message.content);
  logUsage(turn2);

  // (c) thinking demo -- defaults (thinking on) vs. chat_template_kwargs
  // {enable_thinking: false}. `message.thought` is the SPEC's typed reasoning
  // channel; whether it actually arrives there is exactly what we're
  // checking (see README Friction for what we found).
  const question = [userMessage("What is 17 times 24? Show your reasoning.")];

  // Thinking eats from the same max_tokens budget as the answer -- give it
  // enough room to finish reasoning AND reply, or content comes back empty.
  const thinkingOn = await chatWithRawBody(question, 1024);
  console.log("gemma4 (thinking, default):", thinkingOn.data.choices[0].message.content);
  console.log(
    "thought (typed, choices[0].message.thought):",
    thinkingOn.data.choices[0].message.thought ?? "(no thought field returned)",
  );
  console.log(
    "thought (raw response body, top level):",
    thinkingOn.rawThought ?? "(no thought field returned)",
  );
  logUsage(thinkingOn.data);

  const thinkingOff = await chatWithRawBody(question, 96, { enable_thinking: false });
  console.log("gemma4 (enable_thinking: false):", thinkingOff.data.choices[0].message.content);
  console.log(
    "thought (typed, choices[0].message.thought):",
    thinkingOff.data.choices[0].message.thought ?? "(no thought field returned)",
  );
  console.log(
    "thought (raw response body, top level):",
    thinkingOff.rawThought ?? "(no thought field returned)",
  );
  logUsage(thinkingOff.data);
}

main();
