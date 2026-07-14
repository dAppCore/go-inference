// SPDX-Licence-Identifier: EUPL-1.2

// A minimal C# app using the GENERATED lem SDK (task sdk → build/sdk/csharp)
// to chat with a local gemma4 serve — the OpenAPI standard doing the client
// work: models, a two-turn conversation that proves memory, the thinking
// channel typed on the response root, and usage.

using Lethean.LemSdk.Api;
using Lethean.LemSdk.Client;
using Lethean.LemSdk.Model;

var baseUrl = Environment.GetEnvironmentVariable("LEM_BASE_URL") ?? "http://localhost:36911";
var config = new Configuration { BasePath = baseUrl, Timeout = TimeSpan.FromSeconds(180) };
var inference = new InferenceApi(config);

try
{
    var models = inference.GetV1Models();
    foreach (var m in models.Data)
        Console.WriteLine($"serving: {m.Id}");
}
catch (Exception e) when (e is ApiException or HttpRequestException or AggregateException)
{
    Console.Error.WriteLine($"cannot reach {baseUrl} — start one with: lem serve --model <snapshot>");
    Console.Error.WriteLine($"  ({e.GetBaseException().Message})");
    return 1;
}

PostV1ChatCompletions200Response Ask(List<PostV1ChatCompletionsRequestMessagesInner> messages, bool thinking, int maxTokens)
{
    var request = new PostV1ChatCompletionsRequest(
        messages: messages,
        model: "gemma4",
        maxTokens: maxTokens,
        chatTemplateKwargs: new Dictionary<string, object> { ["enable_thinking"] = thinking });
    return inference.PostV1ChatCompletions(request);
}

// Two turns: the resent history is the memory.
var history = new List<PostV1ChatCompletionsRequestMessagesInner>
{
    new(role: "user", content: "My favourite colour is teal. Reply with one short sentence."),
};
var r1 = Ask(history, thinking: false, maxTokens: 96);
var a1 = r1.Choices[0].Message.Content;
Console.WriteLine($"turn 1: {a1}");
Console.WriteLine($"  usage: {r1.Usage.PromptTokens} prompt + {r1.Usage.CompletionTokens} completion tokens");

history.Add(new(role: "assistant", content: a1));
history.Add(new(role: "user", content: "What is my favourite colour? Answer with just the colour."));
var r2 = Ask(history, thinking: false, maxTokens: 96);
Console.WriteLine($"turn 2 (proves memory): {r2.Choices[0].Message.Content}");
Console.WriteLine($"  usage: {r2.Usage.PromptTokens} prompt + {r2.Usage.CompletionTokens} completion tokens");

// Thinking ON (the gemma4 default): the reasoning arrives TYPED on the
// response root — response.Thought — and the answer stays clean.
var rt = Ask(
    new List<PostV1ChatCompletionsRequestMessagesInner> { new(role: "user", content: "Is 17 prime? One word.") },
    thinking: true, maxTokens: 512);
Console.WriteLine($"thinking answer: {rt.Choices[0].Message.Content}");
var thought = rt.Thought ?? "(none)";
Console.WriteLine($"typed response.Thought (first 90 chars): {thought[..Math.Min(90, thought.Length)].ReplaceLineEndings(" ")}");
Console.WriteLine($"  usage: {rt.Usage.PromptTokens} prompt + {rt.Usage.CompletionTokens} completion tokens");
return 0;
