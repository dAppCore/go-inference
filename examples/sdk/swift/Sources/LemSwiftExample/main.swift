// SPDX-Licence-Identifier: EUPL-1.2

// A Swift app using the GENERATED lem SDK (task sdk -> build/sdk/swift) to
// chat with a local gemma4 serve - the OpenAPI standard doing the client
// work. Goes past hello-world: lists served models, holds a two-turn
// conversation (proving the model remembered turn one), and demonstrates
// the thinking channel both on (default) and explicitly disabled.

import Foundation
import LemSDK

let base = ProcessInfo.processInfo.environment["LEM_BASE_URL"] ?? "http://localhost:36911"
let apiConfiguration = LemSDKAPIConfiguration(basePath: base)

func printUsage(_ usage: PostV1ChatCompletions200ResponseUsage?) {
    guard let usage else { return }
    print("  usage: \(usage.promptTokens ?? 0) prompt + \(usage.completionTokens ?? 0) completion tokens")
}

func chat(
    _ messages: [PostV1ChatCompletionsRequestMessagesInner],
    enableThinking: Bool? = nil
) async throws -> PostV1ChatCompletions200Response {
    var request = PostV1ChatCompletionsRequest(messages: messages, model: "gemma4")
    request.maxTokens = 200
    if let enableThinking {
        request.chatTemplateKwargs = JSONValue(["enable_thinking": JSONValue(enableThinking)])
    }
    return try await InferenceAPI.postV1ChatCompletions(
        postV1ChatCompletionsRequest: request, apiConfiguration: apiConfiguration)
}

func userMessage(_ text: String) -> PostV1ChatCompletionsRequestMessagesInner {
    PostV1ChatCompletionsRequestMessagesInner(content: JSONValue(text), role: "user")
}

func assistantMessage(_ text: String) -> PostV1ChatCompletionsRequestMessagesInner {
    PostV1ChatCompletionsRequestMessagesInner(content: JSONValue(text), role: "assistant")
}

do {
    // (a) list the served models
    let models = try await InferenceAPI.getV1Models(apiConfiguration: apiConfiguration)
    for model in models.data {
        print("serving:", model.id)
    }

    // (b) two-turn conversation - turn 2 resends the full history so gemma4
    // can prove it remembered what turn 1 told it (no server-side session).
    var history = [userMessage("My favourite colour is teal. Remember that in one short sentence.")]
    let turn1 = try await chat(history, enableThinking: false)
    let turn1Content = turn1.choices.first?.message.content ?? ""
    print("\nturn 1:", turn1Content)
    printUsage(turn1.usage)

    history.append(assistantMessage(turn1Content))
    history.append(userMessage("What's my favourite colour? Answer in one word."))
    let turn2 = try await chat(history, enableThinking: false)
    print("turn 2:", turn2.choices.first?.message.content ?? "")
    printUsage(turn2.usage)

    // (c) thinking demo - defaults, then explicitly disabled. See the
    // Friction section in README.md: gemma4's own chat template defaults
    // enable_thinking to false, AND go-inference's exported OpenAPI spec
    // mis-places `thought` under choices[].message (the live wire body
    // actually puts it at the response root) - so `.thought` prints empty
    // either way. The completion-token gap below (spent on hidden reasoning
    // vs a two-word answer) is the only typed evidence thinking happened.
    let thinkingQuestion = [userMessage("Is 7 a prime number? Answer briefly.")]

    let thinkingOn = try await chat(thinkingQuestion)
    print("\nthought:", thinkingOn.choices.first?.message.thought ?? "(none returned)")
    print("answer:", thinkingOn.choices.first?.message.content ?? "")
    printUsage(thinkingOn.usage)

    let thinkingOff = try await chat(thinkingQuestion, enableThinking: false)
    print("\nthought (enable_thinking=false):", thinkingOff.choices.first?.message.thought ?? "(none returned)")
    print("answer:", thinkingOff.choices.first?.message.content ?? "")
    printUsage(thinkingOff.usage)
} catch ErrorResponse.error(let statusCode, _, _, let underlying) {
    if let urlError = underlying as? URLError, urlError.code == .cannotConnectToHost {
        FileHandle.standardError.write(Data("""
            no lem serve running at \(base) - start one with \
            `lem serve --model <path>` (or the TUI's Service tab) and retry.\n
            """.utf8))
        exit(1)
    }
    FileHandle.standardError.write(Data("chat: HTTP \(statusCode): \(underlying)\n".utf8))
    exit(1)
}
