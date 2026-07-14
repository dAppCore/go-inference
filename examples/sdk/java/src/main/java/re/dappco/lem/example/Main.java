// SPDX-Licence-Identifier: EUPL-1.2

package re.dappco.lem.example;

// A Java app using the GENERATED lem SDK (task sdk -> build/sdk/java) to
// chat with a local gemma4 serve -- the OpenAPI standard doing the client
// work. Goes past hello-world: lists the served models, runs a two-turn
// conversation that proves the model remembers turn one, and probes the
// `thought` (reasoning channel) field with and without
// chat_template_kwargs.enable_thinking.
//
// See README.md's "Friction" section for why the thinking demo also reads
// the raw response body: the exported OpenAPI spec nests `thought` under
// choices[].message, but this serve emits it as a top-level sibling of
// `choices` -- the typed message.getThought() is always null against the
// real server.

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.net.ConnectException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import okhttp3.Call;
import okhttp3.Response;
import re.dappco.lem.api.InferenceApi;
import re.dappco.lem.invoker.ApiClient;
import re.dappco.lem.invoker.ApiException;
import re.dappco.lem.invoker.JSON;
import re.dappco.lem.model.GetV1Models200ResponseDataInner;
import re.dappco.lem.model.PostV1ChatCompletions200Response;
import re.dappco.lem.model.PostV1ChatCompletions200ResponseChoicesInnerMessage;
import re.dappco.lem.model.PostV1ChatCompletions200ResponseUsage;
import re.dappco.lem.model.PostV1ChatCompletionsRequest;
import re.dappco.lem.model.PostV1ChatCompletionsRequestMessagesInner;

public final class Main {

    private static final String DEFAULT_BASE_URL = "http://localhost:36911";

    // The driver queues requests through a serial scheduler -- a busy queue
    // can make even a small request wait a while behind a bigger one.
    private static final int READ_TIMEOUT_MILLIS = 150_000;
    private static final int CONNECT_TIMEOUT_MILLIS = 10_000;

    private Main() {
    }

    public static void main(String[] args) {
        String base = System.getenv().getOrDefault("LEM_BASE_URL", DEFAULT_BASE_URL);

        ApiClient apiClient = new ApiClient();
        apiClient.setBasePath(base);
        apiClient.setReadTimeout(READ_TIMEOUT_MILLIS);
        apiClient.setConnectTimeout(CONNECT_TIMEOUT_MILLIS);
        InferenceApi inference = new InferenceApi(apiClient);

        try {
            listModels(inference);
            twoTurnConversation(inference);
            thinkingDemo(inference, /* useDefaults= */ true);
            thinkingDemo(inference, /* useDefaults= */ false);
        } catch (ApiException e) {
            if (isConnectionRefused(e)) {
                System.err.println("Could not reach " + base + " -- start a serve first: "
                        + "lem serve --model <path-to-model>");
                System.exit(1);
            }
            throw new RuntimeException("lem API call failed: " + e.getMessage(), e);
        }
    }

    private static void listModels(InferenceApi inference) throws ApiException {
        for (GetV1Models200ResponseDataInner m : inference.getV1Models().getData()) {
            System.out.println("serving: " + m.getId());
        }
    }

    /** Two turns, one connection: turn 2 resends turn 1 in full so the model answers from history. */
    private static void twoTurnConversation(InferenceApi inference) throws ApiException {
        List<PostV1ChatCompletionsRequestMessagesInner> history = new ArrayList<>();
        history.add(userMessage("My favourite colour is teal. Remember that for later."));

        ChatResult turn1 = chat(inference, history, 500, disableThinking());
        String reply1 = firstContent(turn1);
        System.out.println("turn 1: " + reply1);
        printUsage(turn1.typed());

        history.add(assistantMessage(reply1));
        history.add(userMessage("What is my favourite colour?"));

        ChatResult turn2 = chat(inference, history, 500, disableThinking());
        String reply2 = firstContent(turn2);
        System.out.println("turn 2 (proves memory): " + reply2);
        printUsage(turn2.typed());
    }

    /**
     * One call with the model's own defaults, one with chat_template_kwargs
     * {"enable_thinking": false} -- prints the typed choices[0].message.thought
     * field per the OpenAPI spec, and the raw top-level "thought" the live
     * server actually sends (see README Friction).
     */
    private static void thinkingDemo(InferenceApi inference, boolean useDefaults) throws ApiException {
        List<PostV1ChatCompletionsRequestMessagesInner> messages =
                List.of(userMessage("In one sentence, why does local inference matter?"));
        Map<String, Object> kwargs = useDefaults ? null : disableThinking();

        ChatResult result = chat(inference, messages, 500, kwargs);
        PostV1ChatCompletions200ResponseChoicesInnerMessage message = result.typed().getChoices().get(0).getMessage();

        String label = useDefaults ? "thinking (defaults)" : "thinking (enable_thinking=false)";
        System.out.println(label + " -- typed message.thought: " + describe(message.getThought()));
        System.out.println(label + " -- raw top-level thought: " + describe(result.rawThought()));
        System.out.println(label + " -- answer: " + message.getContent());
        printUsage(result.typed());
    }

    /**
     * Executes the chat completion once via the SDK's own OkHttp call, then
     * parses the body TWICE from the same string: once through the SDK's
     * typed Gson (matches the OpenAPI spec's nested message.thought), once
     * as a raw JsonObject (surfaces the top-level "thought" the spec doesn't
     * declare). One network round trip either way.
     */
    private static ChatResult chat(InferenceApi inference, List<PostV1ChatCompletionsRequestMessagesInner> messages,
            int maxTokens, Map<String, Object> chatTemplateKwargs) throws ApiException {
        PostV1ChatCompletionsRequest request = new PostV1ChatCompletionsRequest()
                .model("gemma4")
                .messages(messages)
                .maxTokens(maxTokens);
        if (chatTemplateKwargs != null) {
            request.setChatTemplateKwargs(chatTemplateKwargs);
        }

        Call call = inference.postV1ChatCompletionsCall(request, null);
        try (Response httpResponse = call.execute()) {
            String body = httpResponse.body() != null ? httpResponse.body().string() : "";
            if (!httpResponse.isSuccessful()) {
                throw new ApiException(httpResponse.code(), "lem returned " + httpResponse.code() + ": " + body);
            }
            PostV1ChatCompletions200Response typed =
                    JSON.getGson().fromJson(body, PostV1ChatCompletions200Response.class);
            JsonObject raw = JsonParser.parseString(body).getAsJsonObject();
            String rawThought = raw.has("thought") && !raw.get("thought").isJsonNull()
                    ? raw.get("thought").getAsString()
                    : null;
            return new ChatResult(typed, rawThought);
        } catch (IOException e) {
            throw new ApiException(e);
        }
    }

    private record ChatResult(PostV1ChatCompletions200Response typed, String rawThought) {
    }

    private static String firstContent(ChatResult result) {
        return result.typed().getChoices().get(0).getMessage().getContent();
    }

    private static PostV1ChatCompletionsRequestMessagesInner userMessage(String content) {
        return new PostV1ChatCompletionsRequestMessagesInner().role("user").content(content);
    }

    private static PostV1ChatCompletionsRequestMessagesInner assistantMessage(String content) {
        return new PostV1ChatCompletionsRequestMessagesInner().role("assistant").content(content);
    }

    private static Map<String, Object> disableThinking() {
        return Map.of("enable_thinking", false);
    }

    private static void printUsage(PostV1ChatCompletions200Response response) {
        PostV1ChatCompletions200ResponseUsage usage = response.getUsage();
        if (usage == null) {
            System.out.println("usage: (none reported)");
            return;
        }
        System.out.printf("usage: %d prompt + %d completion tokens%n",
                usage.getPromptTokens(), usage.getCompletionTokens());
    }

    private static String describe(String value) {
        return (value == null || value.isEmpty()) ? "(absent)" : value;
    }

    private static boolean isConnectionRefused(ApiException e) {
        Throwable cause = e.getCause();
        return cause instanceof ConnectException;
    }
}
