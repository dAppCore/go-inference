// SPDX-Licence-Identifier: EUPL-1.2

// A Kotlin app using the GENERATED lem SDK (task sdk -> build/sdk/kotlin) to
// exercise a local gemma4 serve past hello-world: list the served models, a
// two-turn conversation that proves the model remembers turn 1, and a
// thinking-channel demo — the OpenAPI standard doing the client work.
package re.dappco.lem.example

import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import re.dappco.lem.apis.InferenceApi
import re.dappco.lem.infrastructure.Serializer
import re.dappco.lem.models.PostV1ChatCompletions200Response
import re.dappco.lem.models.PostV1ChatCompletionsRequest
import re.dappco.lem.models.PostV1ChatCompletionsRequestMessagesInner
import java.net.ConnectException
import java.time.Duration
import kotlin.system.exitProcess

// model is cosmetic on a single-model serve — the loaded model answers every name
private const val MODEL = "gemma4"

fun main() {
    val base = System.getenv("LEM_BASE_URL") ?: "http://localhost:36911"

    // Requests queue through lem's serial scheduler, so give the client
    // generous timeouts rather than trust OkHttp's 10s default.
    val http = OkHttpClient.Builder()
        .connectTimeout(Duration.ofSeconds(10))
        .readTimeout(Duration.ofSeconds(150))
        .writeTimeout(Duration.ofSeconds(30))
        .callTimeout(Duration.ofSeconds(180))
        .build()
    val api = InferenceApi(base, http)

    try {
        listModels(api)
        twoTurnConversation(api)
        thinkingDemo(api, base, http)
    } catch (e: ConnectException) {
        System.err.println("cannot reach $base — start `lem serve` first (connection refused: ${e.message})")
        exitProcess(1)
    }
}

private fun listModels(api: InferenceApi) {
    println("== models ==")
    val models = api.getV1Models()
    for (m in models.`data`) println("serving: ${m.id}")
}

/** One chat call through the typed client; prints usage tokens as it goes. */
private fun chat(
    api: InferenceApi,
    history: List<PostV1ChatCompletionsRequestMessagesInner>,
    maxTokens: Int,
    kwargs: Map<String, Any>? = null,
): PostV1ChatCompletions200Response {
    val request = PostV1ChatCompletionsRequest(
        messages = history,
        model = MODEL,
        chatTemplateKwargs = kwargs,
        maxTokens = maxTokens,
    )
    val response = api.postV1ChatCompletions(request)
    response.usage?.let { usage ->
        println("usage: ${usage.promptTokens} prompt + ${usage.completionTokens} completion tokens")
    }
    return response
}

private fun twoTurnConversation(api: InferenceApi) {
    println("\n== two-turn conversation (memory) ==")
    val history = mutableListOf<PostV1ChatCompletionsRequestMessagesInner>()

    history += PostV1ChatCompletionsRequestMessagesInner("user", "My favourite number is 42. Remember it.")
    val turn1 = chat(api, history, maxTokens = 128, kwargs = mapOf("enable_thinking" to false))
    val reply1 = turn1.choices.first().message.content.orEmpty()
    println("turn 1: $reply1")
    history += PostV1ChatCompletionsRequestMessagesInner("assistant", reply1)

    history += PostV1ChatCompletionsRequestMessagesInner("user", "What's my favourite number? Answer with just the number.")
    val turn2 = chat(api, history, maxTokens = 64, kwargs = mapOf("enable_thinking" to false))
    val reply2 = turn2.choices.first().message.content.orEmpty()
    println("turn 2: $reply2")
    println(
        if ("42" in reply2) "turn 2 proves memory: contains \"42\" from turn 1 (history was resent)"
        else "turn 2 did NOT echo \"42\" — the served model may not have honoured history"
    )
}

private fun thinkingDemo(api: InferenceApi, base: String, http: OkHttpClient) {
    println("\n== thinking demo ==")
    val prompt = listOf(PostV1ChatCompletionsRequestMessagesInner("user", "Say hello in one short sentence."))

    val defaultResp = chat(api, prompt, maxTokens = 200)
    val typedThought = defaultResp.choices.first().message.thought
    println("thought (typed, choices[0].message.thought, defaults): ${typedThought ?: "null"}")
    if (typedThought == null) {
        // See README Friction: the live server puts `thought` on the TOP LEVEL
        // of the response object, but the OpenAPI spec (and so the generated
        // model) nests it under choices[].message instead — it's never there
        // to deserialise. One raw call, reusing the SDK's own Moshi instance,
        // shows the value genuinely exists on the wire.
        val raw = rawThought(base, http, prompt, maxTokens = 200)
        println("thought (raw top-level 'thought' field — NOT exposed by the generated client): ${raw ?: "<absent>"}")
    }

    val noThinkResp = chat(api, prompt, maxTokens = 64, kwargs = mapOf("enable_thinking" to false))
    println("thought (typed, choices[0].message.thought, enable_thinking=false): ${noThinkResp.choices.first().message.thought ?: "null"}")
}

private fun rawThought(
    base: String,
    http: OkHttpClient,
    messages: List<PostV1ChatCompletionsRequestMessagesInner>,
    maxTokens: Int,
): String? {
    val request = PostV1ChatCompletionsRequest(messages = messages, model = MODEL, maxTokens = maxTokens)
    val bodyJson = Serializer.moshi.adapter(PostV1ChatCompletionsRequest::class.java).toJson(request)
    val call = http.newCall(
        Request.Builder()
            .url("$base/v1/chat/completions")
            .post(bodyJson.toRequestBody("application/json".toMediaType()))
            .build()
    )
    val body = call.execute().use { it.body?.string() }.orEmpty()
    @Suppress("UNCHECKED_CAST")
    val parsed = Serializer.moshi.adapter(Map::class.java).fromJson(body) as? Map<String, Any>
    return parsed?.get("thought") as? String
}
