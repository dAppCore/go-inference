<?php

// SPDX-Licence-Identifier: EUPL-1.2

// A minimal PHP app using the GENERATED lem SDK (task sdk -> build/sdk/php)
// to chat with a local gemma4 serve -- the OpenAPI standard doing the client
// work. Goes past hello-world: lists the served models, runs a two-turn
// conversation that proves the model remembers turn one, demonstrates the
// typed `thought` field with and without chat_template_kwargs, and prints
// usage tokens after every call.

declare(strict_types=1);

require __DIR__ . '/vendor/autoload.php';

use GuzzleHttp\Client;
use Lethean\LemSdk\Api\InferenceApi;
use Lethean\LemSdk\ApiException;
use Lethean\LemSdk\Configuration;
use Lethean\LemSdk\Model\PostV1ChatCompletions200ResponseUsage;
use Lethean\LemSdk\Model\PostV1ChatCompletionsRequest;
use Lethean\LemSdk\Model\PostV1ChatCompletionsRequestMessagesInner;

function printUsage(?PostV1ChatCompletions200ResponseUsage $usage): void
{
    if ($usage === null) {
        return;
    }
    printf(
        "usage: %d prompt + %d completion tokens\n",
        $usage->getPromptTokens(),
        $usage->getCompletionTokens(),
    );
}

$base = getenv('LEM_BASE_URL') ?: 'http://localhost:36911';

$config = (new Configuration())->setHost($base);
// Requests queue through lem's serial scheduler -- a busy driver means a
// slow reply, not a stuck one, so the client timeout needs real headroom.
$client = new Client(['timeout' => 120]);
$inference = new InferenceApi($client, $config);

try {
    $models = $inference->getV1Models();
    foreach ($models->getData() as $model) {
        echo "serving: {$model->getId()}\n";
    }

    // --- two-turn conversation: turn 2 proves the model remembered turn 1 ---
    $history = [
        new PostV1ChatCompletionsRequestMessagesInner([
            'role' => 'user',
            'content' => 'My favourite colour is teal. Remember it.',
        ]),
    ];
    $turn1 = $inference->postV1ChatCompletions(new PostV1ChatCompletionsRequest([
        'model' => 'gemma4',
        'messages' => $history,
        'max_tokens' => 64,
        'chat_template_kwargs' => (object) ['enable_thinking' => false],
    ]));
    $answer1 = $turn1->getChoices()[0]->getMessage()->getContent();
    echo "turn 1: {$answer1}\n";
    printUsage($turn1->getUsage());

    $history[] = new PostV1ChatCompletionsRequestMessagesInner([
        'role' => 'assistant',
        'content' => $answer1,
    ]);
    $history[] = new PostV1ChatCompletionsRequestMessagesInner([
        'role' => 'user',
        'content' => 'What colour did I just tell you, one word?',
    ]);
    $turn2 = $inference->postV1ChatCompletions(new PostV1ChatCompletionsRequest([
        'model' => 'gemma4',
        'messages' => $history,
        'max_tokens' => 64,
        'chat_template_kwargs' => (object) ['enable_thinking' => false],
    ]));
    $answer2 = $turn2->getChoices()[0]->getMessage()->getContent();
    echo "turn 2 (should recall teal): {$answer2}\n";
    printUsage($turn2->getUsage());

    // --- thinking demo: default request vs. enable_thinking disabled ---
    //
    // FRICTION: the OpenAPI spec declares `thought` nested under
    // choices[0].message (build/sdk/openapi.json, matching
    // go/engine/driver/inference.go:138), so that's where the generated
    // model exposes it -- and where the sibling go/python/rust/typescript
    // examples all read it. But the live wire response actually puts
    // `thought` at the TOP LEVEL of the response body
    // (go/serving/provider/openai/openai.go:133, ChatCompletionResponse.Thought)
    // -- a real spec/implementation mismatch, not a client bug. The typed
    // getThought() below reliably comes back null even with thinking on.
    // Proving the model actually reasoned needs one raw decode alongside
    // the typed call; see README Friction for the full writeup.
    $questionMessages = [new PostV1ChatCompletionsRequestMessagesInner([
        'role' => 'user',
        'content' => 'What is 17 times 24? Work it out.',
    ])];
    $question = [
        'model' => 'gemma4',
        'messages' => $questionMessages,
        'max_tokens' => 256,
    ];

    $thinkingOn = $inference->postV1ChatCompletions(new PostV1ChatCompletionsRequest($question));
    $thought = $thinkingOn->getChoices()[0]->getMessage()->getThought();
    echo 'thought via typed client: ' . ($thought ?? '(null -- see Friction, it is really top-level)') . "\n";

    $rawResponse = $client->post($base . '/v1/chat/completions', ['json' => $question]);
    $rawBody = json_decode((string) $rawResponse->getBody(), true);
    echo 'thought via raw decode: ' . ($rawBody['thought'] ?? '(none)') . "\n";
    echo 'answer: ' . $thinkingOn->getChoices()[0]->getMessage()->getContent() . "\n";
    printUsage($thinkingOn->getUsage());

    $thinkingOff = $inference->postV1ChatCompletions(new PostV1ChatCompletionsRequest([
        'model' => 'gemma4',
        'messages' => $questionMessages,
        'max_tokens' => 256,
        'chat_template_kwargs' => (object) ['enable_thinking' => false],
    ]));
    $thoughtOff = $thinkingOff->getChoices()[0]->getMessage()->getThought();
    echo 'thought (enable_thinking=false): ' . ($thoughtOff ?? '(none, as expected)') . "\n";
    echo 'answer: ' . $thinkingOff->getChoices()[0]->getMessage()->getContent() . "\n";
    printUsage($thinkingOff->getUsage());
} catch (ApiException $e) {
    // FRICTION: the generated client swallows GuzzleHttp\Exception\ConnectException
    // and rewraps it as ApiException before it ever reaches caller code (see
    // build/sdk/php/src/Api/InferenceApi.php, the getV1ModelsWithHttpInfo-style
    // methods) -- catching ConnectException here never fires. A connection
    // failure is instead an ApiException with no HTTP response at all: no
    // response headers, code 0 (curl error codes aren't HTTP statuses).
    if ($e->getResponseHeaders() === null) {
        fwrite(STDERR, "Could not reach {$base} -- start a serve first: lem serve --model <path>\n");
        exit(1);
    }
    throw $e;
}
