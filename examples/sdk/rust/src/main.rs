// SPDX-Licence-Identifier: EUPL-1.2

// A minimal Rust app using the GENERATED lem SDK (task sdk → build/sdk/rust)
// to chat with a local gemma4 serve — the OpenAPI standard doing the client
// work.

use lem_sdk::apis::configuration::Configuration;
use lem_sdk::apis::inference_api;
use lem_sdk::models::{PostV1ChatCompletionsRequest, PostV1ChatCompletionsRequestMessagesInner};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base = std::env::var("LEM_BASE_URL").unwrap_or_else(|_| "http://localhost:36911".into());
    let config = Configuration {
        base_path: base,
        ..Configuration::default()
    };

    let models = inference_api::get_v1_models(&config).await?;
    for m in &models.data {
        println!("serving: {}", m.id);
    }

    let message = PostV1ChatCompletionsRequestMessagesInner {
        role: "user".into(),
        content: Some(Some(serde_json::json!(
            "In one sentence, why does local inference matter?"
        ))),
    };
    let request = PostV1ChatCompletionsRequest {
        model: "gemma4".into(),
        messages: vec![message],
        max_tokens: Some(96),
        temperature: None,
        top_p: None,
        stream: None,
        chat_template_kwargs: Some(serde_json::json!({ "enable_thinking": false })),
    };

    let response = inference_api::post_v1_chat_completions(&config, request).await?;
    let choice = response.choices.first().ok_or("no choices in response")?;
    println!(
        "gemma4: {}",
        choice.message.content.as_deref().unwrap_or_default()
    );
    if let Some(usage) = &response.usage {
        println!(
            "usage: {} prompt + {} completion tokens",
            usage.prompt_tokens.unwrap_or_default(),
            usage.completion_tokens.unwrap_or_default()
        );
    }
    Ok(())
}
