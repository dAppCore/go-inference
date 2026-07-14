# SPDX-Licence-Identifier: EUPL-1.2

# A Ruby app using the GENERATED lem SDK (task sdk -> build/sdk/ruby) to chat
# with a local gemma4 serve -- the OpenAPI standard doing the client work.
# Past hello-world: lists the served models, runs a two-turn conversation
# that proves the model remembers turn 1, and demonstrates the thinking
# channel on and off.

require "uri"
require "lem_sdk"

base_url = ENV.fetch("LEM_BASE_URL", "http://localhost:36911")
target = URI.parse(base_url)

config = LemSdk::Configuration.new
config.scheme = target.scheme
config.host = target.port ? "#{target.host}:#{target.port}" : target.host
config.timeout = 240 # requests queue through a serial scheduler -- be generous

inference = LemSdk::InferenceApi.new(LemSdk::ApiClient.new(config))

def print_usage(usage)
  return unless usage

  puts "  usage: #{usage.prompt_tokens} prompt + #{usage.completion_tokens} completion tokens"
end

def print_reply(label, message, finish_reason)
  puts "#{label}: #{message.content}"
  puts "  (finish_reason: #{finish_reason}, content empty -- ran out of max_tokens)" if message.content.to_s.empty?
end

def user_message(text)
  LemSdk::PostV1ChatCompletionsRequestMessagesInner.new(role: "user", content: text)
end

def assistant_message(text)
  LemSdk::PostV1ChatCompletionsRequestMessagesInner.new(role: "assistant", content: text)
end

# --- (a) list the served models ---
begin
  models = inference.get_v1_models
rescue LemSdk::ApiError => e
  warn "Could not reach lem serve at #{base_url}: #{e.message.lines.first.strip}"
  warn "Start it first: lem serve --model <path>  (or: lem tui -> Service tab -> enter)"
  exit 1
end

models.data.each { |m| puts "serving: #{m.id}" }

# --- (b) two-turn conversation, proving memory by resending history ---
# gemma4 sometimes reasons before a plain acknowledgement too, so leave enough
# budget that a "length" truncation can't be mistaken for a forgotten fact.
history = [user_message("My favourite colour is teal. Remember it for later.")]

turn1 = inference.post_v1_chat_completions(
  LemSdk::PostV1ChatCompletionsRequest.new(
    model: "gemma4",
    messages: history,
    max_tokens: 400,
    chat_template_kwargs: { "enable_thinking" => false },
  ),
)
reply1 = turn1.choices[0].message
print_reply("turn 1", reply1, turn1.choices[0].finish_reason)
print_usage(turn1.usage)

history << assistant_message(reply1.content)
history << user_message("What colour did I just tell you? Answer in one word.")

turn2 = inference.post_v1_chat_completions(
  LemSdk::PostV1ChatCompletionsRequest.new(
    model: "gemma4",
    messages: history,
    max_tokens: 400,
    chat_template_kwargs: { "enable_thinking" => false },
  ),
)
reply2 = turn2.choices[0].message
print_reply("turn 2 (proves memory)", reply2, turn2.choices[0].finish_reason)
print_usage(turn2.usage)

# --- (c) thinking channel: default (on) vs explicitly disabled ---
# "Why" questions reliably trigger a long reasoning pass on this model, so
# this pair needs a much bigger budget than the memory turns above.
question = [user_message("In one sentence, why does local inference matter?")]

thinking_on = inference.post_v1_chat_completions(
  LemSdk::PostV1ChatCompletionsRequest.new(model: "gemma4", messages: question, max_tokens: 700),
)
choice_on = thinking_on.choices[0].message
puts "thought (default, thinking on): #{choice_on.thought || "(nil -- see README Friction: thought is on the wrong node in the generated client)"}"
print_reply("answer", choice_on, thinking_on.choices[0].finish_reason)
print_usage(thinking_on.usage)

thinking_off = inference.post_v1_chat_completions(
  LemSdk::PostV1ChatCompletionsRequest.new(
    model: "gemma4",
    messages: question,
    max_tokens: 700,
    chat_template_kwargs: { "enable_thinking" => false },
  ),
)
choice_off = thinking_off.choices[0].message
puts "thought (enable_thinking: false): #{choice_off.thought || "(nil -- same as above)"}"
print_reply("answer", choice_off, thinking_off.choices[0].finish_reason)
print_usage(thinking_off.usage)
