#!/usr/bin/env perl
# SPDX-Licence-Identifier: EUPL-1.2

# A Perl app using the GENERATED lem SDK (task sdk -> build/sdk/perl) to
# chat with a local gemma4 serve -- the OpenAPI standard doing the client
# work. Goes past hello-world: lists the served models, holds a two-turn
# conversation (turn 2 resends the history so the model can prove it
# remembered turn 1), and demonstrates the thinking channel both on
# (the default) and off (chat_template_kwargs => {enable_thinking => false}).

use strict;
use warnings;
use utf8;
use feature 'say';
binmode(STDOUT, ':encoding(UTF-8)');

use FindBin qw($RealBin);
use lib "$RealBin/local/lib/perl5";        # cpanm -l local (Log::Any, URI::Query)
use lib "$RealBin/../../../build/sdk/perl/lib"; # generated client (task sdk, gitignored)

use JSON;
use LemSDK::InferenceApi;
use LemSDK::Object::PostV1ChatCompletionsRequest;
use LemSDK::Object::PostV1ChatCompletionsRequestMessagesInner;

my $base = $ENV{LEM_BASE_URL} // 'http://localhost:36911';

# Requests queue through a serial scheduler behind a single-model serve --
# a generous client timeout stops a slow neighbour timing your request out.
my $api = LemSDK::InferenceApi->new(base_url => $base, http_timeout => 180);

# Every SDK call goes through here so a refused connection produces one
# clear instruction instead of a raw LWP::UserAgent stack trace.
sub call_or_die {
    my ($what, $code) = @_;
    my $result = eval { $code->() };
    if (my $err = $@) {
        if ($err =~ /connection refused|can'?t connect/i) {
            die "$what: could not reach lem at $base -- start a serve first:\n  lem serve\n";
        }
        die "$what: $err";
    }
    return $result;
}

sub user_msg {
    my ($content) = @_;
    return LemSDK::Object::PostV1ChatCompletionsRequestMessagesInner->new(
        role => 'user', content => $content);
}

sub assistant_msg {
    my ($content) = @_;
    return LemSDK::Object::PostV1ChatCompletionsRequestMessagesInner->new(
        role => 'assistant', content => $content);
}

sub print_usage {
    my ($resp) = @_;
    my $usage = $resp->usage or return;
    printf "  usage: %d prompt + %d completion tokens\n",
        $usage->prompt_tokens, $usage->completion_tokens;
}

# One chat-completion call: builds the request, calls the API, prints
# usage, and hands back the reply message so the caller can inspect
# content/thought or feed it back in as history.
sub chat {
    my ($what, $messages, %opts) = @_;
    my $req = LemSDK::Object::PostV1ChatCompletionsRequest->new(
        model      => 'gemma4',
        messages   => $messages,
        max_tokens => $opts{max_tokens} // 256,
        (exists $opts{chat_template_kwargs}
            ? (chat_template_kwargs => $opts{chat_template_kwargs}) : ()),
    );
    my $resp = call_or_die($what, sub {
        $api->post_v1_chat_completions(post_v1_chat_completions_request => $req);
    });
    print_usage($resp);
    return $resp->choices->[0]->message;
}

# (a) list the served models
my $models = call_or_die('list models', sub { $api->get_v1_models });
for my $m (@{ $models->data }) {
    say "serving: " . $m->id;
}

# (b) two-turn conversation -- turn 2 resends the full history, so an
# answer that recalls "teal" proves the model was given turn 1's context,
# not just re-guessing. Thinking is switched off here: it's a memory demo,
# not a reasoning one, and a thinking pass can burn the whole max_tokens
# budget on the reasoning channel before any reply content gets written
# (see "the thought field is invisible" in the README's Friction section).
my %no_thinking = (chat_template_kwargs => { enable_thinking => JSON::false });

my $turn1_user = user_msg('My favourite colour is teal. Remember it for later.');
my $turn1 = chat('turn 1', [$turn1_user], %no_thinking);
say "turn 1: " . $turn1->content;

my $turn2 = chat('turn 2', [
    $turn1_user,
    assistant_msg($turn1->content),
    user_msg('What did I say my favourite colour was?'),
], %no_thinking);
say "turn 2: " . $turn2->content;

# (c) thinking demo -- defaults leave the thinking channel on; setting
# chat_template_kwargs => {enable_thinking => JSON::false} turns it off.
my $riddle = 'A farmer has 17 sheep, all but 9 die. How many are left?';

my $thinking_on = chat('thinking (default)', [user_msg($riddle)]);
say 'thinking on, thought:  ' . ($thinking_on->thought // '(no thought field returned)');
say 'thinking on, content:  ' . $thinking_on->content;

my $thinking_off = chat('thinking (disabled)', [user_msg($riddle)],
    chat_template_kwargs => { enable_thinking => JSON::false });
say 'thinking off, thought: ' . ($thinking_off->thought // '(no thought field returned)');
say 'thinking off, content: ' . $thinking_off->content;
