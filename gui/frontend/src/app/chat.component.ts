// SPDX-Licence-Identifier: EUPL-1.2

import {
  Component,
  CUSTOM_ELEMENTS_SCHEMA,
  computed,
  resource,
  signal,
  type ResourceStreamItem,
} from '@angular/core';
import { FormsModule } from '@angular/forms';

interface ChatTurn {
  role: 'user' | 'assistant';
  content: string;
  thought?: string;
}

interface StreamState {
  content: string;
  thought: string;
}

/**
 * Chat with the local lem serve, built on Angular's AI design patterns
 * (angular.dev/ai/design-patterns):
 *
 *  1. signal-triggered requests — `submitted` changes only on send, never on
 *     keystrokes; the resource's params track it.
 *  2. derived template data — history is a signal the stream finalises into;
 *     the live turn renders from the resource value while it accumulates.
 *  3. status UX — isLoading()/hasValue()/reload() drive the spinner, the
 *     error state and retry.
 *  4. streaming — resource({stream}) yields ResourceStreamItem updates per
 *     SSE chunk; lem's chunks carry the answer delta AND the typed thought
 *     channel, so reasoning renders live and separately.
 */
@Component({
  selector: 'lem-chat',
  standalone: true,
  imports: [FormsModule],
  schemas: [CUSTOM_ELEMENTS_SCHEMA],
  templateUrl: './chat.component.html',
  styleUrl: './chat.component.css',
})
export class ChatComponent {
  readonly baseURL = signal('http://127.0.0.1:36911');
  draft = '';

  readonly history = signal<ChatTurn[]>([]);
  private readonly submitted = signal<{ prompt: string; seq: number } | null>(null);
  private seq = 0;

  readonly reply = resource({
    params: () => this.submitted(),
    stream: async ({ params, abortSignal }) => {
      const out = signal<ResourceStreamItem<StreamState>>({ value: { content: '', thought: '' } });
      if (!params) {
        return out;
      }
      void this.pump(params.prompt, abortSignal, out);
      return out;
    },
  });

  readonly liveTurn = computed<StreamState>(() =>
    this.reply.hasValue() ? this.reply.value() : { content: '', thought: '' },
  );

  send(): void {
    const prompt = this.draft.trim();
    if (prompt === '' || this.reply.isLoading()) {
      return;
    }
    this.history.update((h) => [...h, { role: 'user', content: prompt }]);
    this.draft = '';
    this.submitted.set({ prompt, seq: ++this.seq });
  }

  retry(): void {
    this.reply.reload();
  }

  /** Streams one completion from lem's OpenAI route into the resource signal. */
  private async pump(
    prompt: string,
    abortSignal: AbortSignal,
    out: ReturnType<typeof signal<ResourceStreamItem<StreamState>>>,
  ): Promise<void> {
    let content = '';
    let thought = '';
    try {
      const messages = [
        ...this.history()
          .filter((t) => t.content !== '')
          .map((t) => ({ role: t.role, content: t.content })),
      ];
      const resp = await fetch(this.baseURL() + '/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: abortSignal,
        body: JSON.stringify({ model: 'lem', stream: true, max_tokens: 1024, messages }),
      });
      if (!resp.ok || !resp.body) {
        out.set({ error: new Error(`lem serve returned ${resp.status} — is a model loaded?`) });
        return;
      }
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      for (;;) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        buffer += decoder.decode(value, { stream: true });
        for (;;) {
          const split = buffer.indexOf('\n\n');
          if (split < 0) {
            break;
          }
          const frame = buffer.slice(0, split);
          buffer = buffer.slice(split + 2);
          for (const line of frame.split('\n')) {
            if (!line.startsWith('data:')) {
              continue;
            }
            const payload = line.slice(5).trim();
            if (payload === '' || payload === '[DONE]') {
              continue;
            }
            const chunk = JSON.parse(payload) as {
              choices?: { delta?: { content?: string } }[];
              thought?: string;
            };
            const delta = chunk.choices?.[0]?.delta?.content ?? '';
            if (delta !== '') {
              content += delta;
            }
            if (chunk.thought) {
              thought += chunk.thought;
            }
            out.set({ value: { content, thought } });
          }
        }
      }
      this.history.update((h) => [...h, { role: 'assistant', content, thought }]);
      out.set({ value: { content: '', thought: '' } });
    } catch (err) {
      if (!abortSignal.aborted) {
        out.set({ error: err instanceof Error ? err : new Error(String(err)) });
      }
    }
  }
}
