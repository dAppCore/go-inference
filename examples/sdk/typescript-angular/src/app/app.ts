// SPDX-Licence-Identifier: EUPL-1.2

// gemma4 from Angular via the GENERATED lem SDK (task sdk →
// build/sdk/typescript-angular): the InferenceService is a normal
// @Injectable — list the served models, run a two-turn conversation that
// proves memory, and show the thinking channel typed on the response root.

import { Component, inject, signal } from '@angular/core';
import { InferenceService, PostV1ChatCompletionsRequest } from '@lethean/lem-sdk-angular';
import { firstValueFrom } from 'rxjs';

@Component({
  selector: 'app-root',
  template: `
    <main style="font-family: system-ui; max-width: 44rem; margin: 2rem auto; line-height: 1.5">
      <h1>Gemma 4 from Angular</h1>
      <p id="model">serving: {{ model() }}</p>
      <p id="turn1"><b>turn 1:</b> {{ turn1() }}</p>
      <p id="turn2"><b>turn 2 (memory):</b> {{ turn2() }}</p>
      <p id="answer"><b>thinking answer:</b> {{ answer() }}</p>
      <p id="thought" style="opacity: 0.6"><b>thought:</b> {{ thought() }}</p>
      <p id="usage" style="opacity: 0.6">{{ usage() }}</p>
      <p id="error" style="color: #c66">{{ error() }}</p>
    </main>
  `,
})
export class App {
  private readonly inference = inject(InferenceService);

  protected readonly model = signal('…');
  protected readonly turn1 = signal('…');
  protected readonly turn2 = signal('…');
  protected readonly answer = signal('…');
  protected readonly thought = signal('…');
  protected readonly usage = signal('');
  protected readonly error = signal('');

  constructor() {
    void this.run();
  }

  private ask(request: PostV1ChatCompletionsRequest) {
    return firstValueFrom(
      this.inference.postV1ChatCompletions({ postV1ChatCompletionsRequest: request }),
    );
  }

  private async run(): Promise<void> {
    try {
      const models = await firstValueFrom(this.inference.getV1Models());
      this.model.set(models.data.map((m) => m.id).join(', '));

      const history = [
        { role: 'user', content: 'My favourite colour is teal. Reply with one short sentence.' },
      ];
      const r1 = await this.ask({
        model: 'gemma4',
        messages: history,
        max_tokens: 96,
        chat_template_kwargs: { enable_thinking: false },
      });
      const a1 = r1.choices[0]?.message.content ?? '';
      this.turn1.set(a1);

      const r2 = await this.ask({
        model: 'gemma4',
        messages: [
          ...history,
          { role: 'assistant', content: a1 },
          { role: 'user', content: 'What is my favourite colour? Answer with just the colour.' },
        ],
        max_tokens: 96,
        chat_template_kwargs: { enable_thinking: false },
      });
      this.turn2.set(r2.choices[0]?.message.content ?? '');

      // Thinking ON (the gemma4 default): the reasoning arrives TYPED on the
      // response root, the answer stays clean.
      const rt = await this.ask({
        model: 'gemma4',
        messages: [{ role: 'user', content: 'Is 17 prime? One word.' }],
        max_tokens: 512,
      });
      this.answer.set(rt.choices[0]?.message.content ?? '');
      this.thought.set((rt.thought ?? '(none)').slice(0, 120));
      this.usage.set(
        `usage: ${rt.usage?.prompt_tokens} prompt + ${rt.usage?.completion_tokens} completion tokens`,
      );
    } catch (err) {
      this.error.set(`cannot reach lem — start one with: lem serve --model <snapshot> (${err})`);
    }
  }
}
