// SPDX-Licence-Identifier: EUPL-1.2

import { Component, CUSTOM_ELEMENTS_SCHEMA, computed, inject } from '@angular/core';
import { LemService } from './lem.service';

/**
 * LEM Desktop — the design pack's native window shell (titlebar, rail, stat
 * tiles, panels, statusbar) fed by the real Wails bindings. The <lthn-*>
 * elements are the pack's Lit components; CUSTOM_ELEMENTS_SCHEMA lets Angular
 * host them while signals/computed drive the data — Lit for the system,
 * Angular for the features.
 */
@Component({
  selector: 'lem-root',
  standalone: true,
  schemas: [CUSTOM_ELEMENTS_SCHEMA],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  readonly lem = inject(LemService);

  readonly snap = this.lem.snapshot;

  readonly activeTraining = computed(() => this.snap().training.filter((t) => t.status === 'running').length);
  readonly firstRun = computed(() => this.snap().training[0]);

  readonly trainingColumns = JSON.stringify([
    { key: 'model', label: 'Model' },
    { key: 'runId', label: 'Run', type: 'mono' },
    { key: 'iteration', label: 'Iter', type: 'num' },
    { key: 'pct', label: '%', type: 'num' },
    { key: 'loss', label: 'Loss', type: 'num' },
    { key: 'status', label: 'State', type: 'status' },
  ]);
  readonly trainingRows = computed(() => JSON.stringify(this.snap().training));

  readonly modelColumns = JSON.stringify([
    { key: 'name', label: 'Model' },
    { key: 'tag', label: 'Tag', type: 'mono' },
    { key: 'accuracy', label: 'Accuracy', type: 'num' },
    { key: 'iterations', label: 'Iters', type: 'num' },
    { key: 'status', label: 'State', type: 'status' },
  ]);
  readonly modelRows = computed(() => JSON.stringify(this.snap().models));

  serviceVariant(running: boolean): string {
    return running ? 'ok' : 'idle';
  }
}
