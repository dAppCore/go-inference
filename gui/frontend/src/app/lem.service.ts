// SPDX-Licence-Identifier: EUPL-1.2

import { Injectable, signal } from '@angular/core';

/* The Go-side wire shapes (gui/tray.go, dashboard.go, docker.go). */
export interface TrainingRow {
  model: string;
  runId: string;
  status: string;
  iteration: number;
  totalIters: number;
  pct: number;
  loss: number;
}
export interface GenerationStats {
  goldenCompleted: number;
  goldenTarget: number;
  goldenPct: number;
  expansionCompleted: number;
  expansionTarget: number;
  expansionPct: number;
}
export interface ModelRow {
  name: string;
  tag: string;
  accuracy: number;
  iterations: number;
  status: string;
}
export interface ContainerStatus {
  name: string;
  image: string;
  status: string;
  health: string;
  ports: string;
  running: boolean;
}
export interface TraySnapshot {
  stackRunning: boolean;
  containedRunning: boolean;
  agentRunning: boolean;
  agentTask: string;
  training: TrainingRow[];
  generation: GenerationStats;
  models: ModelRow[];
  dockerServices: number;
}

/* The Wails v3 binding surface the embedded page sees. */
interface WailsBindings {
  main: {
    TrayService: {
      GetSnapshot(): Promise<TraySnapshot>;
      StartStack(): Promise<void>;
      StopStack(): Promise<void>;
      StartAgent(): Promise<void>;
      StopAgent(): Promise<void>;
    };
    DockerService: {
      GetStatus(): Promise<{ running: boolean; services: Record<string, ContainerStatus> }>;
    };
  };
}

const emptySnapshot: TraySnapshot = {
  stackRunning: false,
  containedRunning: false,
  agentRunning: false,
  agentTask: '',
  training: [],
  generation: {
    goldenCompleted: 0,
    goldenTarget: 0,
    goldenPct: 0,
    expansionCompleted: 0,
    expansionTarget: 0,
    expansionPct: 0,
  },
  models: [],
  dockerServices: 0,
};

/**
 * The Wails-bindings bridge as Angular signals: one poll loop feeds the whole
 * dashboard. Outside the Wails webview (ng serve in a plain browser) the
 * bindings are absent — state stays empty and `bridge` reports it honestly.
 */
@Injectable({ providedIn: 'root' })
export class LemService {
  readonly snapshot = signal<TraySnapshot>(emptySnapshot);
  readonly services = signal<ContainerStatus[]>([]);
  readonly bridge = signal<'connected' | 'absent'>('absent');
  readonly error = signal('');
  readonly updatedAt = signal('');

  private get bindings(): WailsBindings['main'] | null {
    const w = window as unknown as { go?: WailsBindings };
    return w.go?.main ?? null;
  }

  constructor() {
    void this.refresh();
    setInterval(() => void this.refresh(), 10_000);
  }

  async refresh(): Promise<void> {
    const b = this.bindings;
    if (!b) {
      this.bridge.set('absent');
      return;
    }
    this.bridge.set('connected');
    try {
      this.snapshot.set(await b.TrayService.GetSnapshot());
      const docker = await b.DockerService.GetStatus();
      this.services.set(Object.values(docker.services ?? {}));
      this.error.set('');
      this.updatedAt.set(new Date().toLocaleTimeString());
    } catch (err) {
      this.error.set(String(err));
    }
  }

  async toggleStack(): Promise<void> {
    const b = this.bindings;
    if (!b) return;
    try {
      if (this.snapshot().stackRunning) {
        await b.TrayService.StopStack();
      } else {
        await b.TrayService.StartStack();
      }
    } catch (err) {
      this.error.set(String(err));
    }
    setTimeout(() => void this.refresh(), 1000);
  }

  async toggleAgent(): Promise<void> {
    const b = this.bindings;
    if (!b) return;
    try {
      if (this.snapshot().agentRunning) {
        await b.TrayService.StopAgent();
      } else {
        await b.TrayService.StartAgent();
      }
    } catch (err) {
      this.error.set(String(err));
    }
    setTimeout(() => void this.refresh(), 500);
  }
}
