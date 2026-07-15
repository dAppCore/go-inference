// SPDX-Licence-Identifier: EUPL-1.2

/* Register the shared Lit custom elements once, before Angular renders —
   side-effect imports; they call customElements.define. The Angular app then
   uses <lthn-*> freely via CUSTOM_ELEMENTS_SCHEMA (the design pack's model:
   Lit elements, Angular features). */
import '../lit/lthn-core.js';
import '../lit/lthn-charts.js';
import '../lit/lthn-datatable.js';

import { bootstrapApplication } from '@angular/platform-browser';
import { App } from './app/app';

bootstrapApplication(App).catch((err) => console.error(err));
