// SPDX-Licence-Identifier: EUPL-1.2

import { ApplicationConfig, provideBrowserGlobalErrorListeners } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';

import { BASE_PATH } from '@lethean/lem-sdk-angular';

export const appConfig: ApplicationConfig = {
  providers: [
    provideBrowserGlobalErrorListeners(),
    provideHttpClient(),
    // Empty base path = same-origin requests; the dev server's proxy
    // (proxy.conf.json) forwards /v1 to the lem serve. lem itself sends no
    // CORS headers, so a browser app on another origin cannot call it
    // directly — the proxy is the standard Angular dev answer.
    { provide: BASE_PATH, useValue: '' },
  ],
};
