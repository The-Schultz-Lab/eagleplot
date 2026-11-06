// EaglePlot web app: service worker (roadmap D4.2).
//
// Makes the app installable ("Install EaglePlot" in Chrome and Edge) and lets it open
// offline after one online visit. index.html registers it and, on a first visit, waits
// until it controls the page before starting Python, so even the first visit's downloads
// (including the Python worker's) pass through here and are cached.
//
// - Files of this site: network first, so a new deploy shows up at once; the cached copy
//   when offline. Cached per build (BUILD changes with every deploy); older builds' caches
//   are removed when this version activates.
// - Pyodide from the CDN: cache first. Its URLs contain the version and never change.
//
// packaging/web/build_site.py replaces __EAGLEPLOT_BUILD__ with a hash of the build.
'use strict';

const BUILD = '__EAGLEPLOT_BUILD__';
const APP_CACHE = `eagleplot-app-${BUILD}`;
const CDN_CACHE = 'eagleplot-cdn';
const CDN_ORIGINS = ['https://cdn.jsdelivr.net'];

self.addEventListener('install', () => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const names = await caches.keys();
    await Promise.all(
      names.filter((n) => n.startsWith('eagleplot-app-') && n !== APP_CACHE).map((n) => caches.delete(n)),
    );
    await self.clients.claim();
  })());
});

async function networkFirst(request) {
  const cache = await caches.open(APP_CACHE);
  try {
    const response = await fetch(request);
    if (response.ok) await cache.put(request, response.clone());
    return response;
  } catch (err) {
    const cached = await cache.match(request, { ignoreSearch: true });
    if (cached) return cached;
    // Offline and never seen: a page of the app (…/app/settings) gets the app itself.
    if (request.mode === 'navigate') {
      const shell = await cache.match(new URL('./', self.registration.scope).href);
      if (shell) return shell;
    }
    throw err;
  }
}

async function cacheFirst(request) {
  const cache = await caches.open(CDN_CACHE);
  const cached = await cache.match(request);
  if (cached) return cached;
  const response = await fetch(request);
  if (response.ok) await cache.put(request, response.clone());
  return response;
}

// On a first visit the page itself and the stlite runtime load before this worker controls
// the page, so they never pass through the fetch handler. index.html then sends their
// URLs; cache them (mostly from the HTTP cache) so the app can open offline next time.
self.addEventListener('message', (event) => {
  if (event.data?.type !== 'cache-urls') return;
  event.waitUntil((async () => {
    const cache = await caches.open(APP_CACHE);
    const scope = self.registration.scope;
    await Promise.all(
      (event.data.urls || [])
        .filter((u) => typeof u === 'string' && u.startsWith(scope))
        .map(async (u) => {
          try {
            if (await cache.match(u)) return;
            const response = await fetch(u);
            if (response.ok) await cache.put(u, response);
          } catch (err) {
            /* offline or gone: skip it */
          }
        }),
    );
  })());
});

self.addEventListener('fetch', (event) => {
  const request = event.request;
  if (request.method !== 'GET') return;
  const url = new URL(request.url);
  if (url.origin === self.location.origin) {
    if (url.href.startsWith(self.registration.scope)) event.respondWith(networkFirst(request));
  } else if (CDN_ORIGINS.includes(url.origin)) {
    event.respondWith(cacheFirst(request));
  }
});
