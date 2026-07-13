# MoE block device-path coverage notes

`encMoEBlockQuantDevice` has one device-allocation fallback that cannot be
honestly forced from an in-package test without adding a production test seam:

- Lines 1873-1887: `ensureAllRoutesScratch` or either
  `gatherQMVAllRoutesMetadata` call can fail, setting `allRoutes = false` and
  selecting the per-route serial encoder at lines 2100-2117. The first failure
  requires Metal shared-buffer allocation failure; the metadata failures require
  an internal cache/allocation failure. Neither dependency is injectable, and
  corrupting an allocated scratch would test a state that no public allocation
  path produces. Current behavior is therefore pinned only by the successful
  all-routes cases and the normal serial schedule test; no fake failure test was
  added.
