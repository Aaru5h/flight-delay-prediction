# Phase 2 — Feature quality

- Eligible labelled flights: **327,346**
- Approved estimator features: **37**
- Fresh origin weather rows: **326,570**
- Stale or missing weather rows: **776**
- No carrier history: **928**
- No route history: **1,280**

Historical outcomes are restricted to training dates and a conservative completion bound (NYC scheduled departure + 24 hours + positive arrival delay) before T−60. This is an availability proxy, since the source has no outcome release timestamp. Weather observations are retrospective, not forecast vintages.

See [the feature manifest](phase2_feature_manifest.md) for every approved input and its rule.
