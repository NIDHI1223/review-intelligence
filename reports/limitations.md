# Limitations & Methodology Notes

This platform never fabricates data. Everything it could not access or verify is recorded here.

## Sources not collected

- **youtube** — YOUTUBE_API_KEY not set in .env
- **reddit** — REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET not set in .env
- **X/Twitter** — excluded by design: API is paid-only and scraping violates ToS.
- **Instagram/Facebook/Quora** — excluded by design: platform ToS prohibit collection.

## Structural caps

- Apple App Store RSS feed caps at ~500 most recent reviews per app per country.
- Google Play reviews fetched newest-first up to the configured per-app/language cap; the corpus skews recent by construction.
- App-store reviews over-represent complaint/praise moments vs. everyday usage.

## Processing drops (full audit in manifests table)

- `cleaning` (cleaning-20260719-172747): {'total': 10807, 'kept': 7477, 'too_short': 3185, 'noise': 145, 'language': 0}
- `dedup` (dedup-20260719-172800): {'unique': 5979, 'exact_dups': 1498}
- `clustering` (clustering-20260719-172843): {'embedded': 5756, 'near_dups_removed': 223, 'clusters': 53, 'unclustered_noise': 1706}
- `enrichment` (enrichment-20260719-174612): {}
- `cleaning` (cleaning-20260719-175259): {'total': 13744, 'kept': 10354, 'too_short': 3186, 'noise': 204, 'language': 0}
- `dedup` (dedup-20260719-175321): {'unique': 8702, 'exact_dups': 1652}
- `enrichment` (enrichment-20260719-180718): {'requests': 436, 'errored_requests': 262, 'enriched_written': 3479}
- `enrichment` (enrichment-20260720-004718): {'requests': 262, 'errored_requests': 0, 'enriched_written': 5214}
- `clustering` (clustering-20260720-011248): {'embedded': 5601, 'near_dups_removed': 59, 'clusters': 46, 'unclustered_noise': 1332}
- `validation` (validation-20260720-203611): {'validated': 62, 'rejected_citation': 0, 'rejected_quote': 5, 'rejected_support': 77, 'corpus_size': 13744}

## Insight rejections

82 candidate insights were rejected by validation (missing citations, unverifiable quotes, or support below threshold) and are retained in the insight store with their rejection reasons for audit.

## Interpretation caveats

- Public reviews are self-selected feedback, not a representative user sample.
- Sentiment and behavioral tags are model-generated (tag audit trail: model + prompt version stamped on every enriched record).
- Per-question categories and their review membership are model-generated single-label classifications over each question's signal pool; reviews the classifier judged to fit no category are counted but not listed.
- Segment hints derive only from what reviewers explicitly stated.