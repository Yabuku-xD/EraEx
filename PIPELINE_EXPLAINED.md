# EraEx Pipeline Explained

## 1. Purpose
EraEx solves two related but different ranking problems.

1. Search ranking: given a text query, return the best matching tracks.
2. Personalized recommendation ranking: given user behavior, return the most relevant next tracks.

Both pipelines are multi-stage. They do not rely on a single score or a single model.

## 2. Design Philosophy
The system is built around signal fusion, not single-signal dominance.

- Semantic meaning matters.
- Genre, vibe, and mood overlap matter.
- Audio-shape compatibility matters.
- Artist/title exactness matters when explicitly requested.
- Year and instrumental intent matter when present.
- Popularity is used as a weak stabilizer, not as the primary objective.

This is why broad vibe queries and precise artist/title queries can both be handled by the same system.

## 3. Shared Data Representation
Each track is represented in two ways.

1. Dense semantic representation.
2. Structured feature representation.

The dense representation supports fuzzy natural-language retrieval. The structured representation supports precise reranking and intent constraints.

A track profile can include these fields.

- Title, artist, album.
- Tags, genres, mood tokens.
- Description text.
- Numeric audio attributes such as tempo, energy, brightness, mood, valence.
- Year.
- Instrumental flag and confidence.
- Popularity proxies such as views, playcount, or rank.

## 4. Search Pipeline

### 4.1 Query Understanding
Incoming query text is normalized and tokenized. Then intent signals are extracted.

- Like-mode intent: phrases like “songs like X”.
- Strict artist intent: short artist-only lookup behavior.
- Title-by-artist intent: patterns like “title by artist”.
- Facet-heavy intent: strong genre/vibe/mood composition.
- Audio intent: inferred from phrases and tokens mapped to target audio profiles.
- Year intent: explicit 4-digit year.
- Instrumental intent: explicit vocal/instrumental preference.

### 4.2 Query Bot Routing
A lightweight router maps the query into high-level style profiles (for example R&B, Hip-Hop, Electronic, Chill) and computes confidence per profile.

- Token and phrase matches are combined.
- Only confident profiles are kept.
- The strongest profiles are attached to query intent metadata.

These profiles are not final ranking decisions. They provide an additional compatibility signal.

### 4.3 Candidate Generation
Candidate generation is multi-channel and merged.

- Primary semantic ANN retrieval.
- Alternate semantic retrieval from raw query text.
- Facet-focused semantic retrieval when the query is compositional.
- Artist-target retrieval for strict artist intent.
- Anchor-profile retrieval for like-mode intent.
- Rare title/artist token channel for hard+soft mixed facet queries.

The merged candidate set is deduplicated before scoring.

### 4.4 Track-Level Feature Computation
For each candidate, the pipeline computes a feature vector that includes.

- `semantic_score`.
- `overlap_score` from tag/query overlap.
- `genre_score`, `vibe_score`, `mood_score`.
- `facet_score` as a weighted aggregate of genre/vibe/mood.
- `description_score` with description trust scaling.
- `title_score`.
- `audio_query_score` from query audio target similarity.
- `audio_anchor_score` for like-mode anchor similarity.
- `artist_query_score`, `title_hint_score` and exact-match flags.
- `year_match_score`.
- `instrumental_match_score`.
- `popularity_score` normalized within the candidate set.
- Mixed-intent coverage features for compositional queries.

### 4.5 Core Similarity and Coverage Methods
Key methods used by search scoring.

- Balanced token overlap uses an F1-style overlap to avoid one-token saturation.
- Audio similarity uses mean per-dimension closeness: `1 - |candidate - target|`, then averaged.
- Facet token expansion adds lightweight variants (for example token families around related genre terms).
- Rare focus token selection uses document-frequency-aware token picking.

### 4.6 Facet Alignment Score
Facet alignment blends core facet fit and contextual support.

\[
\text{facet\_alignment} = 0.56\cdot\text{core\_facet}
+ 0.20\cdot\text{lexical\_context}
+ 0.16\cdot\text{audio\_query\_score}
+ 0.08\cdot\text{bot\_profile\_score}
\]

Where `core_facet` is the strongest facet signal among facet-related fields.

### 4.7 Anti-Generic Title Bias Control
Title-only matches are penalized when context is weak.

\[
\text{title\_bias\_penalty} = \max(0, \text{title\_score} - \text{context\_support})
\]

This prevents irrelevant “keyword-in-title” results from outranking true style matches.

### 4.8 Final Ranking Score
There are different score equations by intent mode.

Like-mode emphasizes semantic + facet + anchor compatibility.

\[
\text{score} =
w_s\cdot\text{semantic}
+ w_o\cdot\text{overlap}
+ w_p\cdot\text{popularity}
+ w_f\cdot\text{facet}
+ w_a\cdot\text{artist\_style}
+ w_{au}\cdot\text{audio\_anchor}
+ w_b\cdot\text{bot\_match}
+ \Delta_{instr}
\]

Normal search emphasizes semantic + lexical + facet + audio query + year, with penalties.

\[
\text{score} =
w_s\cdot\text{semantic}
+ w_o\cdot\text{overlap}
+ w_p\cdot\text{popularity}
+ w_f\cdot\text{facet}
+ w_d\cdot\text{description}
+ w_{aq}\cdot\text{audio\_query}
+ w_y\cdot\text{year\_match}
+ w_{aqr}\cdot\text{artist\_query}
+ w_{tq}\cdot\text{title\_query}
+ w_b\cdot\text{bot\_match}
- w_t\cdot\text{title\_bias\_penalty}
- \text{focus\_gate\_penalty}
+ \Delta_{instr}
\]

### 4.9 Dynamic Weighting
Weights are query-adaptive and may be externally tuned.

- Like-mode queries raise facet/artist-style influence.
- Strict artist queries reduce broad semantic drift.
- Facet-heavy queries increase facet/context emphasis.
- Audio/year/instrumental intent increases matching dimensions.
- Very short and very long queries are rebalanced differently.

Weight groups are normalized to keep scoring numerically stable.

### 4.10 Mixed-Intent Handling
For hard+soft compositional queries, additional coverage controls are applied.

- Hard coverage and soft coverage are measured separately.
- Mixed intent coverage combines hard and soft signals.
- Missing hard/soft focus may add penalties.
- Top-k quota repair can enforce a minimum number of mixed-intent-covered results.
- After quota repair, top block can be resorted by query-fit quality.

### 4.11 Diversification
When enabled and appropriate, top candidates are diversified with MMR.

\[
\text{MMR}(i)=\lambda\cdot\text{relevance}(i) - (1-\lambda)\cdot\max_{j\in S}\text{similarity}(i,j)
\]

Similarity between two search rows includes artist match, tag overlap, genre overlap, mood overlap, and audio closeness.

### 4.12 Fallback Path
If semantic ranking yields no rows, lexical fallback ranking is used.

- Token/title/artist/description lexical signals are scored.
- Bot compatibility can still contribute.
- Strict title-by-artist reordering is applied when relevant.

### 4.13 Search Output Shape
Search returns.

- Ranked results.
- Query-level quality metrics.
- Query-intent diagnostics.
- Pagination-ready behavior using `limit` and `offset`.

## 5. Recommendation Pipeline

### 5.1 Mode Selection
Recommendation mode depends on available profile history.

- Cold start: no meaningful behavior history.
- Adaptive personalization: likes/playlist tracks/plays available.

### 5.2 Similarity Components
Candidate-to-seed similarity combines three components.

- Mood similarity: semantic + audio + tag overlap blend.
- Era similarity: Gaussian year distance.
- Style similarity: same-artist boost or genre overlap.

Era similarity uses.

\[
\exp\left(-\frac{(y_1-y_2)^2}{2\sigma^2}\right)
\]

### 5.3 Dynamic Feature Weights
Weights over mood/era/style are personalized from recent behavior coherence.

- Tight year preference increases era weight.
- High mood coherence increases mood weight.
- Repeated artist patterns increase style weight.

### 5.4 Dynamic Source Weights
User behavior channels are weighted dynamically.

- Likes channel.
- Playlist channel.
- Plays channel.
- Dislike ratio can down-weight noisy implicit behavior.

Channel coherence and amount both contribute.

### 5.5 Positive/Negative Similarity Coefficients
Two adaptive coefficients are computed.

- Positive similarity gain.
- Negative similarity penalty.

These respond to explicit-vs-implicit strength and dislike ratio.

### 5.6 Candidate Pool Construction
Adaptive candidate pool combines multiple buckets.

- Personalized neighbors from seed tracks via artist/tag/year indexes.
- Seed-affinity artist pool.
- Long-tail pool.
- Popularity baseline pool.

Mix proportions are dynamic.

- Strong explicit profile increases personalized and long-tail share.
- Weak profile increases popularity floor.

### 5.7 Long-Tail and Mainstream Signals
Popularity percentile is converted into two complementary strengths.

- Long-tail strength rises as popularity percentile increases.
- Mainstream strength rises near chart-head percentile.

These are used as controlled ranking priors, not absolute gates.

### 5.8 Live Profile Bot Matching
User behavior is routed to high-level taste profiles, and candidates receive bot-match compatibility scores.

- Seed token distributions infer active profile bots.
- Candidate tokens are compared to active bot token sets.
- Bot confidence and overlap combine into a bounded contribution.

### 5.9 Final Recommendation Candidate Score
Each candidate receives an adjusted score that blends.

- Profile quality normalization.
- Best positive similarity to recent positive seeds.
- Best negative similarity to recent disliked seeds.
- Long-tail bonus.
- Replay/previous-track boosts.
- Seed-affinity artist boost.
- Bot boost.
- Track and artist skip penalties.
- Mainstream penalty.

### 5.10 Diversity Selection
Two selection paths exist.

- DPP path using kernel decomposition and greedy determinant-based selection.
- Fast path using top-k with lightweight artist-cap diversity.

DPP kernel is formed as quality-scaled similarity.

\[
L = S \odot (q q^T)
\]

Then greedy selection is used to approximate high-diversity high-quality subsets.

### 5.11 Output Quotas and Repair
Final ordering is quota-aware.

- Seed-affinity quota.
- Long-tail quota.
- Exploration quota.
- Per-artist maximum cap.

For strong top-20 personalized scenarios, hard targets are enforced with swap repair if needed.

- Minimum seed-affinity coverage target.
- Minimum long-tail coverage target.
- Strict artist concentration cap.

If strict targets are infeasible under constraints, effective targets are adjusted and reported.

### 5.12 Final Recommendation Output
Recommendation output includes.

- Ranked tracks.
- Bot profile signals.
- Long-tail/mainstream strengths.
- Quality and constraint diagnostics.
- Deduplication by title+artist.

## 6. Search Metrics Glossary
All search metrics are computed as bounded values in `[0,1]` unless otherwise noted.

- `avg_query_fit`: average final query-fit quality over returned rows.
- `top1_query_fit`: query-fit quality of the first ranked row.
- `avg_semantic_similarity`: mean semantic match strength.
- `avg_token_overlap`: average token-level overlap between query and candidate text.
- `avg_prompt_similarity`: average string-level prompt similarity.
- `genre_hit_rate_at_k`: share of results passing genre/facet coverage threshold.
- `facet_hit_rate_at_k`: share of results with sufficient facet alignment.
- `avg_facet_alignment`: mean facet alignment score.
- `avg_facet_token_coverage`: mean coverage of requested facet tokens.
- `avg_mixed_intent_coverage`: mean mixed hard+soft intent coverage.
- `mixed_intent_hit_rate_at_k`: share of results above mixed-intent threshold.
- `avg_bot_match`: mean bot-profile compatibility over results.
- `fallback_lexical_used`: indicates semantic path failed and lexical fallback was used.

### 6.1 Query Intent Diagnostic Fields
- `like_mode`: style-anchor retrieval mode flag.
- `strict_artist_mode`: strict artist lookup mode flag.
- `facet_heavy_query`: compositional facet query flag.
- `title_artist_query`: explicit title-by-artist query flag.
- `active_bots`: top active query bot profiles with confidence.
- `bot_intent_confidence`: confidence of strongest active query bot.

## 7. Recommendation Metrics Glossary
Recommendation quality metrics are computed on ranked output.

- `avg_profile_fit`: average profile relevance fit.
- `top1_profile_fit`: profile fit of rank 1 item.
- `avg_profile_similarity`: average behavioral similarity to seeds.
- `avg_token_overlap`: average token overlap with seed token bag.
- `avg_bot_match`: average live bot compatibility.
- `avg_long_tail_strength`: average long-tail intensity.
- `long_tail_share`: fraction of rows qualifying as long-tail.
- `seed_artist_coverage`: fraction of rows aligned with seed-affinity artists.
- `unique_artist_ratio`: artist diversity ratio.

### 7.1 Constraint Metrics
- `seed_target_desired` and `seed_target_effective`.
- `long_tail_target_desired` and `long_tail_target_effective`.
- `max_artist_target` and `max_artist_actual`.
- `met`: whether all active hard constraints are satisfied.

## 8. Validation Metrics
Two evaluation metrics are used across automated bot validation.

- `MRR` (Mean Reciprocal Rank): rewards placing relevant tracks earlier.
- `Hit Rate`: fraction of ranked lists containing at least one relevant result above threshold.

Macro MRR is the mean of bot-level MRR values.

## 9. Frontend and CLI Consistency
Search and recommendation logic are shared through the same backend ranking engines.

- CLI and frontend are expected to match when query, user signals, `limit`, `offset`, metadata snapshot, and runtime settings are the same.
- Any mismatch is usually due to differences in paging window, caching state, or data freshness, not separate ranking logic.

## 10. Implemented Product-Layer Changes
Recent behavior-level changes now active in the product experience.

1. Home query pagination shows 15 items per page.
2. Home queue can keep fetching additional batches while only rendering one page at a time.
3. Paging uses `limit/offset` style retrieval and can continue forward without replacing the queue.
4. Internal ranking metrics are computed for diagnostics/evaluation even when not displayed in UI.
5. Global settings interaction includes support contact entry and version label.

## 11. Practical Interpretation Guide
Use these quick checks when quality looks off.

1. High semantic similarity but low query-fit usually means lexical/facet mismatch or focus penalties.
2. High token overlap but low facet alignment usually means title or text coincidence without style agreement.
3. Strong recommendation similarity but weak diversity usually indicates quotas or diversity settings need rebalance.
4. Low long-tail share with a strong explicit profile indicates exploration floor is too conservative.
5. Good average metrics with weak top-1 usually indicates ordering weights need top-rank emphasis.

## 12. Summary
EraEx is a hybrid retrieval-and-rerank system with adaptive weighting, compositional intent handling, explicit anti-noise controls, optional diversification, and quota-aware personalized recommendation assembly. The pipeline is designed to remain stable for precise queries while still handling vague human language and mixed emotional/genre intent.
