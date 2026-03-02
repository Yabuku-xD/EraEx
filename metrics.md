# EraEx Metrics Report

## 1) Search Evaluation: Specific Song Query

### Query
- `query`: `coffee by miguel`
- `results`: `10`

### Aggregate Metrics
| Metric | Value |
|---|---|
| `avg_query_fit` | `26.6%` |
| `top1_query_fit` | `81.1%` |
| `avg_semantic_similarity` | `48.5%` |
| `avg_token_overlap` | `4.6%` |

### Main Metrics
- `Top1 Query-Fit`: `81.07%`
- `Macro MRR`: `0.3929`

### Ranked Results
| # | Title | Artist | ID | Year | Query Fit | Semantic | Token Overlap | Prompt Similarity |
|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | coffee | Miguel | `9Z55sZ2oVY4` | 2015 | 81.1% | 50.7% | 8.7% | 15.4% |
| 2 | Pineapple Skies | Miguel | `eKwAib6LosM` | 2017 | 20.6% | 49.1% | 3.8% | 10.6% |
| 3 | DEAL | Miguel | `qmH6zIV6f1g` | 2015 | 20.6% | 49.3% | 4.3% | 9.6% |
| 4 | Now | Miguel | `6eFL1zzGK8o` | 2017 | 20.6% | 48.7% | 3.2% | 0.9% |
| 5 | a beautiful exit | Miguel | `2fngq5osh94` | 2015 | 20.5% | 48.3% | 4.0% | 10.3% |
| 6 | waves | Miguel | `7ShsRESNR34` | 2015 | 20.5% | 48.3% | 4.8% | 11.0% |
| 7 | leaves | Miguel | `sqNDZgDWecs` | 2015 | 20.5% | 47.9% | 5.0% | 11.8% |
| 8 | How Many Drinks? | Miguel | `4rw6fEIgaXo` | 2012 | 20.5% | 47.4% | 4.3% | 11.2% |
| 9 | Arch & Point | Miguel | `rxLv8naj0V0` | 2012 | 20.5% | 47.6% | 4.5% | 11.1% |
| 10 | Anointed | Miguel | `mv0Vt3xMqME` | 2017 | 20.5% | 47.4% | 3.4% | 8.5% |

### Automated Bot Validation
- `Macro MRR`: `0.3929` (`steps=5`, `k=10`)

| Bot | MRR | HitRate |
|---|---:|---:|
| 2015 Fan | 0.0286 | 20.00% |
| Pop Fan | 1.0000 | 100.00% |
| Rock Fan | 0.1500 | 40.00% |

---

## 2) Search Evaluation: Descriptive Query

### Query
- `query`: `midnight drive rnb trap`
- `results`: `10`

### Aggregate Metrics
| Metric | Value |
|---|---|
| `avg_query_fit` | `50.9%` |
| `top1_query_fit` | `55.1%` |
| `avg_semantic_similarity` | `72.0%` |
| `avg_token_overlap` | `3.4%` |
| `genre_hit_rate_at_k` | `100.0%` |
| `avg_facet_token_coverage` | `44.3%` |
| `mixed_intent_hit_rate_at_k` | `40.0%` |

### Main Metrics
- `Top1 Query-Fit`: `55.12%`
- `Macro MRR`: `0.8540`

### Live Bot Router
- `R&B Bot` (`18.0%` confidence)

### Ranked Results
| # | Title | Artist | ID | Year | Query Fit | Semantic | Token Overlap | Prompt Similarity | Bot Profile | Bot Match |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|
| 1 | Late Night Canyon Drive | Big Boned Defect Thugs | `dFRIOMWZl3k` | 2018 | 55.1% | 70.8% | 2.9% | 6.2% | R&B Bot | 7.4% |
| 2 | Night Drive | DeKobe | `UJKVdDS0RsE` | 2018 | 54.1% | 69.6% | 3.4% | 21.8% | R&B Bot | 6.3% |
| 3 | Midnight | Take/Five | `Xc2eocJgOrk` | 2018 | 49.2% | 71.3% | 3.3% | 20.2% | Hip-Hop Bot | 5.5% |
| 4 | Late Night Call | saib. | `Y6826n5qojM` | 2016 | 50.1% | 72.9% | 3.3% | 19.2% | R&B Bot | 5.7% |
| 5 | Late at Night | Paul Carrack | `qeATuvlZBpA` | 2016 | 50.0% | 73.4% | 3.2% | 18.8% | R&B Bot | 4.9% |
| 6 | Night Whispers | Liquid Mind | `vl1Z8VISg9Q` | 2016 | 50.6% | 73.1% | 0.0% | 4.7% | Hip-Hop Bot | 3.7% |
| 7 | Saturday Night | 2 Chainz | `cugjazbGYgQ` | 2017 | 50.1% | 72.7% | 3.4% | 16.6% | Hip-Hop Bot | 3.7% |
| 8 | Workin' Late | SWEAT | `V1JKKXOG24M` | 2016 | 49.9% | 73.0% | 4.0% | 15.1% | Hip-Hop Bot | 3.7% |
| 9 | It Came Upon A Midnight Clear | Jordan Smith | `vbXpPb9KsA8` | 2016 | 50.2% | 72.9% | 3.1% | 17.2% | Hip-Hop Bot | 3.7% |
| 10 | Midnight | Mike WiLL Made-It | `yw6OS8AkH4c` | 2018 | 49.4% | 70.5% | 6.9% | 19.9% | Hip-Hop Bot | 7.4% |

### Automated Bot Validation
- `Macro MRR`: `0.8540` (`steps=5`, `k=10`)

| Bot | MRR | HitRate |
|---|---:|---:|
| 2015 Fan | 0.6952 | 100.00% |
| Pop Fan | 1.0000 | 100.00% |
| Rock Fan | 0.8667 | 100.00% |

---

## 3) For You Evaluation: Specific Profile

### Profile
- `user_id`: `69826fa8-dbaa-4fb7-ba1b-bcd67a6de434`
- `mode`: `adaptive`
- `results`: `20`

### Signals
- `likes=9`
- `plays=25`
- `playlists=9`
- `dislikes=0`

### Aggregate Metrics
| Metric | Value |
|---|---|
| `avg_profile_fit` | `42.6%` |
| `top1_profile_fit` | `63.2%` |
| `avg_profile_similarity` | `68.3%` |
| `avg_bot_match` | `7.1%` |
| `long_tail_share` | `55.0%` |

### Main Metrics
- `Top1 Profile-Fit`: `63.20%`
- `Avg Bot Match`: `7.05%`
- `Long-tail Share`: `55.00%`
- `Macro MRR`: `0.5714`

### Top-20 Constraints
- `PASS`
- `seed 8/8 (pool=18, exact=18)`
- `long_tail 11/5 (thr=0.1%)`
- `max_per_artist 1/1`

### Live Bot Router
- `Rock Bot` (`44.8%` confidence)

### Ranked Results
| # | Title | Artist | ID | Profile Fit | Similarity | Token Overlap | Bot Match | Long-tail |
|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | Melbourne | DMA'S | `b0LuwAMS3Xk` | 63.2% | 99.8% | 15.9% | 13.0% | 4.7% |
| 2 | Reflections | Edison Herbert | `KBmumkfrTYA` | 49.1% | 70.8% | 6.8% | 13.0% | 59.6% |
| 3 | Stacy | Quinn XCII | `TKL8Sq14MiQ` | 55.3% | 88.4% | 12.8% | 10.6% | 0.0% |
| 4 | The Bones | Maren Morris | `gvPMVKUI9go` | 47.4% | 79.0% | 8.8% | 0.0% | 0.0% |
| 5 | Grown Woman | PARTYNEXTDOOR | `M_4GAreca-s` | 59.4% | 100.0% | 7.8% | 0.0% | 0.0% |
| 6 | Hallucinations | dvsn | `gDVufnezUAs` | 45.9% | 72.2% | 8.3% | 15.9% | 0.0% |
| 7 | Sure Thing | Miguel | `esKJ8BbauGk` | 45.7% | 73.6% | 10.9% | 6.5% | 0.0% |
| 8 | Heathens | twenty one pilots | `oLeROuCMwj8` | 44.2% | 71.7% | 9.0% | 6.5% | 0.0% |
| 9 | Eternity Leave | Makari | `Ch6z9BTU0Yw` | 21.8% | 33.8% | 5.4% | 6.5% | 2.4% |
| 10 | Temptress | Stellar | `QFTJ-6AMnUI` | 24.1% | 38.2% | 4.0% | 6.5% | 2.4% |
| 11 | Coffee | Pink Sweat$ | `s0meeKszuf8` | 22.3% | 36.6% | 4.7% | 0.0% | 2.4% |
| 12 | Where The Joy Is | We Are Messengers | `m6XV5JDlvCU` | 22.5% | 36.8% | 5.2% | 0.0% | 2.4% |
| 13 | Praise | AratheJay | `-2FSi-674Fs` | 23.8% | 36.4% | 8.2% | 6.5% | 2.4% |
| 14 | All About That Bass | Meghan Trainor | `Fccz4gz5erM` | 51.3% | 83.9% | 8.7% | 6.5% | 0.0% |
| 15 | Daydreaming | Keke Wyatt | `vj3MDlR6pDw` | 44.8% | 71.8% | 6.1% | 11.2% | 2.4% |
| 16 | Keep Running | Andre Ward | `OTHAWQBvLDY` | 44.1% | 71.6% | 8.5% | 5.6% | 2.4% |
| 17 | The First Time Ever I Saw Your Face | Peter, Paul and Mary | `sx07pH-wSxI` | 44.9% | 71.4% | 8.4% | 11.2% | 2.4% |
| 18 | Name Game | Yo Gabba Gabba | `DYejwyn5YdI` | 43.2% | 70.2% | 7.6% | 5.6% | 2.4% |
| 19 | All of Me | John Legend | `sQtnhwU2R9Y` | 50.0% | 79.4% | 7.5% | 15.9% | 0.0% |
| 20 | I'm Not The Only One | Sam Smith | `O-D0nXOanQY` | 48.3% | 80.2% | 10.1% | 0.0% | 0.0% |

### Validation Metrics
| Metric | Value |
|---|---:|
| `seed_artist_coverage` | 40.00% |
| `unique_artist_ratio` | 100.00% |
| `avg_long_tail_strength` | 4.32% |

### Automated Bot Validation
- `Macro MRR`: `0.5714` (`top_k=15`)

| Bot | MRR | HitRate | AvgMatch |
|---|---:|---:|---:|
| Rock Bot | 0.1429 | 13.33% | 1.87% |
| Electronic Bot | 1.0000 | 13.33% | 3.99% |
