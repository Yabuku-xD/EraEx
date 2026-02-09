# Project Methodology & Data Dictionary 📚

## 1. Dataset Overview: The "EraEx" Sonic Index
**Status**: Completed & Indexed
**Volume**: ~24,365 High-Quality Tracks (2012-2018)

Our dataset is a highly curated "Sonic Index" specifically built for the **2012-2018 era**. Unlike static datasets (e.g., Million Song Dataset), our data is a **Hybrid Dynamic Corpus** — combining a persisted vector index for core retrieval with real-time API enrichment.

### Source & Scope
-   **Primary Source**: [Deezer API](https://developers.deezer.com/api) (Metadata & Audio Previews)
-   **Time Period**: Strictly **2012-01-01** to **2018-12-31**.
-   **Scale**: ~24,000 unique tracks indexed with vector embeddings.
-   **Diversity**: 36+ Genres including Pop, Hip Hop, R&B, Alternative, Electro, Jazz, and Niche subgenres (e.g., "French Touch", "Lo-Fi").

---

## 2. Data Collection & Preprocessing (The Filter Pipeline) 🛠️

We employ a multi-stage filtering pipeline to ensure data quality and user trust.

### Stage 1: The "Wide Net" (Ingestion)
-   **Process**: We crawled Deezer's "Top Playlists" and "Genre Charts" for each year (2012-2018).
-   **Raw Volume**: Initially ~45,000 candidate tracks were inspected.

### Stage 2: The Nostalgia Filter (Strict Temporal Validation)
*Addressing Feedback: Empirical Justification for Filtering*
-   **Logic**: Many playlists labeled "2012 Hits" contain re-releases or incorrect tagging. We enforce a strict `release_date` check.
-   **Empirical Impact**:
    -   **Accepted**: ~24,365 tracks (54% of candidates).
    -   **Rejected**: ~20,000+ tracks (46% of candidates).
    -   *Reasons for Rejection*:
        -   **Out of Era**: 35% (Tracks from 2011, 2019+, or "Remasters").
        -   **No Preview**: 5% (Tracks missing 30s audio preview for analysis).
        -   **Duplicates**: 6% (Same ID or Exact Title+Artist match).

### Stage 3: Metadata Clean-up
*Addressing Feature Leakage & Quality*
-   **Feat. Stripping**: We observed that "Featured" artists clutter the display and search results.
    -   *Action*: Regex removal of `(feat. ...)`, `[ft. ...]`, `(featuring ...)` from titles and artist names.
-   **Deduplication**: We deduplicate not just by ID, but by `Normalized(Title) + Normalized(Artist)` to handle "Deluxe Version" vs "Radio Edit" redundancy.

---

## 3. Data Dictionary (Schema) 📝

### A. The Sonic Index (`sonic_*.pkl`)
Our core retrieval structure. Optimized for memory and speed (~150MB total).

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `id` | `int` | Unique Deezer Track ID |
| `title` | `str` | Song Title |
| `artist` | `str` | Artist Name |
| `release_date` | `str` | Strict "YYYY-MM-DD" |
| **`semantic_vector`** | `float32[384]` | **SBERT (all-MiniLM-L6-v2)** embedding of Title + Artist + Genre tags. Captures *meaning* ("sad love song"). |
| **`audio_vector`** | `float32[128]` | **MFCC + Spectral Contrast** embedding of 30s preview. Captures *timbre/mood* ("upbeat", "acoustic"). |
| `preview` | `url` | URL to 30s MP3 snippet (Verified valid at indexing time). |

### B. Real-Time Enrichment (GLM-4.7)
We do not store every possible metadata field. Instead, we use **LLM-Based Enrichment** at query time.
-   **Model**: Z.AI GLM-4.7 (Reasoning Model)
-   **Input**: User Query (e.g., "sad midnight drive")
-   **Output**: JSON `{ "mood": "Melancholic", "genre": "Synthwave", "keywords": ["night", "drive"] }`
-   **Integration**: These enriched terms are then re-vectorized to search the Sonic Index.

---

## 4. Exploratory Data Analysis (EDA) Highlights 📊

*Addressing Feedback: Full Dataset Analysis*

### A. Genre Distribution
-   **Dominant Genres**: Pop (25%), Hip Hop (20%), Alternative (15%).
-   **Long-Tail**: Significant presence of "Indie Pop", "Electro", "R&B".
-   **Observation**: The mid-2010s saw a massive rise in "Electronic-Pop" crossover, reflected in our vector clusters.

### B. Word-Level Analysis
-   **Top Keywords**: "Love", "Night", "Time", "Life", "Girl".
-   **Nostalgia Markers**: High frequency of words like "Yeah", "Baby", "Tonight" in 2012-2014 tracks vs more introspective lyrics in 2017-2018.

### C. Cache Implementation Details
*Addressing System Architecture Feedback*
-   **Goal**: Minimize API latency and rate limits.
-   **Mechanism**:
    -   **L1 Cache (Memory)**: Recently accessed track metadata/vectors.
    -   **L2 Cache (Disk - Pickle)**: The `sonic_*.pkl` files serve as a persistent cold storage for the 24k track universe.
    -   **L3 Cache (Browser)**: Album art and Previews are cached by the browser via standard HTTP headers.

---

## 5. Methodology: Advanced Hybrid Retrieval 🧠

*Addressing Feedback: Beyond Simple Cosine Similarity*

We moved beyond basic vector matching to a **Multi-Stage Retrieval System**:

1.  **Query Understanding (GLM-4.7)**:
    -   User query is expanded using a large language model to extract *intent* (Mood vs Content).
    -   *Example*: "late night vibes" -> "Melancholy, R&B, Slow Tempo, Nocturnal".

2.  **Dual-Encoder Retrieval**:
    -   **Semantic Search**: SBERT vector similarity finds tracks matching the *expanded* concepts.
    -   **Audio Search (Serendipity)**: (Optional) Vector similarity on audio features finds tracks that *sound* similar.

3.  **Deduplication & Reranking**:
    -   Results are deduplicated by `(Title, Artist)` key to remove remix clutter.
    -   Ranked by a weighted score of `VectorSimilarity + PopularityBias`.

4.  **Verification**:
    -   Final results are sanity-checked against the `NostalgiaFilter` one last time (though index is already clean).
