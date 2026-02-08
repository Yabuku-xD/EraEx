# Project Methodology & Data Dictionary 📚

## 1. Dataset Overview: The "EraEx" Collection

Our dataset is a curated collection of music tracks specifically from the **2012-2018 era**, designed to power a nostalgia-based recommendation engine. Unlike standard datasets (e.g., Million Song Dataset), our data is dynamically crawled to ensure high relevance to our target time period.

### Source & Scope
-   **Primary Source**: [Deezer API](https://developers.deezer.com/api)
-   **Target Years**: 2012, 2013, 2014, 2015, 2016, 2017, 2018.
-   **Volume**: Targeting ~140,000 unique tracks (20,000 per year).
-   **Genres**: **Dynamic Discovery** (30+ genres). We fetch the full list of available genres from the Deezer API, including niche categories (e.g., "Afrobeat", "K-Pop", "Techno", "Classical", "Films/Games") alongside major genres like Pop, Rock, and Hip Hop.

---

## 2. Data Collection Process (The "Wide Net" Strategy) 🕸️

We employ a custom crawler (`notebooks/build_sonic_index.ipynb`) that operates in three stages:

### Stage 1: Dynamic Genre Discovery
-   The crawler queries the Deezer API for a list of all available genre IDs.
-   It retrieves ~30 primary genres (e.g., "Alternative", "Dance") to ensure diversity beyond top 40 hits.

### Stage 2: Playlist Search & Aggregation
-   For each `Year` (2012-2018) and `Genre`, we generate search queries:
    -   `"Top 2012"` (General)
    -   `"2012 Jazz"` (Specific)
    -   `"Best of 2015 Metal"` (Niche)
-   We fetch the **Top 30 Playlists** for each query to gather a massive pool of "Candidate Tracks."

### Stage 3: The Nostalgia Filter (Strict Enforcement) 🛡️
-   Many playlists named "2012 Hits" contain newer songs.
-   **Filtering Logic**:
    -   We fetch the precise `release_date` for every candidate track (often requiring a secondary API call to the `album` endpoint).
    -   If `release_date < 2012-01-01` or `release_date > 2018-12-31`, the track is **discarded**.
    -   This rigorous filtering ensures 100% temporal accuracy, crucial for the user value proposition.

---


---

## 3. Data Dictionary (Schema) 📝

### A. Raw Data (Source Material)
Before processing, we ingest data from two primary APIs.

#### 1. Deezer API (Track Object)
For every candidate track, we receive a JSON object. Key fields we utilize (and some we filter out):
```json
{
  "id": 3135556,
  "title": "Harder, Better, Faster, Stronger",
  "link": "https://www.deezer.com/track/3135556",
  "duration": 224,
  "rank": 958932,  // Popularity Score (0-1000000)
  "preview": "https://cdns-preview-d.dzcdn.net/stream/c-ded...", // 30s MP3
  "artist": {
    "id": 27,
    "name": "Daft Punk",
    "picture_medium": "https://e-cdns-images.dzcdn.net/images/artist/..."
  },
  "album": {
    "id": 302127,
    "title": "Discovery",
    "cover_medium": "https://e-cdns-images.dzcdn.net/images/cover/..."
  },
  "release_date": "2001-03-13" // CRITICAL: This is what we filter on!
}
```

#### 2. Last.fm API (Track Tags) - *For Enrichment*
For a subset of tracks, we query Last.fm to get "crowdsourced tags" to improve semantic matching.
```json
{
  "toptags": {
    "tag": [
      {"name": "electronic", "count": 100},
      {"name": "house", "count": 85},
      {"name": "dance", "count": 70},
      {"name": "french touch", "count": 55}, // Niche descriptive tag
      {"name": "workout", "count": 30}      // Contextual tag
    ]
  }
}
```

### B. Final Processed Dataset (`sonic_index.pkl`)
The final index is a highly optimized subset of the above, discarding unused metadata (like `link`, `duration`, `rank`) to save memory, while adding our computed vectors.

| Field Name | Data Type | Source | Description |
| :--- | :--- | :--- | :--- |
| `id` | `int` | Deezer | Unique Track ID |
| `title` | `str` | Deezer | Song Title |
| `artist` | `str` | Deezer | Artist Name |
| `release_date` | `str` | Deezer | "YYYY-MM-DD" (Filtered 2012-2018) |
| `preview` | `url` | Deezer | URL to 30s MP3 snippet |
| `audio_vector` | `float32[128]` | **Computed** | Derived from `preview` MP3 (MFCCs) |
| `semantic_vector` | `float32[384]` | **Computed** | Derived from `title` + `artist` + `tags` |

---

## 4. Feature Engineering & Preprocessing 🛠️

We transform raw audio and text into mathematical vectors to enable "Sonic Search."

### A. Audio Vectorization (The "Vibe")
-   **Input**: 30-second MP3 preview.
-   **Library**: `librosa` (Python Audio Analysis).
-   **Technique**:
    1.  **MFCCs (Mel-Frequency Cepstral Coefficients)**: Extracts 13 coefficients representing the short-term power spectrum of sound (timbre).
    2.  **Spectral Contrast**: Measures the difference between peaks and valleys in the sound spectrum (texture).
    3.  **Aggregation**: We compute the **Mean** and **Variance** of these features over the 30s clip to create a fixed-size representation.
-   **Result**: A 128-dimensional vector representing the song's "acoustical fingerprint."

### B. Semantic Vectorization (The "Meaning")
-   **Input**: Metadata String (`"{Title} by {Artist} album {Album}"`).
-   **Model**: `all-MiniLM-L6-v2` (Sentence-BERT).
-   **Technique**:
    -   The text is passed through a pre-trained Transformer model.
    -   The model outputs a dense vector that captures semantic concepts (e.g., "love", "party", "heartbreak") without relying on keyword matching.
-   **Result**: A 384-dimensional vector representing the song's "narrative context."

---

## 5. Candidate Generation & Ranking Pipeline 🚀

When a user searches or uses "I'm Feeling Lucky," the system performs the following steps:

1.  **Vector Search**:
    -   The User Query is converted to a vector (Audio or Text).
    -   We calculate **Cosine Similarity** between the Query Vector and all 140k Track Vectors.
    -   Top N most similar tracks are retrieved.

2.  **Hybrid Reranking (Future Work)**:
    -   We plan to combine this Content-Based Score with Collaborative Filtering (ALS) scores to boost popular/relevant items.

3.  **Result**: A list of tracks that match the User's Era (2012-2018) AND Vibe.
