# Modeling Phase: EraEx Recommendation System (2012-2018)

## 1. Problem Framing and Modeling Goal 🎯

### Problem Statement
In the crowded music streaming landscape, users struggle to discover music that fits a specific "nostalgic vibe" (e.g., "sad piano from 2012-2018"). Standard recommendation engines prioritize mass popularity and recent hits, often burying niche tracks from specific eras. Our goal is to build a **Hybrid Recommendation System** that surfaces high-quality tracks strictly from the **2012-2018 era**, using both audio features (how it sounds) and semantic meaning (lyrics/mood).

### Modeling Goal
To predict the relevance of a candidate track $t$ to a user's query $q$ or listening history $h$, such that the top-K recommendations maximize both similarity and "nostalgia fit."

- **Target Variable**: A relevance score $S_{total} \in [0, 1]$ for each track.
-   **Task Type**: Information Retrieval & Recommendation (Hybrid: Content-Based + Collaborative Filtering).
-   **Key Challenge: The "Metadata Gap"**: 
    -   *Constraint*: API metadata is often sparse or "noisy" (e.g., a track tagged "Pop" might actually be "Sad Acoustic").
    -   *Solution*: We cannot rely on text tags alone. We must process raw audio to extracting the "true vibe."
    -   *Constraint*: **Strict Temporal Filter**: All recommendations must utilize metadata to verify release date is between Jan 1, 2012, and Dec 31, 2018.

---

## 2. Data Overview and Feature Considerations 📊

Based on our completed **EDA and Preprocessing (Phase 1)**, we are utilizing the **Deezer API** dataset.

### Dataset Profile
-   **Source**: Deezer API (Playlist Crawling: "Top 2012" through "Top 2018").
-   **Size**: ~140,000 Tracks (20,000 per year for 7 years).
-   **Key Characteristics**:
    -   **Noisy Tags**: Genre tags are broad ("Alternative" covers both Grunge and Folk). Use Audio Vectors to differentiate.
    -   **Temporal Noise**: "Remastered 2023" versions of 2012 songs.
    -   **Cold Start**: 140k tracks have no user interaction history initially.

### Features Used
1.  **Audio Features (The "Truth")**:
    -   *Source*: 30s MP3 Previews.
    -   *Engineering*: MFCCs & Spectral Contrast -> 128d Vector.
    -   *Purpose*: Overcome missing mood tags by analyzing actual timbre.
2.  **Semantic Features (The "Context")**:
    -   *Source*: Title/Artist/Album strings.
    -   *Engineering*: SBERT embedding.
3.  **Temporal Features (The "Filter")**:
    -   *Source*: `release_date` (Strict Range Validation).

---

## 3. Overall Modeling Strategy and Algorithms 🧠

We propose a **"Two-Tower" Hybrid Strategy** specifically designed to solve the **Metadata Sparsity** problem.

### Algorithm Selection Rationale
1.  **Why not just Metadata?**
    -   Deezer tags are too generic. A user wanting "Dark 2014 Pop" would get "Happy 2014 Pop" if we only filtered by "Pop."
    -   **Solution**: **Audio Content-Based Filtering (Sonic Engine)**. We analyze the raw waveform to find "darkness" (Spectral Contrast) that metadata misses.

2.  **Why not just Audio?**
    -   Audio similarity is subjective. A "Sad Piano" song might sound like a "Calm Piano" song but have very different lyrical meanings.
    -   **Solution**: **Semantic Vector Search (SBERT)**. We embed the title/artist to capture semantic intent.

3.  **Why Collaborative Filtering (Als)?**
    -   Content-based search is "precise" but lacks "surprise." ALS introduces community popularity to surface hits.

### The Hybrid Ranker
-   **Goal**: robustly score $t$ even if metadata is poor.
-   **Formula**: $Score = \alpha \cdot Sim_{Audio} (Truth) + \beta \cdot Sim_{Semantic} (Meaning) + \gamma \cdot Score_{CF} (Popularity)$

---

## 4. Modeling Execution Timeline (Feb 9 – Feb 23) 📅

### Week 1: Data Pipeline & Indexing (Feb 9 – Feb 15)
| Task | Owner | Expected Outcome |
| :--- | :--- | :--- |
| **1. Data Collection (Crawler)** | **Shyamalan** | `sonic_index.pkl`: Crawler logic, API Integration, Parallel Processing. |
| **2. Cleaning & EDA** | **Ann** | `eda_report.ipynb`: Genre distribution analysis, Nostalgia Filter validation. |
| **3. Vector Pipeline (SBERT)** | **Atul** | `semantic.py`: Text embedding pipeline for titles/artists. |
| **4. Vector Retrieval (FAISS)** | **Atul** | `search_engine.py`: Functional FAISS index for fast similarity search. |

### Week 2: Evaluation & Presentation (Feb 16 – Feb 23)
| Task | Owner | Expected Outcome |
| :--- | :--- | :--- |
| **5. Hybrid Search Prototype** | **[Team]** | `hybrid.py`: Merging Audio + Semantic scores (Weighted Sum). |
| **6. Evaluation & Metrics** | **Atul** | `metrics_report.md`: Precision@10, Diversity analysis. |
| **7. Slide Deck Prep** | **[Team]** | Final Presentation Draft (Focus on "Metadata Gap" Solution). |
| **8. Final Rehearsal** | **[Team]** | 10-minute presentation walkthrough. |

---

## 5. Peer Evaluation (Team Contribution Breakdown) 🤝

| Team Member | Contribution (%) | Responsibilities & Achievements |
| :--- | :--- | :--- |
| **Ann** | **33.3%** | **Data Cleaning & Preprocessing:**<br>- Implemented "Nostalgia Filter" logic (Release Date validation).<br>- Performed EDA on genre distribution and release years.<br>- Cleaned dataset by removing tracks with missing audio previews. |
| **Atul** | **33.3%** | **Vectorization & Indexing:**<br>- Implemented **SBERT** pipeline for Semantic Embeddings.<br>- Set up **FAISS** index for fast vector retrieval.<br>- Validated vector dimensions (384d) and similarity metrics. |
| **Shyamalan** | **33.3%** | **Data Collection (Crawler):**<br>- Built `build_sonic_index.ipynb` crawler using Deezer API.<br>- Implemented dynamic genre fetching to ensure diversity.<br>- Handled API rate limits and data serialization (`sonic_index.pkl`). |

