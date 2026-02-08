# Modeling Phase: EraEx Recommendation System (2012-2018)

## 1. Problem Framing and Modeling Goal 🎯

### Problem Statement
In the crowded music streaming landscape, users struggle to discover music that fits a specific "nostalgic vibe" (e.g., "sad piano from 2012-2018"). Standard recommendation engines prioritize mass popularity and recent hits, often burying niche tracks from specific eras. Our goal is to build a **Hybrid Recommendation System** that surfaces high-quality tracks strictly from the **2012-2018 era**, using both audio features (how it sounds) and semantic meaning (lyrics/mood).

### Modeling Goal
To predict the relevance of a candidate track $t$ to a user's query $q$ or listening history $h$, such that the top-K recommendations maximize both similarity and "nostalgia fit."

- **Target Variable**: A relevance score $S_{total} \in [0, 1]$ for each track.
- **Task Type**: Information Retrieval & Recommendation (Hybrid: Content-Based + Collaborative Filtering).
- **Constraints**: 
    - **Strict Temporal Filter**: All recommendations must utilize metadata to verify release date is between Jan 1, 2012, and Dec 31, 2018.
    - **Latency**: Inference must be < 200ms for real-time search.

---

## 2. Data Overview and Feature Considerations 📊

Based on our completed **EDA and Preprocessing (Phase 1)**, we are utilizing the **Deezer API** dataset.

### Dataset Profile
- **Source**: Deezer API (Playlist Crawling: "Top 2012" through "Top 2018").
- **Size**: ~140,000 Tracks (20,000 per year for 7 years).
- **Key Characteristics**:
    - **Imbalanced Genres**: Pop/Rock dominate; niche genres (Jazz, Folk) required targeted crawling.
    - **Missing Audio**: ~15% of tracks lack 30s previews (excluded during preprocessing).
    - **Noise**: "Remastered 2023" versions of 2012 songs (handled by strict date parsing).

### Features Used
1.  **Audio Features (Content-Based)**:
    -   *Source*: 30s MP3 Previews.
    -   *Engineering*: Mel-Frequency Cepstral Coefficients (MFCCs) and Spectral Contrast extracted via `librosa`.
    -   *Vectorization*: 128-dimensional dense vector representing "timbre/mood".
2.  **Semantic Features (Content-Based)**:
    -   *Source*: Track Title, Artist Name, Album Name.
    -   *Engineering*: Sentence-BERT (SBERT) embedding.
    -   *Vectorization*: 384-dimensional dense vector capturing "semantic concept" (e.g., "heartbreak" vs "party").
3.  **Temporal Features (Hard Filter)**:
    -   *Source*: `release_date`.
    -   *Engineering*: Year extraction and range validation (2012 <= Y <= 2018).

---

## 3. Overall Modeling Strategy and Algorithms 🧠

We propose a **"Two-Tower" Hybrid Strategy** to balance accuracy and discovery.

### Stage 1: Candidate Generation (The "Retrievers")
We use two retrieval models in parallel:
1.  **"Sonic" Content-Engine (Vector Search)**:
    -   **Algorithm**: **Cosine Similarity** (approximated via FAISS for speed).
    -   **Why?**: Finds songs that *sound* like the query (e.g., "sad acoustic") even if the user has never interacted with them (Cold Start problem solution).
    -   **Input**: User text query OR Seed Track Vector.
2.  **Collaborative Filtering (CF)**:
    -   **Algorithm**: **Alternating Least Squares (ALS)** (Matrix Factorization).
    -   **Why?**: Captures latent user preferences based on community listening patterns (e.g., "Users who liked Lorde also liked Lana Del Rey").
    -   **Input**: User-Item Interaction Matrix (Implicit Feedback).

### Stage 2: Filtering (The "Time Machine")
-   **Algorithm**: **Boolean Masking**.
-   **Logic**: `Constraint(track) = (ReleaseDate >= 2012-01-01) AND (ReleaseDate <= 2018-12-31)`.
-   **Why?**: Enforces the business logic of "Nostalgia."

### Stage 3: Ranking (The "Merger")
-   **Algorithm**: **Weighted Linear Combination**.
-   **Formula**: $Score = \alpha \cdot Sim_{Audio} + \beta \cdot Sim_{Semantic} + \gamma \cdot Score_{CF}$
-   **Why?**: Allows tuning. For "Search," $\beta$ is high. For "Discover," $\alpha$ (Audio) is high.

---

## 4. Modeling Execution Timeline (Feb 9 – Feb 23) 📅

### Week 1: Model Implementation (Feb 9 – Feb 15)
| Task | Owner | Expected Outcome |
| :--- | :--- | :--- |
| **1. Build Sonic Index** | **Shyamalan & Ann** | `sonic_index.pkl`: Crawler (Shyamalan) + Vectors/Filter (Ann). |
| **2. Implement ALS** | **Shyamalan** | `cf_model.pkl`: Trained Matrix Factorization model on user interactions. |
| **3. Hybrid Ranker Logic** | **Atul** | Python function `rank_tracks(query, user_id)` that merges (FAISS + SBERT) scores. |
| **4. Tuning & Validation** | **Shyamalan** | Hyperparameter tuning ($\alpha, \beta$ weights) to maximize perceived relevance. |

### Week 2: Evaluation & Presentation (Feb 16 – Feb 23)
| Task | Owner | Expected Outcome |
| :--- | :--- | :--- |
| **5. Evaluation Metrics** | **Atul** | Report on `Precision@10` and `Diversity Score` (genre coverage). |
| **6. App Integration** | **Ann** | Connect Model pipeline to Flask API (`/recommend` endpoint) & UI. |
| **7. Slide Deck Prep** | **[Team]** | Final Presentation Draft (Problem -> Data -> Model -> Demo). |
| **8. Final Rehearsal** | **[Team]** | 10-minute presentation walkthrough. |

---

## 5. Peer Evaluation (Team Contribution Breakdown) 🤝

| Team Member | Contribution (%) | Responsibilities & Achievements |
| :--- | :--- | :--- |
| **Ann** | **33.3%** | - Implemented "Nostalgia Filter" logic (Release Date validation).<br>- Engineered Audio (MFCC) & Semantic (SBERT) vector pipelines.<br>- Planned to optimize crawling with parallel processing (ThreadPoolExecutor). |
| **Atul** | **33.3%** | - Implementing FAISS-based vector retrieval.<br>- Planned to implement `HybridRanker` logic (Linear Combination of Audio + Semantic scores). |
| **Shyamalan** | **33.3%** | - Built crawler (Deezer API).<br>- Working on `CollaborativeFiltering` module (ALS Matrix Factorization).<br>- Planned to tune hyperparameters ($\alpha, \beta$) for "Vibe" vs "Accuracy". |

