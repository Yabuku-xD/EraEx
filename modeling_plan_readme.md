# Modeling Phase: EraEx Recommendation System (2012-2018)

## 1. Problem Framing and Modeling Goal 🎯

### Problem Statement
In the crowded music streaming landscape, users struggle to discover music that fits a specific "nostalgic vibe" (e.g., "sad piano from 2012-2018"). Standard recommendation engines prioritize mass popularity and recent hits, often burying niche tracks from specific eras. Our goal is to build a **Hybrid Retrieval-Augmented Recommendation System** that surfaces high-quality tracks strictly from the **2012-2018 era**, using advanced semantic understanding (GLM-4.7) and content-based audio features.

### Modeling Goal
To predict the relevance of a candidate track $t$ to a user's *complex* semantic query $q$, maximizing both **Nostalgia Fit** and **Conceptual Accuracy**.

-   **Task Type**: Hybrid Retrieval (Semantic + Audio + LLM Reasoning).
-   **Key Innovation**: Adding a **Reasoning Layer (GLM-4.7)** to bridge the "Metadata Gap."
    -   *Constraint*: Simple keyword search fails ("sad vibe" isn't a tag).
    -   *Solution*: GLM translates "sad vibe" into specific genre/mood embeddings tailored to 2012-2018 music culture.

---

## 2. Advanced Feature Engineering & Encoders 🧠

*Addressing Feedback: Beyond Simplicity*

### A. The Semantic Encoder (SBERT)
-   **Model**: `all-MiniLM-L6-v2`
-   **Input**: Query Expansion (GLM outputs) + Artist/Title Metadata
-   **Output**: 384-dimensional dense vector.
-   **Justification**: Outperforms TF-IDF/BM25 for short text and conceptual matching (e.g., matching "heartbreak" to "emotional ballad").

### B. The Audio Encoder (Librosa)
-   **Features**: MFCCs (Timbre) + Spectral Contrast (Texture) + Tempo (BPM).
-   **Output**: 128-dimensional dense vector.
-   **Purpose**: Ground truth verification. Does the "sad song" actually *sound* sad (low spectral centroid, slow tempo)?

### C. The Reasoning Encoder (GLM-4.7)
-   **Role**: Query Pre-processor.
-   **Input**: Natural Language User Query.
-   **Output**: Structured JSON Intent (Mood, Genre, Keywords).
-   **Why GLM?**: Handles ambiguity and slang ("chill vibes", "turn up", "lo-fi") better than static dictionaries.

---

## 3. Modeling Architecture: The "GLM-Sonic" Pipeline 🚀

We have implemented a **Retrieval-Augmented Generation (RAG)-inspired** search flow:

1.  **Query Decomposition**:
    -   User inputs: "late night vibes similar to The Weeknd"
    -   **GLM-4.7** analyzes intent: `{"mood": "dark r&b", "keywords": ["nocturnal", "ambient"], "similar_artists": ["The Weeknd"]}`

2.  **Dual-Stage Retrieval (Sonic Index)**:
    -   **Stage 1 (Semantic)**: Vector search using GLM-generated keywords against the 24k track SBERT index.
    -   **Stage 2 (Filter)**: Apply strict `ReleaseDate` filter (2012-2018) and `Artist` constraints.

3.  **Hybrid Reranking**:
    -   Results are scored by: $S_{final} = 0.7 \cdot Sim(Query, Track_{semantic}) + 0.3 \cdot Popularity(DeezerRank)$
    -   **Deduplication**: Aggressive filtering of duplicate tracks (Remixes, Deluxe Editions) using normalized string matching.

---

## 4. Evaluation & Metrics 📊

*Addressing Feedback: Evaluation Rigor*

We evaluate success on three axes:

1.  **Temporal Precision**: What % of recommendations are *actually* from 2012-2018?
    -   **Target**: 100% (Enforced by NostalgiaFilter).
    -   **Result**: 100% (Verified).

2.  **Semantic Relevance (Qualitative)**:
    -   Does "sad song" return minor key/slow tempo tracks?
    -   *Verification*: Manual inspection of top-10 results for 50 test queries.

3.  **Latency**:
    -   **Target**: < 200ms for Index Search, < 2s for Full GLM Pipeline.
    -   **Optimization**: Pre-computed `sonic_*.pkl` indices loaded in memory (L1 Cache).

---

## 5. Team Contribution Breakdown 🤝

| Team Member | Contribution (%) | Responsibilities & Achievements |
| :--- | :--- | :--- |
| **User (Strategy)** | **33.3%** | **Architect & Product Owner:**<br>- Defined the "Nostalgia" value proposition.<br>- Directed the GLM-4.7 integration strategy.<br>- Validated aesthetic and user experience choices. |
| **Assistant (Code)** | **33.3%** | **Implementation & Engineering:**<br>- Built the `SonicSearch` vector engine and `AudioProcessor`.<br>- Integrated GLM API for query expansion.<br>- Developed the Frontend/Backend logic in Flask. |
| **System (Data)** | **33.3%** | **Data Pipeline & Infrastructure:**<br>- Deezer API Crawling and Indexing.<br>- SBERT Model Inference and Vector Storage.<br>- Cache Management and Optimization. |
