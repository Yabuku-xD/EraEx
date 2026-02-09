# Technical Architecture: The EraEx Engine 🏗️

> **"Not Just Another API Wrapper"**
> *EraEx is a specialized Hybrid Retrieval System that combines Large Language Model (LLM) reasoning with a custom High-Dimensional Vector Search Engine.*

---

## 1. System Overview: The "Two-Brain" Architecture 🧠

Most music apps just search for keywords ("sad song"). EraEx uses two "brains" to find the perfect track:

1.  **The Reasoning Brain (GLM-4.7)**: Understands *intent*. It translates vague user vibes ("late night drive") into concrete musical concepts ("Synthwave", "Nocturnal", "Melancholy").
2.  **The Retrieval Brain (Sonic Index)**: A custom vector database of ~24,000 tracks that matches songs based on their *mathematical similarity* to those concepts, not just text matching.

```mermaid
graph TD
    User[User Query: "late night vibes"] --> Frontend[Frontend (Glassmorphism UI)]
    Frontend --> |API Call| Backend[Flask API]
    
    subgraph "Reasoning Layer"
        Backend --> |Raw Query| GLM[GLM-4.7 Reasoning Engine]
        GLM --> |"Mood: Melancholic, Genre: Synthwave"| ExpandedQuery[Structured Intent]
    end
    
    subgraph "Retrieval Layer (The Sonic Engine)"
        ExpandedQuery --> |Encode| SBERT[SBERT Encoder]
        SBERT --> |Vector Search| Index[Sonic Vector Index (24k Tracks)]
        Index --> |Top 50 Matches| Candidates[Candidate Tracks]
    end
    
    subgraph "Enrichment Layer"
        Candidates --> |Fetch Metadata| Deezer[Deezer API]
        Deezer --> |Cover Art & Preview| FinalResults[Enriched Results]
    end
    
    FinalResults --> Frontend
```

---

## 2. Codebase Tour: The "Secret Sauce" 🌶️

Here is where the magic happens. All core logic is in `src/`.

### A. The Search Engine (`src/search/`)
*   **`sonic_search.py` (The Core Engine)**:
    *   **Purpose**: This is our custom Vector Database implementation.
    *   **How it works**: It loads the ~24,000 track vectors from disk (`.pkl` files) into memory (RAM). When you search, it performs cosine similarity calculations between your query vector and all 24k tracks in milliseconds.
    *   **Why custom?**: We needed strict control over the *Release Date* filter (2012-2018) which off-the-shelf vector DBs struggle to combine with vector search efficiently.

*   **`enhancer.py` (The Reasoning Bridge)**:
    *   **Purpose**: The interface to Z.AI's GLM-4.7 model.
    *   **Logic**: It constructs a rigorous prompt ("You are a music expert...") to force the LLM to output structured JSON instead of chatty text. It handles the "Reasoning" step.

### B. The Feature Engineering (`src/audio/`)
*   **`semantic.py` (The Text Encoder)**:
    *   **Technique**: Uses `all-MiniLM-L6-v2` (Sentence-BERT).
    *   **Function**: Converts text like "Sad songs about rain" into a 384-dimensional vector of numbers. This allows us to match "Sad" with "Melancholy" mathematically.

*   **`processor.py` (The Audio Analyzer)**:
    *   **Technique**: Uses `librosa` to analyze MP3 waveforms.
    *   **Function**: Extracts **MFCCs** (Timbre/Texture) and **Spectral Contrast**. This powers the "Serendipity" mode (finding songs that *sound* similar even if they have different genres).

---

## 3. The Data Layer: Sonic Indices (`data/indices/`) 💾

We do NOT rely on external APIs for search. We built our own index.

*   **Format**: `.pkl` (Python Pickle files) - highly optimized binary format for serialized objects.
*   **Structure**:
    *   `sonic_2012.pkl` ... `sonic_2018.pkl`: We sharded the index by year.
    *   **Total Volume**: ~24,365 curated tracks.
    *   **Content**: Each entry contains:
        *   `id`: Track ID
        *   `semantic_vector`: The "Meaning" (384 floats)
        *   `audio_vector`: The "Vibe" (128 floats)
        *   `metadata`: Title, Artist, Date.

**Why this matters**:
*   **Speed**: Search is local and instant (in-memory).
*   **Reliability**: If Deezer's search API goes down, *our* search still works (we own the index).
*   **Quality**: We pre-filtered "garbage" tracks (karaoke, covers, wrong dates) during the indexing phase.

---

## 4. Why This Architecture? (The "Sell") 💼

| Feature | Standard API Wrapper | EraEx Hybrid Engine |
| :--- | :--- | :--- |
| **Search Quality** | Matches keywords only ("Sad" finds songs with "Sad" in title). | Matches **concepts** ("Sad" finds "Heartbreak", "Grief", "Melancholy"). |
| **Era Accuracy** | Poor (APIs return "Remastered 2023" versions). | **100% Verified** (We filtered by date during indexing). |
| **Privacy/Control** | Data lives on external servers. | **Self-Hosted Index**. The knowledge graph is yours. |
| **Intelligence** | Dumb string matching. | **Reasoning-Enhanced**. Understands slang, vibes, and culture. |

---

## 5. Future Scalability 🚀

Currently, we use in-memory Pickle files (perfect for <100k tracks). To scale to millions:
1.  **Migration**: Move `.pkl` data to **Parquet** (columnar storage) for efficient disk I/O.
2.  **Indexing**: Use **FAISS** (Facebook AI Similarity Search) instead of raw NumPy arrays for sub-millisecond search at scale.
