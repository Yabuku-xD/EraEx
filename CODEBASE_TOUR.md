# EraEx: The Full Pipeline Tour 🐍

This document explains every Python file in the `EraEx` pipeline, from data ingestion to the final search API.

---

## Phase 1: Data Ingestion & Indexing (Offline)

These files are run *once* to build the dataset.

### 1. `notebooks/build_sonic_index.ipynb` (The Crawler)
*   **What it does:** This is the master script that builds our database.
*   **How it works:**
    1.  **Iterates Years:** Loops through 2012, 2013, 2014... up to 2018.
    2.  **Fetches Genres:** Queries Deezer API for active genres (Pop, Rock, Electro, etc.).
    3.  **Search & Filter:** For each Year+Genre combination, it searches for tracks. It checks the `release_date`. If a track is from 2011 or 2019, it's discarded.
    4.  **Vectorization:** It sends the track title+artist to `semantic.py` to get a vector.
    5.  **Persist:** Saves the accepted tracks into `data/indices/sonic_20XX.pkl`.

### 2. `src/audio/semantic.py` (The Text Encoder)
*   **What it does:** Converts text into numbers (vectors) that represent *meaning*.
*   **How it works:**
    1.  **Loads Model:** Uses the `sentence-transformers/all-MiniLM-L6-v2` model (a small, fast BERT model).
    2.  **Encodes:** Takes a string like "Sad piano song" and outputs a list of 384 floating-point numbers.
    3.  **Why:** This allows us to find "Sad" songs even if the title doesn't say "Sad" but implies it semantically.

### 3. `src/audio/processor.py` (The Audio Analyzer)
*   **What it does:** Analyzes the actual sound of the music (the MP3 file).
*   **How it works:**
    1.  **Download:** Fetches the 30-second preview MP3 from Deezer.
    2.  **Load:** Uses `librosa` library to read the audio waveform.
    3.  **Feature Extraction:** Calculates MFCCs (timbre) and Spectral Contrast (texture).
    4.  **Vectorize:** Outputs a 128-dimensional vector representing the "vibe" of the sound. This is used for "Serendipity" mode (finding sound-alikes).

---

## Phase 2: The Core Logic (Runtime)

These files power the actual application when you run the server.

### 4. `src/search/sonic_search.py` (The Vector Engine)
*   **What it does:** Acts as our custom In-Memory Database and Search Engine.
*   **How it works:**
    1.  **Initialization:** On startup, it reads all the `.pkl` files created in Phase 1 and loads ~24,000 tracks into RAM.
    2.  **Search:** When given a query vector, it calculates the **Era-Weighted Tanimoto Fusion Score** between that vector and every single track. This proprietary algorithm combines semantic meaning, audio features, and temporal nostalgia.
    3.  **Sort & Return:** It returns the top N **unique** matches (deduplicated by title/artist). It also handles fetching fresh Album Art URLs from Deezer API on the fly if needed.

### 5. `src/search/enhancer.py` (The Reasoning Bridge)
*   **What it does:** Integrates the Z.AI GLM-4.7 Large Language Model.
*   **How it works:**
    1.  **Prompt Engineering:** It constructs a system prompt telling GLM it is a "Advanced Musicologist".
    2.  **Query Analysis:** It sends the user's raw query (e.g., "late night vibes") to GLM.
    3.  **Structuring:** It forces GLM to return a strict JSON object containing `Mood`, `Genre`, `Keywords`, and `Era Context`. This turns vague inputs into precise search terms.

### 6. `src/ranking/nostalgia.py` (The Gatekeeper)
*   **What it does:** Enforces strict 2012-2018 rules.
*   **How it works:**
    1.  **Date Check:** It parses date strings ("2015-05-12") and ensures they fall *strictly* between Jan 1, 2012 and Dec 31, 2018.
    2.  **Filter:** Even if a search result looks perfect, if this file says "Date: 2011", the track is hidden.

---

## Phase 3: The API & Server (Runtime)

### 7. `src/api/main.py` (The Backend)
*   **What it does:** The Flask Web Server that calls all the other modules.
*   **How it works:**
    1.  **Routing:** Defines URL endpoints like `/sonic` (Search) and `/serendipity` (Random).
    2.  **Orchestration:** When `/sonic` is called:
        *   It calls `enhancer.py` to understand the query.
        *   It calls `semantic.py` to vectorize the enhanced query.
        *   It calls `sonic_search.py` to find matches.
        *   It returns the JSON result to the frontend.

### 8. `run.py` (The Entry Point)
*   **What it does:** The simple script to start the server.
*   **How it works:** Just imports the Flask app from `main.py` and tells it to run on port 5000. It also enables convenient "Hot Reloading" so code changes appear instantly.
