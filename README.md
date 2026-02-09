# EraEx - The 2012-2018 Nostalgia Machine 📼

A retro-styled music discovery engine that takes you back to the golden era of streaming. Powered by **GLM-4.7 Semantic Reasoning** and a custom **Sonic Vector Index**.

## 🚀 How to Run

### 1. Setup
Install requirements:
```bash
pip install -r requirements.txt
```

Create a `.env` file with your API keys (GLM_API_KEY, etc.).

### 2. Start the App
```bash
python run.py
```
**Open your browser to:** [http://localhost:5000](http://localhost:5000)

## ✨ Core Features (Methodology)

### 🧠 Semantic Search 2.0
-   **GLM-4.7 Integration**: Uses a Large Language Model to understand complex queries like "sad midnight drive" -> interpreting it as "Melancholic, Synthwave, Nocturnal".
-   **Hybrid Retrieval**: Combines semantic understanding with strict keyword filtering.

### 🛡️ The Nostalgia Filter
-   **Strict Temporal Validaton**: Every single track is verified against Deezer's database to ensure it was released strictly between **2012-01-01** and **2018-12-31**.
-   **No Leakage**: "Remastered 2023" compilation tracks are strictly stripped out.

### 🎧 Sonic Indexing
-   **24k+ Track Dataset**: A pre-computed vector index of the era's music.
-   **Audio Fingerprinting**: Uses SBERT and Librosa to match songs by *vibe*, not just text.

### 🎨 Retro UI
-   **Aesthetic**: Frutiger Aero / Vaporwave design language.
-   **Interactive Player**: Integrated Deezer 30s previews with dynamic album art visualization.

## 💻 Tech Stack
-   **AI/ML**: Z.AI GLM-4.7 (Reasoning), SBERT (Embeddings), Librosa (Audio Analysis)
-   **Backend**: Flask, Deezer API (Metadata), NumPy (Vector Ops)
-   **Frontend**: HTML5, CSS3 (Glassmorphism), Vanilla JS
-   **Data**: Pickle-based Vector Index (L2 Cache), Browser Caching (L3)

## 📊 Dataset & Modeling
For detailed methodology, see:
-   [Dataset & Methodology](dataset_and_methodology.md) - Full breakdown of the 24k track index and filtering logic.
-   [Modeling Plan](modeling_plan_readme.md) - Deep dive into the GLM-Sonic hybrid architecture.
