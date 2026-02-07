# Advanced Music Recommendation System: Architecture & Implementation Plan

## Project Overview

This document outlines a sophisticated music recommendation system that transcends keyword-based matching to deliver mood-aware, Spotify-quality recommendations. The system is designed to work with a challenging SoundCloud dataset (70-80% null values, dead links) while maintaining <500ms API response times on consumer hardware.

**Core Principle**: Add architectural complexity through multi-stage retrieval, hybrid filtering, and audio signal processing—NOT through LLM/API dependencies.

---

## Current State Assessment

### What You Have
- **Embeddings**: SBERT (Sentence-BERT) text embeddings from metadata
- **Vector Search**: FAISS for approximate nearest neighbor search
- **Dataset**: SoundCloud crawl with columns: `title`, `artist`, `year`, `created_at`, `genre`, `tags`, `description`, `playback_count`, `permalink_url`, `extracted_vibe_text`
- **Challenge**: 70-80% null values in `genre`, `tags`, `description` columns
- **Problem**: Keyword matching instead of mood understanding

### What You're Missing
- **Audio Features**: No acoustic signal analysis (tempo, energy, spectral features)
- **Hybrid Architecture**: No combination of collaborative + content-based filtering
- **Multi-Stage Retrieval**: No ranking and re-ranking pipeline
- **Mood Mapping**: No emotional dimension modeling (valence-arousal space)
- **Dead Link Handling**: No healing system for broken audio URLs

---

## System Architecture

### The Five-Stage Pipeline

```
┌────────────────────────────────────────────────────────────────┐
│                     USER QUERY                                  │
│                "i miss my ex" / "upbeat morning"                │
└──────────────────┬─────────────────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────────────────┐
│  STAGE 1: QUERY UNDERSTANDING & MOOD EXTRACTION                │
│  - Extract emotional dimensions (valence, arousal)              │
│  - Map to 2D mood space without keyword matching               │
│  - Parse temporal/contextual cues (morning, workout, sad)      │
│  Output: [valence_score, arousal_score, context_tags]          │
└──────────────────┬─────────────────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────────────────┐
│  STAGE 2: CANDIDATE RETRIEVAL (Multi-Index)                    │
│  - Query 3 FAISS indices in parallel:                          │
│    1. Text embeddings (SBERT) - metadata                       │
│    2. Audio embeddings (acoustic features)                     │
│    3. Mood embeddings (valence-arousal vectors)                │
│  - Retrieve top 500 candidates from each (1500 total)          │
│  - Diversity filtering to avoid artist/genre clustering        │
│  Output: ~800-1000 diverse candidates                          │
└──────────────────┬─────────────────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────────────────┐
│  STAGE 3: SCORING & HYBRID WEIGHTING                           │
│  - Combine scores from multiple signals:                       │
│    * Text similarity (20%)                                     │
│    * Audio feature matching (35%)                              │
│    * Mood space distance (30%)                                 │
│    * Collaborative signals (10%)                               │
│    * Popularity penalty (5%)                                   │
│  - Apply learned weights per query type                        │
│  Output: Top 100 scored candidates                             │
└──────────────────┬─────────────────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────────────────┐
│  STAGE 4: RE-RANKING & DIVERSIFICATION                         │
│  - Maximal Marginal Relevance (MMR) for diversity              │
│  - Temporal dynamics filtering (avoid repetition)              │
│  - Dead link detection & removal                               │
│  - Artist/album spacing                                        │
│  Output: Top 50 final recommendations                          │
└──────────────────┬─────────────────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────────────────┐
│  STAGE 5: POST-PROCESSING & API RESPONSE                       │
│  - Verify audio availability (cached status)                   │
│  - Hydrate metadata from healing system                        │
│  - Format response with confidence scores                      │
│  Output: JSON response in <500ms                               │
└────────────────────────────────────────────────────────────────┘
```

---

## Component Deep Dive

### Component 1: Data Healing & Preprocessing

**Objective**: Transform the messy SoundCloud dataset into a clean, feature-rich corpus.

#### 1.1 Dead Link Detection & Healing
**Preprocessing Phase (One-Time)**
1. **Audio Availability Check**
   - Attempt HEAD request to each `permalink_url`
   - Mark status: `VALID`, `DEAD`, `MOVED`, `REQUIRES_AUTH`
   - Store in new column: `url_status`, `last_checked_timestamp`
   - Expected survival rate: 30-50% of tracks

2. **Dead Link Recovery Strategies**
   - **Strategy A**: Search for alternate sources using `title + artist`
   - **Strategy B**: Use SoundCloud API (if available) to find relocated tracks
   - **Strategy C**: Find similar tracks and inherit their metadata
   - Store mapping: `original_url → recovered_url` in lookup table

3. **Filtering Music vs Non-Music**
   - **Heuristic 1**: If `playback_count > 100` → likely music
   - **Heuristic 2**: If `title` contains keywords like "podcast", "interview", "chapter", "episode" → exclude
   - **Heuristic 3**: If `extracted_vibe_text` contains "spoken words", "interview", "chapter" → exclude
   - **Heuristic 4**: Cross-reference with `genre` column (if not null) - exclude "Spoken words", "Audiobooks"
   - Manual review of edge cases with 1000-sample validation set

#### 1.2 Null Value Imputation
**The 70-80% Null Problem**

| Column | Null % | Imputation Strategy |
|--------|--------|---------------------|
| `genre` | ~75% | Use artist's most common genre + clustering of audio features |
| `tags` | ~80% | Extract from `title`, `description`, `extracted_vibe_text` using TF-IDF |
| `description` | ~70% | Use `extracted_vibe_text` as fallback; leave null if both missing |

**Imputation Process**
1. **Genre Filling**
   ```
   Step 1: Group by artist → assign most frequent genre to that artist's tracks
   Step 2: For remaining nulls, use audio feature clustering (k-means with k=20)
           Map each cluster to a genre label using non-null samples
   Step 3: If still null, assign "Unknown" or "Electronic" (most common in SoundCloud)
   ```

2. **Tag Extraction from Text**
   ```
   Step 1: Combine title + description + extracted_vibe_text into unified text field
   Step 2: Run TF-IDF with n-gram range (1, 2) to extract salient phrases
   Step 3: Filter top 5-10 phrases per track, clean with stopwords removal
   Step 4: Manually curate tag vocabulary (500-1000 common music tags)
   Step 5: Map extracted phrases to curated vocabulary using fuzzy matching
   ```

3. **Description Enrichment**
   ```
   Step 1: If description null but extracted_vibe_text exists → copy
   Step 2: If both null, generate synthetic description:
           "A {genre} track by {artist} from {year}"
   ```

#### 1.3 Feature Engineering for Metadata
**Create Rich Metadata Regardless of Nulls**

1. **Temporal Features**
   - `release_year`: Extract from `year` or `created_at`
   - `era_category`: ["1950s", "1960s", ..., "2020s"]
   - `recency_score`: Decaying function based on age (newer tracks get slight boost)

2. **Popularity Signals**
   - `playback_count`: Normalize to [0, 1] range with log scaling
   - `popularity_tier`: ["Underground", "Niche", "Popular", "Viral"] based on percentile
   - `engagement_rate`: If available, calculate likes/plays ratio

3. **Artist Authority**
   - `artist_track_count`: Number of tracks by artist in dataset
   - `artist_avg_playback`: Average playback count across artist's catalog
   - `is_prolific`: Binary flag if artist has >10 tracks

4. **Text Density Score**
   - `metadata_richness`: Score 0-1 based on non-null fields
   - `text_length`: Combined character count of title + description + tags

---

### Component 2: Audio Feature Extraction

**Objective**: Extract acoustic features from audio files to enable mood-based matching.

#### 2.1 Audio Processing Pipeline

**Assumptions**
- Only ~30-50% of tracks have working URLs (based on dead link detection)
- For tracks without audio: Use metadata-based feature imputation

**Feature Categories**

| Feature Group | Features | Purpose |
|---------------|----------|---------|
| **Tempo & Rhythm** | BPM, onset strength, beat regularity | Energy and danceability |
| **Spectral** | MFCC (13 coefficients), spectral centroid, spectral rolloff, spectral flux | Timbre and texture |
| **Energy** | RMS energy, zero crossing rate | Intensity and activity |
| **Harmony** | Chroma features (12 bins), tonnetz | Tonality and mood |
| **Loudness** | Loudness (LUFS), dynamic range | Perceived volume |

**Extraction Code Flow** (Librosa)
```
FOR each track with valid audio URL:
    1. Download audio or stream first 30 seconds (sufficient for features)
    2. Load with librosa.load(sr=22050, mono=True)
    3. Extract features:
       - BPM: librosa.beat.beat_track()
       - MFCC: librosa.feature.mfcc(n_mfcc=13, n_fft=2048, hop_length=512)
       - Spectral centroid: librosa.feature.spectral_centroid()
       - Spectral rolloff: librosa.feature.spectral_rolloff()
       - Chroma: librosa.feature.chroma_stft()
       - RMS: librosa.feature.rms()
       - Zero crossing rate: librosa.feature.zero_crossing_rate()
    4. Aggregate temporal features using statistics:
       - Mean, median, std, min, max, 25th/75th percentiles
    5. Result: 50-80 dimensional audio feature vector per track
    6. Store in HDF5 or Parquet file: {track_id: audio_features}
```

**Computational Estimate**
- Hardware: Ryzen 5 9000X, 32GB RAM, RX7800XT
- Processing: ~2-5 seconds per track (with 30-sec audio samples)
- Total time for 100K valid tracks: ~6-14 hours (parallelizable)
- Storage: ~80 features × 4 bytes × 100K tracks = ~32 MB (negligible)

#### 2.2 Audio Feature Imputation (For Dead Links)

**Problem**: 50-70% of tracks have no working audio
**Solution**: Use metadata-to-audio mapping learned from valid tracks

**Imputation Strategy**
1. **Genre-Based Prototypes**
   ```
   FOR each genre:
       Compute mean and std of audio features from tracks with valid audio
       Store as genre_prototype[genre_name] = {mean_vector, std_vector}
   
   FOR track with dead link:
       IF genre is known:
           Impute audio_features = genre_prototype[genre].mean_vector + 
                                   random_noise * genre_prototype[genre].std_vector
       ELSE:
           Use global mean of all audio features
   ```

2. **k-NN Metadata Matching**
   ```
   FOR track with dead link:
       Find k=10 nearest neighbors based on text embeddings (SBERT)
       Filter to only neighbors with valid audio features
       Impute audio_features = weighted_average(neighbors' audio_features)
       Weight by text similarity distance
   ```

3. **Tag-to-Audio Regression**
   ```
   Train a regression model (Random Forest or XGBoost):
       Input: Text features (TF-IDF of tags/description/title) + metadata
       Output: Audio feature vector (50-80 dimensions)
       Training data: Tracks with valid audio features
   
   Use trained model to predict audio features for tracks without audio
   ```

**Validation**: Reserve 10% of tracks with valid audio for validation. Check if imputed features are within reasonable range (mean absolute error, cosine similarity).

---

### Component 3: Mood Space Construction

**Objective**: Map all tracks into a 2D emotional space (valence-arousal) to enable mood-based retrieval.

#### 3.1 The Valence-Arousal Model

**Dimensions**
- **Valence**: Positive (happy, joyful) ↔ Negative (sad, melancholic) [-1, +1]
- **Arousal**: High energy (excited, angry) ↔ Low energy (calm, sleepy) [-1, +1]

**Mood Quadrants**
```
         High Arousal
              ^
              |
   Q2         |         Q1
   Angry      |      Excited
   Energetic  |      Joyful
   Tense      |      Upbeat
              |
--------------+--------------> Valence
   Negative   |    Positive
              |
   Q3         |         Q4
   Sad        |      Calm
   Melancholic|      Peaceful
   Depressed  |      Relaxed
              |
              v
         Low Arousal
```

#### 3.2 Mood Score Calculation

**Input Features → Mood Mapping**

| Audio Feature | Valence Contribution | Arousal Contribution |
|---------------|----------------------|----------------------|
| **BPM** | Neutral | Positive (fast → high arousal) |
| **Major/Minor Mode** | Positive (major), Negative (minor) | Neutral |
| **Spectral Centroid** | Slightly positive (bright → happy) | Positive (bright → energetic) |
| **RMS Energy** | Slightly positive (loud → exciting) | Positive (loud → energetic) |
| **Spectral Rolloff** | Positive (high rolloff → bright) | Positive |
| **Chroma Std** | Negative (dissonance → negative) | Positive (variance → complex) |

**Mood Calculation Algorithm**
```python
def calculate_mood_scores(audio_features):
    """
    Maps audio features to valence and arousal scores
    Returns: (valence, arousal) each in [-1, +1]
    """
    # Normalize all features to [0, 1] first
    bpm_norm = normalize(audio_features['bpm'], min=60, max=200)
    energy_norm = normalize(audio_features['rms_mean'], min=0, max=0.5)
    brightness_norm = normalize(audio_features['spectral_centroid_mean'], 
                                 min=500, max=5000)
    
    # Arousal calculation (energy-based)
    arousal = (
        0.4 * bpm_norm +           # Fast tempo → high arousal
        0.3 * energy_norm +         # Loud → high arousal
        0.2 * brightness_norm +     # Bright → high arousal
        0.1 * audio_features['onset_strength']  # Rhythmic → high arousal
    )
    arousal = 2 * arousal - 1  # Scale to [-1, 1]
    
    # Valence calculation (tonality + brightness)
    # Note: Would need mode detection (major/minor) for better accuracy
    # Using proxy: spectral features + harmonic content
    valence = (
        0.4 * brightness_norm +     # Bright → positive
        0.3 * (1 - chroma_std_norm) +  # Low dissonance → positive
        0.2 * energy_norm +         # Energetic → slightly positive
        0.1 * (spectral_flux < threshold)  # Stable → positive
    )
    valence = 2 * valence - 1  # Scale to [-1, 1]
    
    return (valence, arousal)
```

**Calibration**
- Manually label 500-1000 tracks with mood tags ("sad", "happy", "energetic", "calm")
- Map these tags to valence-arousal coordinates
- Train a regression model (Ridge/Lasso) to fine-tune the feature-to-mood mapping
- Apply learned model to all tracks

#### 3.3 Mood Embedding Creation

**Create 2D Mood Vectors**
```
FOR each track:
    mood_vector = [valence_score, arousal_score]
    mood_embedding[track_id] = mood_vector
```

**Store in FAISS Index**
```python
# Create 2D FAISS index for mood space
import faiss
import numpy as np

mood_vectors = np.array([mood_embedding[tid] for tid in track_ids])
# Use flat index for exact search (only 2D, very fast)
mood_index = faiss.IndexFlatL2(2)
mood_index.add(mood_vectors.astype('float32'))
```

**Why This Works**
- 2D mood space is computationally trivial (<1ms search on 1M tracks)
- Captures emotional essence independent of keywords
- Enables geometric reasoning: "find songs near (valence=-0.7, arousal=-0.3) for 'i miss my ex'"

---

### Component 4: Multi-Index Retrieval System

**Objective**: Query multiple specialized indices in parallel to retrieve diverse candidates.

#### 4.1 The Three-Index Architecture

**Index 1: Text Embeddings (SBERT)**
- **Content**: SBERT embeddings from title + description + tags + extracted_vibe_text
- **Dimension**: 384 (typical for sentence-transformers models)
- **FAISS Type**: IVF (Inverted File Index) with PQ (Product Quantization)
  - `nlist=1024` (1024 clusters)
  - `nprobe=32` (search 32 clusters per query)
  - `PQ code size=64` (compression)
- **Query Speed**: ~10-30ms for top 500 results on 1M tracks

**Index 2: Audio Embeddings**
- **Content**: 50-80D audio feature vectors (with imputation)
- **Dimension**: 64 (after PCA reduction from 80D)
- **FAISS Type**: HNSW (Hierarchical Navigable Small World)
  - `M=32` (connections per node)
  - `efConstruction=200` (build quality)
  - `efSearch=128` (search quality)
- **Query Speed**: ~20-50ms for top 500 results

**Index 3: Mood Embeddings**
- **Content**: 2D valence-arousal vectors
- **Dimension**: 2
- **FAISS Type**: Flat (exact search, very fast due to low dimensionality)
- **Query Speed**: <1ms for top 500 results

#### 4.2 Parallel Retrieval Strategy

**Query Processing**
```python
def retrieve_candidates(query_text, query_mood=None):
    """
    Parallel retrieval from three indices
    """
    # Stage 1: Parse query into components
    text_query = extract_text_query(query_text)  # "rock songs"
    mood_query = extract_mood_query(query_text) or query_mood  # "sad" → valence=-0.7
    
    # Stage 2: Generate query vectors
    text_embedding = sbert_model.encode(text_query)
    audio_query_vector = None  # Initially unknown
    mood_query_vector = map_mood_to_vector(mood_query)  # [valence, arousal]
    
    # Stage 3: Parallel index queries (use threading)
    import concurrent.futures
    
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Query text index
        future_text = executor.submit(
            text_index.search, text_embedding, k=500
        )
        
        # Query audio index (if we have audio query features)
        # For text-only queries, use centroid of mood space
        if audio_query_vector is not None:
            future_audio = executor.submit(
                audio_index.search, audio_query_vector, k=500
            )
        else:
            # Use text-to-audio mapping or skip
            future_audio = executor.submit(lambda: (None, None))
        
        # Query mood index
        future_mood = executor.submit(
            mood_index.search, mood_query_vector, k=500
        )
        
        # Collect results
        results['text'] = future_text.result()  # (distances, indices)
        results['audio'] = future_audio.result()
        results['mood'] = future_mood.result()
    
    return results
```

**Candidate Aggregation**
```python
def aggregate_candidates(results):
    """
    Merge candidates from three indices with diversity filtering
    """
    # Collect all unique track IDs
    candidates = set()
    candidate_scores = {}
    
    # Text results (top 500)
    for idx, dist in zip(results['text'][1][0], results['text'][0][0]):
        if idx not in candidates:
            candidates.add(idx)
            candidate_scores[idx] = {
                'text_dist': dist,
                'text_rank': len(candidates)
            }
    
    # Audio results (top 500)
    if results['audio'][0] is not None:
        for idx, dist in zip(results['audio'][1][0], results['audio'][0][0]):
            if idx not in candidates:
                candidates.add(idx)
                candidate_scores[idx] = {'audio_dist': dist, 'audio_rank': len(candidates)}
            else:
                candidate_scores[idx]['audio_dist'] = dist
    
    # Mood results (top 500)
    for idx, dist in zip(results['mood'][1][0], results['mood'][0][0]):
        if idx not in candidates:
            candidates.add(idx)
            candidate_scores[idx] = {'mood_dist': dist, 'mood_rank': len(candidates)}
        else:
            candidate_scores[idx]['mood_dist'] = dist
    
    # Diversity filtering: Remove excessive tracks from same artist
    candidates = apply_artist_diversity(candidates, max_per_artist=10)
    
    return list(candidates)[:1000]  # Cap at 1000 for next stage
```

**Expected Performance**
- Text index query: ~20ms
- Audio index query: ~30ms
- Mood index query: <1ms
- Parallel execution: ~35ms (max of the three + overhead)
- Aggregation: ~5ms
- **Total Stage 2 Time: ~40-50ms**

---

### Component 5: Hybrid Scoring & Weighting

**Objective**: Combine signals from multiple sources to rank candidates.

#### 5.1 Score Components

**1. Text Similarity Score**
```python
text_score = 1 / (1 + text_distance)  # Convert distance to similarity
normalized_text_score = (text_score - min_score) / (max_score - min_score)
```

**2. Audio Similarity Score**
```python
# Cosine similarity between query audio features and candidate audio features
audio_score = cosine_similarity(query_audio_vector, candidate_audio_vector)
# Already in [0, 1] range
```

**3. Mood Distance Score**
```python
# Euclidean distance in 2D mood space
mood_distance = sqrt((v1 - v2)^2 + (a1 - a2)^2)
mood_score = 1 / (1 + mood_distance)  # Convert to similarity
```

**4. Collaborative Signal (Implicit)**
```python
# Use playback count as proxy for collaborative filtering
# Assumption: popular tracks within a cluster are good recommendations
collab_score = log(1 + playback_count) / log(1 + max_playback_count)
```

**5. Popularity Penalty (Anti-Filter Bubble)**
```python
# Penalize extremely popular tracks to promote diversity
popularity_penalty = 1 - (0.3 * is_viral)  # Reduce score by 30% if viral
```

#### 5.2 Weighted Combination

**Query-Type Adaptive Weighting**

| Query Type | Text Weight | Audio Weight | Mood Weight | Collab Weight | Popularity Penalty |
|------------|-------------|--------------|-------------|---------------|--------------------|
| **Text-heavy** ("rock guitar solo") | 0.50 | 0.25 | 0.15 | 0.10 | 0.05 |
| **Mood-heavy** ("i miss my ex", "pump up") | 0.15 | 0.35 | 0.35 | 0.10 | 0.05 |
| **Genre** ("electronic music") | 0.35 | 0.30 | 0.20 | 0.10 | 0.05 |
| **Artist** ("songs like Radiohead") | 0.40 | 0.30 | 0.15 | 0.10 | 0.05 |
| **Default** (ambiguous query) | 0.20 | 0.35 | 0.30 | 0.10 | 0.05 |

**Query Type Detection**
```python
def detect_query_type(query_text):
    """
    Classify query into type for adaptive weighting
    """
    # Mood keywords
    mood_keywords = ['sad', 'happy', 'angry', 'calm', 'miss', 'energetic', 
                     'chill', 'pump', 'relax', 'workout', 'morning']
    
    # Genre keywords
    genre_keywords = ['rock', 'electronic', 'jazz', 'hip hop', 'classical', 
                      'metal', 'indie', 'pop', 'reggae']
    
    # Artist detection (proper nouns, capitalized)
    has_artist = any(word[0].isupper() for word in query_text.split())
    
    # Specific attributes
    attribute_keywords = ['guitar', 'piano', 'vocals', 'instrumental', 'remix']
    
    if any(kw in query_text.lower() for kw in mood_keywords):
        return 'mood-heavy'
    elif any(kw in query_text.lower() for kw in genre_keywords):
        return 'genre'
    elif has_artist:
        return 'artist'
    elif any(kw in query_text.lower() for kw in attribute_keywords):
        return 'text-heavy'
    else:
        return 'default'
```

**Final Scoring**
```python
def compute_hybrid_score(candidate_id, query_type, scores):
    """
    Compute weighted hybrid score
    """
    weights = WEIGHT_MATRIX[query_type]  # Get weights for query type
    
    final_score = (
        weights['text'] * scores.get('text_score', 0) +
        weights['audio'] * scores.get('audio_score', 0) +
        weights['mood'] * scores.get('mood_score', 0) +
        weights['collab'] * scores.get('collab_score', 0)
    ) * scores.get('popularity_penalty', 1.0)
    
    return final_score
```

**Scoring Time Complexity**
- Score computation per candidate: O(1) (just weighted sum)
- Sorting 1000 candidates: O(n log n) = ~10ms
- **Total Stage 3 Time: ~15-20ms**

---

### Component 6: Re-Ranking & Diversification

**Objective**: Refine top candidates for diversity and quality.

#### 6.1 Maximal Marginal Relevance (MMR)

**Problem**: Top results may cluster around same artist/genre
**Solution**: MMR balances relevance and diversity

**Algorithm**
```python
def maximal_marginal_relevance(candidates, scores, lambda_param=0.7, top_k=50):
    """
    MMR re-ranking for diversity
    lambda_param: trade-off between relevance (1.0) and diversity (0.0)
    """
    selected = []
    remaining = set(candidates)
    
    # Start with highest scoring candidate
    best_candidate = max(remaining, key=lambda c: scores[c])
    selected.append(best_candidate)
    remaining.remove(best_candidate)
    
    while len(selected) < top_k and remaining:
        # For each remaining candidate, compute MMR score
        mmr_scores = {}
        for candidate in remaining:
            # Relevance to query
            relevance = scores[candidate]
            
            # Diversity: max similarity to already selected items
            max_similarity = max([
                compute_similarity(candidate, selected_item)
                for selected_item in selected
            ])
            
            # MMR formula
            mmr_scores[candidate] = (
                lambda_param * relevance - 
                (1 - lambda_param) * max_similarity
            )
        
        # Select candidate with highest MMR score
        next_candidate = max(remaining, key=lambda c: mmr_scores[c])
        selected.append(next_candidate)
        remaining.remove(next_candidate)
    
    return selected
```

**Similarity Function**
```python
def compute_similarity(track1_id, track2_id):
    """
    Multi-faceted similarity for MMR
    """
    # Artist similarity (binary)
    artist_sim = 1.0 if track1.artist == track2.artist else 0.0
    
    # Audio feature similarity
    audio_sim = cosine_similarity(
        audio_features[track1_id], 
        audio_features[track2_id]
    )
    
    # Mood similarity
    mood_sim = 1 - euclidean_distance(
        mood_vectors[track1_id], 
        mood_vectors[track2_id]
    ) / 2.0  # Normalize to [0, 1]
    
    # Weighted combination
    similarity = 0.5 * artist_sim + 0.3 * audio_sim + 0.2 * mood_sim
    return similarity
```

#### 6.2 Artist/Album Spacing

**Prevent Consecutive Same-Artist Tracks**
```python
def apply_artist_spacing(ranked_list, min_gap=3):
    """
    Ensure at least min_gap positions between tracks from same artist
    """
    result = []
    artist_last_position = {}
    
    for track in ranked_list:
        artist = get_artist(track)
        
        # Check if artist appeared recently
        if artist in artist_last_position:
            gap = len(result) - artist_last_position[artist]
            if gap < min_gap:
                # Skip this track for now, will add later
                continue
        
        result.append(track)
        artist_last_position[artist] = len(result) - 1
    
    return result[:50]  # Return top 50
```

#### 6.3 Dead Link Filtering

**Real-Time Availability Check**
```python
def filter_dead_links(ranked_list, url_status_cache):
    """
    Remove tracks with dead URLs
    Use cached status from preprocessing
    """
    valid_tracks = []
    for track in ranked_list:
        status = url_status_cache.get(track.id, 'UNKNOWN')
        if status in ['VALID', 'MOVED', 'REQUIRES_AUTH']:
            valid_tracks.append(track)
        # Skip if status is 'DEAD'
    
    return valid_tracks
```

**Cache Structure** (In-Memory Redis or SQLite)
```
url_status_cache = {
    track_id: {
        'status': 'VALID' | 'DEAD' | 'MOVED' | 'REQUIRES_AUTH',
        'last_checked': timestamp,
        'redirect_url': alternate_url (if moved)
    }
}
```

**Re-Ranking Time**
- MMR computation: ~20-30ms (100 candidates, greedy selection)
- Artist spacing: ~5ms
- Dead link filtering: ~1ms (cache lookup)
- **Total Stage 4 Time: ~30-40ms**

---

### Component 7: Query Understanding & Mood Extraction

**Objective**: Parse natural language query into structured mood dimensions.

#### 7.1 Mood Keyword Mapping

**Lexicon-Based Approach**
```python
MOOD_LEXICON = {
    # Valence (positive/negative)
    'happy': {'valence': 0.8, 'arousal': 0.5},
    'sad': {'valence': -0.7, 'arousal': -0.3},
    'miss': {'valence': -0.6, 'arousal': -0.2},  # "i miss my ex"
    'angry': {'valence': -0.5, 'arousal': 0.8},
    'calm': {'valence': 0.3, 'arousal': -0.7},
    'relaxed': {'valence': 0.5, 'arousal': -0.6},
    'excited': {'valence': 0.7, 'arousal': 0.8},
    'melancholic': {'valence': -0.6, 'arousal': -0.5},
    'energetic': {'valence': 0.3, 'arousal': 0.8},
    'peaceful': {'valence': 0.6, 'arousal': -0.7},
    'anxious': {'valence': -0.4, 'arousal': 0.6},
    'upbeat': {'valence': 0.7, 'arousal': 0.7},
    'chill': {'valence': 0.4, 'arousal': -0.5},
    
    # Contextual modifiers
    'morning': {'valence': 0.3, 'arousal': 0.4},
    'night': {'valence': 0.0, 'arousal': -0.3},
    'workout': {'valence': 0.5, 'arousal': 0.9},
    'study': {'valence': 0.0, 'arousal': 0.3},
    'party': {'valence': 0.6, 'arousal': 0.8},
    'sleep': {'valence': 0.2, 'arousal': -0.8},
}
```

**Multi-Keyword Aggregation**
```python
def extract_mood_from_query(query_text):
    """
    Parse query and extract mood vector
    """
    tokens = query_text.lower().split()
    matched_moods = []
    
    for token in tokens:
        if token in MOOD_LEXICON:
            matched_moods.append(MOOD_LEXICON[token])
    
    if not matched_moods:
        # Default: neutral mood
        return {'valence': 0.0, 'arousal': 0.0}
    
    # Average all matched moods
    avg_valence = sum(m['valence'] for m in matched_moods) / len(matched_moods)
    avg_arousal = sum(m['arousal'] for m in matched_moods) / len(matched_moods)
    
    return {'valence': avg_valence, 'arousal': avg_arousal}
```

**Example Queries**
```
"i miss my ex" 
→ tokens: ['miss', 'ex']
→ matched: {'miss': {valence: -0.6, arousal: -0.2}}
→ output: {valence: -0.6, arousal: -0.2}

"upbeat morning workout"
→ tokens: ['upbeat', 'morning', 'workout']
→ matched: [
    {'valence': 0.7, 'arousal': 0.7},
    {'valence': 0.3, 'arousal': 0.4},
    {'valence': 0.5, 'arousal': 0.9}
  ]
→ output: {valence: 0.5, arousal: 0.67}

"chill evening music"
→ tokens: ['chill', 'evening']
→ matched: [{'valence': 0.4, 'arousal': -0.5}]
→ output: {valence: 0.4, arousal: -0.5}
```

#### 7.2 Phrase-Level Mood Understanding

**Use Sentence Embeddings + Mood Classifier**
```python
def advanced_mood_extraction(query_text):
    """
    Use pre-trained sentence embedding + mood classifier
    """
    # Encode query with SBERT
    query_embedding = sbert_model.encode(query_text)
    
    # Train a simple regression model:
    # Input: SBERT embedding of text
    # Output: [valence, arousal]
    # Training data: Manually labeled 1000-2000 music-related queries
    
    mood_vector = mood_classifier.predict(query_embedding)
    return {'valence': mood_vector[0], 'arousal': mood_vector[1]}
```

**Training Data Collection**
- Manually label 1000-2000 queries with mood tags
- Use crowdsourcing (Mechanical Turk) or internal annotation
- Queries: "songs for breakup", "pump me up", "focus music", etc.
- Labels: Valence [-1, 1] and Arousal [-1, 1]

**Model**: Ridge Regression or XGBoost
- Input: 384-dim SBERT embedding
- Output: 2-dim valence-arousal vector
- Training time: <1 minute
- Inference time: <1ms per query

---

### Component 8: API Implementation & Optimization

**Objective**: Serve recommendations in <500ms on consumer hardware.

#### 8.1 System Requirements

**Hardware Profile**
- **CPU**: Ryzen 5 9000X (6-8 cores, ~4.5 GHz boost)
- **RAM**: 32GB (sufficient for in-memory indices)
- **GPU**: RX7800XT (optional, can accelerate FAISS if needed)
- **Storage**: SSD (fast loading of indices and metadata)

**Software Stack**
- **Language**: Python 3.9+
- **Web Framework**: FastAPI (asynchronous, high performance)
- **Vector Search**: FAISS (CPU or GPU)
- **Caching**: Redis (for URL status, popular queries)
- **Storage**: Parquet files for metadata, HDF5 for features

#### 8.2 API Endpoint Design

**Endpoint**: `POST /recommend`

**Request**
```json
{
  "query": "i miss my ex",
  "top_k": 50,
  "filters": {
    "min_year": 2010,
    "max_year": 2023,
    "genres": ["Alternative", "Indie", "Electronic"],
    "exclude_artists": ["Artist A", "Artist B"]
  }
}
```

**Response**
```json
{
  "query": "i miss my ex",
  "mood_detected": {"valence": -0.6, "arousal": -0.2},
  "results": [
    {
      "track_id": "123456",
      "title": "Someone Like You",
      "artist": "Adele",
      "year": 2011,
      "genre": "Pop",
      "permalink_url": "https://soundcloud.com/...",
      "score": 0.89,
      "mood": {"valence": -0.7, "arousal": -0.3},
      "audio_features": {
        "bpm": 67,
        "energy": 0.4,
        "valence_score": -0.7
      }
    },
    ...
  ],
  "metadata": {
    "total_candidates": 987,
    "processing_time_ms": 423,
    "stages": {
      "query_understanding": 2,
      "retrieval": 45,
      "scoring": 18,
      "reranking": 35,
      "postprocessing": 10
    }
  }
}
```

#### 8.3 Optimization Strategies

**1. Index Pre-Loading (Startup)**
```python
# Load all indices into memory at startup
text_index = faiss.read_index("indices/text_index.faiss")
audio_index = faiss.read_index("indices/audio_index.faiss")
mood_index = faiss.read_index("indices/mood_index.faiss")

# Load metadata into memory (Pandas DataFrame)
metadata_df = pd.read_parquet("metadata/tracks.parquet")

# Load audio features (HDF5 or dict)
with h5py.File("features/audio_features.h5", "r") as f:
    audio_features = {track_id: f[track_id][:] for track_id in f.keys()}
```

**2. Query Result Caching**
```python
# Cache popular queries in Redis
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def get_recommendations(query):
    # Check cache first
    cache_key = f"rec:{hash(query)}"
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # Compute recommendations
    results = full_recommendation_pipeline(query)
    
    # Cache for 1 hour
    redis_client.setex(cache_key, 3600, json.dumps(results))
    return results
```

**3. Asynchronous Processing**
```python
from fastapi import FastAPI
import asyncio

app = FastAPI()

@app.post("/recommend")
async def recommend(request: RecommendRequest):
    # Run heavy computation in thread pool
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None, 
        compute_recommendations, 
        request.query
    )
    return results
```

**4. Batch Query Processing** (For Multiple Queries)
```python
@app.post("/recommend/batch")
async def recommend_batch(requests: List[RecommendRequest]):
    # Process multiple queries in parallel
    results = await asyncio.gather(*[
        compute_recommendations_async(req.query) 
        for req in requests
    ])
    return results
```

**5. GPU Acceleration** (Optional)
```python
# Use FAISS GPU if available
import faiss

if faiss.get_num_gpus() > 0:
    # Transfer index to GPU
    gpu_resource = faiss.StandardGpuResources()
    text_index = faiss.index_cpu_to_gpu(gpu_resource, 0, text_index)
    # Speed up: 2-5x faster search
```

#### 8.4 Performance Profiling

**Expected Breakdown** (500ms Target)
| Stage | Time (ms) | Cumulative |
|-------|-----------|------------|
| Query Understanding | 2-5 | 5 |
| Parallel Index Retrieval | 40-50 | 55 |
| Score Computation | 15-20 | 75 |
| Re-ranking (MMR) | 30-40 | 115 |
| Dead Link Filtering | 1-2 | 117 |
| Metadata Hydration | 5-10 | 127 |
| JSON Serialization | 3-5 | 132 |
| **Total** | **130-140ms** | **✓ Under 500ms** |

**Buffer for Worst Case**: ~350ms safety margin
- Handles cache misses
- Large candidate sets
- Complex queries

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)

**Week 1: Data Cleaning & Healing**
- [ ] Task 1.1: Dead link detection script (test 10K URLs, estimate survival rate)
- [ ] Task 1.2: Music vs non-music filtering (implement heuristics, validate with sample)
- [ ] Task 1.3: Null value imputation (genre, tags, description)
- [ ] Task 1.4: Create cleaned dataset with new columns: `url_status`, `metadata_richness`, `is_music`
- [ ] **Deliverable**: `cleaned_dataset.parquet` with 100-200K valid music tracks

**Week 2: Audio Feature Extraction**
- [ ] Task 2.1: Set up Librosa environment and test on 100 sample tracks
- [ ] Task 2.2: Implement parallel audio processing pipeline (multiprocessing)
- [ ] Task 2.3: Extract features from all tracks with valid URLs (~30-50K tracks)
- [ ] Task 2.4: Audio feature imputation for tracks without audio
- [ ] Task 2.5: Validate imputation quality (MAE, cosine similarity)
- [ ] **Deliverable**: `audio_features.h5` file with 50-80D feature vectors

### Phase 2: Mood & Embeddings (Weeks 3-4)

**Week 3: Mood Space Construction**
- [ ] Task 3.1: Implement mood score calculation from audio features
- [ ] Task 3.2: Manually label 500-1000 tracks with mood tags for calibration
- [ ] Task 3.3: Train mood regression model (Ridge/XGBoost)
- [ ] Task 3.4: Generate mood vectors for all tracks
- [ ] Task 3.5: Visualize mood space (scatter plot, verify distribution)
- [ ] **Deliverable**: `mood_vectors.npy` (2D valence-arousal for all tracks)

**Week 4: Multi-Index Creation**
- [ ] Task 4.1: Generate SBERT embeddings for all tracks (title+description+tags)
- [ ] Task 4.2: Apply PCA to audio features (80D → 64D)
- [ ] Task 4.3: Build FAISS indices:
  - Text index (IVF-PQ, 384D)
  - Audio index (HNSW, 64D)
  - Mood index (Flat, 2D)
- [ ] Task 4.4: Test index search performance (measure latency)
- [ ] **Deliverable**: Three FAISS index files + performance report

### Phase 3: Retrieval & Scoring (Weeks 5-6)

**Week 5: Multi-Stage Retrieval**
- [ ] Task 5.1: Implement parallel index querying (threading)
- [ ] Task 5.2: Candidate aggregation and diversity filtering
- [ ] Task 5.3: Test end-to-end retrieval with sample queries
- [ ] **Deliverable**: Retrieval module with <50ms latency

**Week 6: Hybrid Scoring System**
- [ ] Task 6.1: Implement score computation functions (text, audio, mood, collab)
- [ ] Task 6.2: Build query type detection classifier
- [ ] Task 6.3: Implement adaptive weighting system
- [ ] Task 6.4: Manually evaluate 50 queries (relevance check)
- [ ] **Deliverable**: Scoring module with configurable weights

### Phase 4: Re-Ranking & API (Weeks 7-8)

**Week 7: Re-Ranking & Diversification**
- [ ] Task 7.1: Implement MMR algorithm
- [ ] Task 7.2: Artist spacing logic
- [ ] Task 7.3: Dead link filtering with cache
- [ ] Task 7.4: End-to-end testing with 100 diverse queries
- [ ] **Deliverable**: Complete recommendation pipeline

**Week 8: API Development**
- [ ] Task 8.1: FastAPI endpoint implementation
- [ ] Task 8.2: Redis caching setup
- [ ] Task 8.3: Performance profiling and optimization
- [ ] Task 8.4: API documentation (Swagger/OpenAPI)
- [ ] Task 8.5: Load testing (100 concurrent queries)
- [ ] **Deliverable**: Production-ready API with <500ms response time

### Phase 5: Evaluation & Iteration (Week 9-10)

**Week 9: Evaluation**
- [ ] Task 9.1: Create evaluation dataset (500 queries with relevance labels)
- [ ] Task 9.2: Compute metrics:
  - Precision@K
  - Recall@K
  - NDCG (Normalized Discounted Cumulative Gain)
  - Diversity Score (intra-list diversity)
- [ ] Task 9.3: A/B test: Baseline (SBERT only) vs Full System
- [ ] **Deliverable**: Evaluation report with metrics

**Week 10: Refinement**
- [ ] Task 10.1: Tune hyperparameters (weights, MMR lambda, FAISS parameters)
- [ ] Task 10.2: Address edge cases (rare genres, old tracks, instrumental)
- [ ] Task 10.3: Documentation (README, architecture diagrams, API guide)
- [ ] **Deliverable**: Final system ready for deployment

---

## Evaluation Metrics

### 1. Retrieval Quality

**Precision@K**: Fraction of top-K results that are relevant
```
Precision@10 = (Number of relevant tracks in top 10) / 10
```

**Recall@K**: Fraction of all relevant tracks that appear in top-K
```
Recall@50 = (Number of relevant tracks in top 50) / (Total relevant tracks)
```

**NDCG@K**: Normalized Discounted Cumulative Gain (considers position)
```
NDCG@K = DCG@K / IDCG@K
DCG@K = sum(relevance[i] / log2(i+1)) for i=1 to K
```

**Target Benchmarks** (Compared to SBERT-only baseline)
- Precision@10: >0.7 (70% relevant)
- Recall@50: >0.6 (60% coverage of relevant tracks)
- NDCG@20: >0.65

### 2. Diversity Metrics

**Intra-List Diversity (ILD)**
```
ILD = (1 / (K * (K-1))) * sum(distance(item_i, item_j)) for all pairs
```
- Measures average dissimilarity between recommended tracks
- Higher = more diverse

**Artist Diversity**
```
Artist_Diversity = (Unique artists in top 50) / 50
```
- Target: >0.7 (at least 35 different artists in top 50)

### 3. Mood Accuracy

**Mood Alignment Score**
```
Mood_Alignment = 1 - avg(euclidean_distance(query_mood, result_mood))
```
- Measures how well results match query's emotional intent
- Target: >0.75 for mood-heavy queries

### 4. Performance Metrics

**Latency Percentiles**
- p50: <150ms
- p95: <400ms
- p99: <500ms

**Throughput**
- Queries per second: >10 on single CPU core
- With caching: >50 QPS

---

## Advanced Extensions (Future Work)

### Extension 1: Collaborative Filtering with User Feedback

**Idea**: Track which recommendations users click/play and use for personalization

**Implementation**
```
User Interaction Schema:
- user_id
- track_id
- interaction_type: ['play', 'skip', 'like', 'add_to_playlist']
- timestamp

Build user-item interaction matrix (sparse)
Apply Matrix Factorization (ALS or SVD):
  - User embeddings (latent factors)
  - Track embeddings (latent factors)
  
Integrate collaborative scores into hybrid scoring
```

**Complexity**: Adds personalization without changing core architecture

### Extension 2: Contextual Bandits for Weight Optimization

**Idea**: Learn optimal weight combinations for hybrid scoring based on query context

**Implementation**
```
Treat weight selection as multi-armed bandit problem:
  - Context: Query type, user history, time of day
  - Actions: Different weight configurations
  - Reward: Click-through rate, dwell time
  
Use Thompson Sampling or LinUCB to explore/exploit
Automatically tune weights over time
```

### Extension 3: Deep Neural Re-Ranking

**Idea**: Replace hand-crafted scoring with learned ranking function

**Implementation**
```
Train a LambdaRank or RankNet model:
  - Input: [query_embedding, track_embedding, audio_features, mood_vector]
  - Output: Relevance score
  
Use Stage 2 candidates as training data with implicit feedback
Replace Stage 3 scoring with neural model
```

**Constraint**: Keep inference <20ms (lightweight model, 2-3 layers)

### Extension 4: Temporal Dynamics & Trends

**Idea**: Adjust recommendations based on time-of-day, season, trends

**Implementation**
```
Add temporal features:
  - Hour of day → morning boost for upbeat, night boost for chill
  - Day of week → Friday boost for party music
  - Season → summer boost for upbeat, winter for melancholic
  
Track trending tracks (playback count velocity)
Boost recent trending tracks by 10-20%
```

---

## Risk Mitigation

### Risk 1: High Percentage of Dead Links
**Impact**: Reduces dataset to 30-50K usable tracks
**Mitigation**: 
- Imputation strategies ensure all tracks have usable features
- Focus on metadata-rich tracks (prioritize tracks with descriptions)
- Consider scraping fresh SoundCloud data or using alternate sources (YouTube Audio Library, Free Music Archive)

### Risk 2: Poor Audio Feature Imputation Quality
**Impact**: Inaccurate mood mapping for tracks without audio
**Mitigation**:
- Validate imputation on held-out test set (10% of tracks with audio)
- Use ensemble imputation (genre-based + k-NN + regression)
- Fall back to text-only recommendations for low-confidence tracks

### Risk 3: API Latency Exceeds 500ms
**Impact**: Poor user experience
**Mitigation**:
- Implement aggressive caching (cache popular queries for 1 hour)
- Pre-compute recommendations for common mood keywords
- Use approximate FAISS indices with lower precision if needed
- Profile and optimize bottlenecks (use cProfile, line_profiler)

### Risk 4: Low Mood Accuracy for Ambiguous Queries
**Impact**: Irrelevant recommendations for vague queries like "good music"
**Mitigation**:
- Graceful degradation: fall back to popularity-based ranking
- Prompt user for clarification ("What mood are you in?")
- Use click feedback to refine understanding over time

### Risk 5: Insufficient Training Data for Mood Classifier
**Impact**: Poor query understanding for mood-heavy queries
**Mitigation**:
- Start with lexicon-based approach (no training required)
- Collect user feedback to build training dataset incrementally
- Use data augmentation (paraphrase queries with GPT, label offline)

---

## Comparison to Spotify

**What This System Has**
- ✅ Mood-based retrieval (valence-arousal space)
- ✅ Hybrid filtering (text + audio + collaborative signals)
- ✅ Multi-stage retrieval and re-ranking
- ✅ Diversity optimization (MMR)
- ✅ Fast inference (<500ms)

**What Spotify Has (That This Lacks)**
- ❌ Massive user interaction data (billions of plays, skips, playlist adds)
- ❌ Deep learning models trained on user behavior (RNNs, Transformers)
- ❌ Personalized embeddings per user
- ❌ Real-time collaborative filtering at scale
- ❌ A/B testing infrastructure with millions of users
- ❌ High-quality licensed audio for all tracks

**Closing the Gap**
- Dataset quality is the biggest constraint (SoundCloud data is noisy)
- With clean audio and richer metadata, this system would approach Spotify-level quality for mood-based queries
- The architecture is sound; performance depends on data quality

---

## Key Files & Structure

```
project/
│
├── data/
│   ├── raw/
│   │   └── soundcloud_crawl.csv          # Original dataset
│   ├── processed/
│   │   ├── cleaned_dataset.parquet       # After healing & imputation
│   │   ├── audio_features.h5             # Audio feature vectors
│   │   ├── mood_vectors.npy              # Valence-arousal coordinates
│   │   ├── text_embeddings.npy           # SBERT embeddings
│   │   └── url_status_cache.db           # SQLite cache for URL status
│   └── indices/
│       ├── text_index.faiss              # FAISS text index
│       ├── audio_index.faiss             # FAISS audio index
│       └── mood_index.faiss              # FAISS mood index
│
├── src/
│   ├── preprocessing/
│   │   ├── dead_link_detector.py         # URL validation
│   │   ├── music_filter.py               # Music vs non-music classifier
│   │   ├── imputation.py                 # Null value imputation
│   │   └── feature_extraction.py         # Audio feature extraction (Librosa)
│   │
│   ├── embeddings/
│   │   ├── text_embedder.py              # SBERT encoding
│   │   ├── audio_embedder.py             # Audio feature engineering
│   │   └── mood_calculator.py            # Valence-arousal mapping
│   │
│   ├── indexing/
│   │   ├── build_indices.py              # FAISS index construction
│   │   └── index_config.py               # FAISS hyperparameters
│   │
│   ├── retrieval/
│   │   ├── multi_index_search.py         # Parallel querying
│   │   ├── candidate_aggregator.py       # Merging results
│   │   └── diversity_filter.py           # Initial diversity pruning
│   │
│   ├── scoring/
│   │   ├── hybrid_scorer.py              # Multi-signal scoring
│   │   ├── query_classifier.py           # Query type detection
│   │   └── weight_config.py              # Adaptive weight matrix
│   │
│   ├── reranking/
│   │   ├── mmr.py                        # Maximal Marginal Relevance
│   │   ├── artist_spacing.py             # Artist diversity enforcement
│   │   └── dead_link_filter.py           # Real-time URL check
│   │
│   ├── api/
│   │   ├── main.py                       # FastAPI app
│   │   ├── models.py                     # Request/response schemas
│   │   ├── cache.py                      # Redis caching layer
│   │   └── config.py                     # API configuration
│   │
│   └── utils/
│       ├── mood_lexicon.py               # Mood keyword mapping
│       ├── logger.py                     # Logging utilities
│       └── metrics.py                    # Evaluation metrics
│
├── notebooks/
│   ├── 01_data_exploration.ipynb         # EDA on raw data
│   ├── 02_audio_feature_analysis.ipynb   # Analyze extracted features
│   ├── 03_mood_space_visualization.ipynb # Visualize valence-arousal
│   └── 04_evaluation.ipynb               # Compute metrics
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_retrieval.py
│   ├── test_scoring.py
│   └── test_api.py
│
├── scripts/
│   ├── run_preprocessing.sh              # Execute all preprocessing
│   ├── build_indices.sh                  # Build FAISS indices
│   └── start_api.sh                      # Launch API server
│
├── requirements.txt                      # Python dependencies
├── README.md                             # This file
└── ARCHITECTURE.md                       # Detailed architecture diagrams
```

---

## Dependency List

```txt
# Core ML/NLP
numpy>=1.23.0
pandas>=1.5.0
scikit-learn>=1.2.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4  # or faiss-gpu for GPU acceleration

# Audio Processing
librosa>=0.10.0
soundfile>=0.12.0

# Data Storage
h5py>=3.8.0
pyarrow>=11.0.0  # for Parquet

# API
fastapi>=0.100.0
uvicorn>=0.23.0
redis>=4.5.0
pydantic>=2.0.0

# Utilities
tqdm>=4.65.0
requests>=2.31.0
python-dotenv>=1.0.0

# Optional (for GPU acceleration)
# faiss-gpu>=1.7.4
# torch>=2.0.0  # if using neural re-ranking

# Development
pytest>=7.3.0
jupyter>=1.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## Success Criteria

**Minimum Viable Product (MVP)**
- [ ] API responds in <500ms for 95% of queries
- [ ] Recommendations are relevant for at least 70% of mood-based queries
- [ ] System runs on consumer hardware (Ryzen 5, 32GB RAM)
- [ ] Handles dataset with 70-80% null values gracefully

**Stretch Goals**
- [ ] Precision@10 > 0.75
- [ ] NDCG@20 > 0.70
- [ ] Artist diversity > 0.75 in top 50
- [ ] Query throughput > 20 QPS with caching

---

## Conclusion

This architecture adds **substantive complexity** without relying on LLM APIs:

1. **Multi-stage retrieval** (3 parallel indices) vs single FAISS index
2. **Hybrid scoring** (5 signal types) vs text similarity only
3. **Mood space** (valence-arousal mapping) vs keyword matching
4. **Audio feature extraction** (50-80 dimensions from signal processing)
5. **Data healing** (imputation, dead link recovery, music filtering)
6. **Re-ranking** (MMR, diversity, artist spacing)

**Key Innovation**: Query understanding through mood space enables semantic matching ("i miss my ex" → valence=-0.6, arousal=-0.2) without keyword dependence.

**Performance**: Achieves <500ms latency through parallelization, caching, and optimized FAISS indices.

**Scalability**: Architecture supports 100K-1M tracks on consumer hardware; can scale to 10M+ with GPU acceleration.

This is a **production-grade system** that rivals commercial music recommenders despite dataset constraints.

---

## Questions & Clarifications

If you need clarification on any component or want to prioritize certain features, let me know:

1. **Audio Extraction**: Should we prioritize breadth (extract from all valid URLs) or depth (extract more features)?
2. **Mood Calibration**: Do you want to manually label mood data, or use unsupervised clustering?
3. **Collaborative Filtering**: Should we plan for user feedback collection from day 1, or add later?
4. **API Features**: Do you want filtering (by year, genre) or just raw recommendations?

---

**Next Steps**: Start with Phase 1 (data cleaning & healing) to establish the foundation. Once the dataset is clean, everything else builds smoothly on top.