from flask import Flask, jsonify, request
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.search.candidates import CandidateGenerator
from src.ranking.hybrid import HybridRanker
from src.ranking.nostalgia import NostalgiaFilter
from src.data.deezer import DeezerCollector
from flask import render_template
from langdetect import detect

app = Flask(__name__, 
            template_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), '../templates')),
            static_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), '../static')))

# Mock Global Objects for now - in production, load these
# generator = CandidateGenerator('models/als_model.pkl', 'models/items.index')
ranker = HybridRanker()
nostalgia = NostalgiaFilter()
deezer_collector = DeezerCollector() # Use our existing collector for searching

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('q', '')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
        
    # 0. CHECK FOR COMPLEX QUERY (SONIC ROUTE)
    # If query is long (>2 words) or contains mood words, use Sonic Engine (GLM + SBERT)
    is_complex = len(query.split()) > 2
    if is_complex and sonic_engine:
        print(f"Routing to Sonic Engine: {query}")
        try:
            # sonic_engine.search_by_text uses GLM -> Enhancer -> SBERT -> Vector Search
            results = sonic_engine.search_by_text(query, limit=30)
            if results:
                return jsonify({
                    'query': query,
                    'results': results,
                    'source': 'sonic'
                })
        except Exception as e:
            print(f"Sonic Search failed, falling back to Deezer: {e}")

    print(f"User Query: {query}")
    
    # 1. Parse Intent (Mood vs Artist)
    intent = nostalgia.get_search_query(query)
    search_query = intent['query']
    search_type = intent['type']
    
    print(f"Parsed Intent: {search_type} -> Deezer Query: {search_query}")
    

    # 2. Search Deezer
    import requests
    url = "https://api.deezer.com/search"
    

    # Helper to perform search and filter
    def perform_search_and_filter(query_term, query_type, limit=60):
        params = {'q': query_term, 'limit': limit}
        try:
            r = requests.get(url, params=params)
            r_data = r.json()
            raw = r_data.get('data', [])
        except:
            return []
            
        filtered = []
        from langdetect import detect
        import time
        
        for track in raw:
            # STRICT ARTIST FILTERING
            if query_type == 'artist':
                artist_name = track['artist']['name'].lower()
                target_artist = intent['original'].lower()
                if target_artist not in artist_name:
                    continue

            # DATE & LANGUAGE FILTER (2012-2018)
            try:
                # 1. Check Language (Title + Artist)
                # SKIP language check if we are doing a Specific Artist Search (we trust the artist)
                # This fixes issues where 'PARTYNEXTDOOR' tracks are detected as German/Tagalog/etc.
                if query_type != 'artist':
                    full_text = f"{track['artist']['name']} {track['title']}"
                    try:
                        lang = detect(full_text)
                        # Relaxed Filter: Removed 'de' (German), 'it', 'tr', 'vi', etc. as they false-flag English often
                        # Only filtering major non-English markets that definitely aren't 2012-2018 Western Pop context
                        if lang in ['es', 'fr', 'pt', 'ja', 'ko', 'zh', 'ru', 'ar', 'hi']:
                             continue
                    except:
                        pass

                # 2. Release Date Check
                # Deezer Search API doesn't return release_date for tracks usually.
                # We MUST fetch the album to get the date.
                # To avoid rate limits, we only do this for the top N potential matches
                # But we can't sort yet? We have to filter first?
                # Actually, Deezer Search sorts by rank/relevance.
                # So we can try to fetch.
                r_date = track.get('release_date')
                if not r_date and 'album' in track:
                    try:
                        # Fetch Album Details
                        alb_id = track['album']['id']
                        # Simple cache could go here, but for now just fetch
                        import requests
                        alb_res = requests.get(f"https://api.deezer.com/album/{alb_id}", timeout=2).json()
                        r_date = alb_res.get('release_date')
                    except:
                        pass
                
                # If we still don't have a date, we SKIP it to be safe (Strict Mode)
                if not r_date:
                    continue

                # If we have a date, check era
                if not nostalgia.is_in_era(r_date):
                    continue
                    
                track['release_date'] = r_date
                filtered.append(track)
                
            except:
                 continue
            
        # Sort by popularity (rank) descending
        # Deezer 'rank' is an integer (higher is better)
        filtered.sort(key=lambda x: x.get('rank', 0), reverse=True)
        return filtered

    # 1. Primary Search
    # Increase limit to 150 to ensure we find enough matching English tracks from 2012-2018
    filtered_tracks = perform_search_and_filter(search_query, search_type, limit=150)

    # 1b. Sonic Fallback (Safety Net)
    # If Deezer returned too few results (e.g. mostly new songs filtered out, or rate limits),
    # we try the Sonic Engine which generally has good 2012-2018 coverage.
    if len(filtered_tracks) < 5 and sonic_engine:
        print(f"Low results ({len(filtered_tracks)}). Attempting Sonic Fallback for: {query}")
        try:
            sonic_results = sonic_engine.search_by_text(query, limit=20)
            # Create a set of existing IDs to avoid duplicates
            existing_ids = set(t['id'] for t in filtered_tracks)
            
            for r in sonic_results:
                if r['id'] not in existing_ids:
                    # Adapt Sonic result format to Deezer format if needed
                    # Sonic returns dict with 'title', 'artist', 'album', 'preview', 'id'
                    # which matches what we need.
                    # We give them a decent score/rank.
                    r['score'] = r.get('score', 0.8)
                    filtered_tracks.append(r)
                    existing_ids.add(r['id'])
        except Exception as e:
            print(f"Sonic Fallback failed: {e}")

    # 2. Fallback Logic
    # If NO tracks passed the filter (or 0 raw results), and it was a general search
    if not filtered_tracks and search_type == 'general':
         print("Primary search yielded 0 results. Triggering Smart Fallback...")
         
         # Extract genres
         genres = nostalgia.extract_genres(query)
         if genres:
             print(f"Fallback Genres: {genres}")
             for genre in genres:
                 # Search for genre
                 found = perform_search_and_filter(f'genre:"{genre}"', 'mood', limit=50)
                 filtered_tracks.extend(found)
                 if len(filtered_tracks) >= 20: break
         
         # Last resort: Try splitting words
         if not filtered_tracks:
            words = query.split()
            if len(words) > 1:
                # Try last word (often noun)
                fallback_term = words[-1]
                print(f"Fallback Last Resort: {fallback_term}")
                filtered_tracks.extend(perform_search_and_filter(fallback_term, 'general', limit=50))

    # 3. Final Response setup (just remove the old logic block)
    # Re-assign filtered_tracks to local var if needed or just use it in response
            
    return jsonify({
        'query': query,
        'intent': search_type,
        'mood': intent.get('mood'),
        'sentiment': intent.get('sentiment'),
        'results': filtered_tracks
    })

@app.route('/recommend/<user_id>', methods=['GET'])
def recommend(user_id):
    try:
        user_id = int(user_id)
        n = int(request.args.get('n', 20))
        
        # 1. Candidate Generation
        # candidates_tuples = generator.get_candidates(user_id, k=500)
        
        # MOCK candidates for testing without trained model
        candidates_tuples = [(123, 0.95), (456, 0.88), (789, 0.82)] 
        
        # 2. Enrich with metadata (In real app, fetch from DB/Parquet)
        # mock_metadata_db = { ... }
        candidates_enriched = []
        for tid, score in candidates_tuples:
            candidates_enriched.append({
                'track_id': tid,
                'cf_score': score,
                'metadata': {
                    'title': f'Track {tid}',
                    'artist': 'Artist X',
                    'genre': 'Pop',
                    'rank': 500000,
                    'release_date': '2015-01-01' # Updated to be in-era
                }
            })
            
        # 3. Ranking
        # user_prefs = generator.get_user_prefs(user_id) # Need logic to get user history
        user_prefs = {'Pop': 0.5, 'Rock': 0.2} # Mock
        
        ranked = ranker.score(user_id, candidates_enriched, user_prefs)
        
        # 4. Format & Filter (Final Guardrail)
        response = []
        for r in ranked: # Check all, not just slice yet
            # Strict Era Check
            r_date = r['metadata'].get('release_date')
            if not nostalgia.is_in_era(r_date):
                continue
                
            response.append({
                'id': r['track_id'],
                'title': r['metadata']['title'],
                'artist': r['metadata']['artist'],
                'score': r['final_score'],
                'year': r_date[:4] if r_date else 'Unknown'
            })
            if len(response) >= n: break
            
        return jsonify({
            'user_id': user_id,
            'recommendations': response
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 4. Sonic Search Integration
from src.search.sonic_search import SonicSearch
# Initialize Sonic Engine on startup
# Note: This might take a second to load the pickle
print("Initializing Sonic Engine...")
try:
    sonic_engine = SonicSearch()
except Exception as e:
    print(f"Failed to load Sonic Engine: {e}")
    sonic_engine = None

@app.route('/sonic', methods=['GET'])
def sonic_search():
    """
    Multimodal Search Endpoint.
    Expects 'q' (text query) or 'audio_url' (future).
    Currently uses SBERT on the text query to find semantic matches in the index.
    """
    if not sonic_engine:
        return jsonify({'error': 'Sonic Engine not loaded'}), 503
        
    query = request.args.get('q', '')
    limit = int(request.args.get('limit', 10))
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
        
    print(f"Sonic Query: {query}")
    
    # 1. Encode Text Query -> Vector
    # We need the SemanticEncoder to do this live.
    # Ideally, SonicSearch class handles this, but we need to pass the encoder or have it internal.
    # Let's check `search_by_vector`. It expects vectors.
    # We need to instantiate SemanticEncoder here or inside SonicSearch.
    # For efficiency, let's assume SonicSearch has a helper method or we add one now.
    
    # Check if we need to update SonicSearch to handle raw text queries?
    # Yes, let's update SonicSearch class first to include 'search_by_text'
    # But for now, let's try to do it here if possible, or assume we will update the class.
    # WAIT - We didn't import SemanticEncoder here.
    # Let's rely on a new method in SonicSearch: `search_by_text(query)`
    
    try:
        results = sonic_engine.search_by_text(query, limit=limit)
        return jsonify({
            'query': query,
            'results': results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/serendipity', methods=['GET'])
def serendipity():
    """
    Feeling Lucky Endpoint.
    Returns random but valid 2012-2018 tracks based on 'Dream Vectors'.
    """
    if not sonic_engine:
        return jsonify({'error': 'Sonic Engine not loaded'}), 503
        
    try:
        results = sonic_engine.serendipity_search(limit=5)
        return jsonify({
            'mode': 'serendipity',
            'results': results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)