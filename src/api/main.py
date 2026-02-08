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
        count = 0
        import time
        
        for track in raw:
            # We don't break early anymore so we can collect enough candidates to sort by popularity
            # if count >= 20: break
            
            # STRICT ARTIST FILTERING
            if query_type == 'artist':
                artist_name = track['artist']['name'].lower()
                target_artist = intent['original'].lower()
                if target_artist not in artist_name:
                    continue

            # DATE & LANGUAGE FILTER (2012-2018)
            try:
                # 1. Check Language (Title + Artist)
                # Using langdetect on the text
                full_text = f"{track['artist']['name']} {track['title']}"
                try:
                    lang = detect(full_text)
                    # Filter out common non-English languages
                    if lang in ['es', 'fr', 'pt', 'it', 'de', 'ja', 'ko', 'zh', 'ru', 'ar', 'hi', 'tr', 'vi', 'pl', 'nl']:
                        # Double check if it's really non-English or just short text error?
                        # If strict mode, we skip. But let's be safe.
                        # Using album genre as confirmation is safer, but let's filter explicit matches first.
                        # Exception: If user searched for a specific Spanish artist, we shouldn't filter?
                        # But current context is "stick with English".
                        # Let's trust langdetect for explicit matches.
                        continue
                except:
                    pass

                # 2. Fetch Album for Date & Genre
                # Check cached album info first if possible (not implemented here yet)
                r_date = track.get('release_date') 
                alb_genres = []
                
                if not r_date: 
                    alb_id = track['album']['id']
                    alb_resp = requests.get(f"https://api.deezer.com/album/{alb_id}").json()
                    r_date = alb_resp.get('release_date')
                    
                    if 'genres' in alb_resp and alb_resp['genres']['data']:
                        alb_genres = [g['name'] for g in alb_resp['genres']['data']]

                # 3. Genre Blacklist (Proxy for Language)
                # e.g. "Latin Music" is almost always Spanish
                # We do this check regardless of langdetect result to catch things missed by it
                non_english_genres = {
                    'Latin Music', 'Musica Mexicana', 'Traditional Mexicano', 'Reggaeton', 
                    'K-Pop', 'J-Pop', 'Asian Music', 'African Music', 'Brazilian Music', 
                    'French Chanson', 'Schlager', 'German Pop', 'Spanish Pop', 'Indian Music', 
                    'Arabic Music', 'Bollywood', 'C-Pop', 'Salsa', 'Bachata', 'Tango', 'Flamenco'
                }
                
                # Check similarity/intersection
                if any(g in non_english_genres for g in alb_genres):
                    continue

                if nostalgia.is_in_era(r_date):
                    track['release_date'] = r_date
                    filtered.append(track)
                    count += 1
            except:
                continue
            
            time.sleep(0.01) # Mild rate limit protection
            
        # Sort by popularity (rank) descending
        # Deezer 'rank' is an integer (higher is better)
        filtered.sort(key=lambda x: x.get('rank', 0), reverse=True)
        return filtered

    # 1. Primary Search
    # Increase limit to 150 to ensure we find enough matching English tracks from 2012-2018
    filtered_tracks = perform_search_and_filter(search_query, search_type, limit=150)

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