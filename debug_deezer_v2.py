import requests
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

def debug_deezer_v2(query_term):
    print(f"Searching Deezer for: {query_term}")
    url = "https://api.deezer.com/search"
    params = {'q': query_term, 'limit': 50}
    r = requests.get(url, params=params)
    data = r.json().get('data', [])
    
    print(f"Analyzing {len(data)} tracks...")
    
    for track in data:
        full_text = f"{track['artist']['name']} {track['title']}"
        try:
            lang = detect(full_text)
        except:
            lang = "err"
            
        print(f"Text: '{full_text}' -> Lang: {lang}")

debug_deezer_v2("PARTYNEXTDOOR")
