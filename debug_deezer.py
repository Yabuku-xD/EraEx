import requests
from langdetect import detect

def debug_deezer(query_term, target_artist):
    print(f"Searching Deezer for: {query_term} (Target Artist: {target_artist})")
    url = "https://api.deezer.com/search"
    params = {'q': query_term, 'limit': 100}
    r = requests.get(url, params=params)
    data = r.json().get('data', [])
    print(f"Raw Results: {len(data)}")

    filtered_count = 0
    for i, track in enumerate(data):
        artist_name = track['artist']['name'].lower()
        title = track['title']
        
        # 1. Artist Check
        if target_artist.lower() not in artist_name:
            # print(f"  [Skip] Artist mismatch: {artist_name}")
            continue

        # 2. Date Check
        r_date = track.get('release_date')
        if not r_date and 'album' in track:
            try:
                alb_id = track['album']['id']
                alb_res = requests.get(f"https://api.deezer.com/album/{alb_id}").json()
                r_date = alb_res.get('release_date')
            except:
                pass
        
        print(f"  Track: {title} | Artist: {track['artist']['name']} | Date: {r_date}", end=" ")

        if not r_date:
             print("-> [FAIL] No Date")
             continue
             
        year = int(r_date[:4])
        if 2012 <= year <= 2018:
            print("-> [PASS]")
            filtered_count += 1
        else:
            print(f"-> [FAIL] Year {year}")
            
    print(f"Total Passing Filter: {filtered_count}")

debug_deezer("PARTYNEXTDOOR", "PARTYNEXTDOOR")
