import requests

def test(q):
    url = "https://api.deezer.com/search"
    resp = requests.get(url, params={'q': q, 'limit': 5})
    data = resp.json().get('data', [])
    print(f"Query: '{q}' -> Found: {len(data)}")
    from langdetect import detect
    for t in data:
         text = f"{t['artist']['name']} {t['title']}"
         try:
             lang = detect(text)
         except:
             lang = "err"
         print(f" - {text} [Lang: {lang}]")
        

    print(f"Query: '{q}' -> Found: {len(data)}")
    for t in data:
         print(f" - {t['title']} ({t['artist']['name']})")
         # Fetch Album Details
         import time
         time.sleep(0.3)
         try:
             alb = requests.get(f"https://api.deezer.com/album/{t['album']['id']}").json()
             if 'genres' in alb and alb['genres']['data']:
                 print(f"   -> Genres: {[g['name'] for g in alb['genres']['data']]}")
             else:
                 print("   -> No Genres found")
         except Exception as e:
             print(f"   -> Error fetching album: {e}")

test('genre:"trap"')
