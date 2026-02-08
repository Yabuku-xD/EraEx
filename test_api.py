import requests

print("Sending search request...")
try:
    resp = requests.post(
        "http://localhost:8000/search",
        json={"query": "podcast hiv", "year_start": 2012, "year_end": 2018, "top_k": 20},
        timeout=10
    )
    if resp.status_code == 200:
        data = resp.json()
        print(f"Success! Got {len(data['results'])} results.")
        for t in data['results']:
            print(f"- {t.get('title')}")
    else:
        print(f"Error: {resp.status_code} - {resp.text}")
except Exception as e:
    print(f"Request failed: {e}")
