import requests
import re

urls = [
    'https://soundcloud.com/octobersveryown/partynextdoor-make-a-mil', # User says dead
    'https://soundcloud.com/octobersveryown/partynextdoor-wus-good-curious', # User says dead
    'https://soundcloud.com/partynextdoor/partynextdoor-right-now', # Live
    'https://soundcloud.com/partynextdoor/this-track-does-not-exist-12345-xyz' # FAKE
]

print("Testing OEmbed...")
for url in urls:
    print(f"\nChecking {url}...")
    try:
        oembed_url = f"https://soundcloud.com/oembed?url={url}&format=json"
        r = requests.get(oembed_url, timeout=10)
        print(f"Status: {r.status_code}")
        if r.status_code == 200:
            print("OEmbed suggests: ALIVE")
        else:
            print("OEmbed suggests: DEAD")
            
    except Exception as e:
        print(f"Error: {e}")