import pickle

with open('data/indices/sonic_merged.pkl', 'rb') as f:
    data = pickle.load(f)

matches = [t for t in data if 'partynextdoor' in t['artist'].lower()]
print(f'PARTYNEXTDOOR songs in index: {len(matches)}')
for t in matches:
    print(f"  - {t['title']} ({t.get('year', 'N/A')})")
