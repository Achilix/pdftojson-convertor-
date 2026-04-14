import json
from pathlib import Path

# Load one embedding from the file to inspect its format
with open('output/embeddings/codedecommerce_embedded.json', 'r', encoding='utf-8') as f:
    articles = json.load(f)

# Check first article in detail
first = articles[0]
print(f'First article keys: {list(first.keys())}')
print(f'Article number field: {first.get("article_number", "N/A")}')
print(f'Article field: {first.get("article", "N/A")}')

# Check embedding structure
emb = first['embedding']
print(f'\nEmbedding structure:')
print(f'  Type: {type(emb)}, Length: {len(emb)}')
if isinstance(emb, list) and len(emb) > 0:
    print(f'  [0] type: {type(emb[0])}, length: {len(emb[0]) if isinstance(emb[0], (list, tuple)) else "N/A"}')
    if isinstance(emb[0], list) and len(emb[0]) > 0:
        print(f'      [0][0]: {emb[0][0]}')
        if len(emb[0]) > 1 and isinstance(emb[0][1], list):
            print(f'      [0][1] is list with {len(emb[0][1])} floats')
    if len(emb) > 1:
        print(f'  [1] type: {type(emb[1])}, length: {len(emb[1]) if isinstance(emb[1], (list, tuple)) else "N/A"}')

# Find article 18
print(f'\n\nSearching for article_number 18...')
for i, article in enumerate(articles):
    if article.get('article_number') == 18:
        print(f'Found Article 18 at index {i}!')
        print(f'Content preview: {article.get("content", "")[:100]}')
        if 'obligation' in article.get("content", "").lower():
            print('MATCH: Content contains "obligation"!')
        break
else:
    # Check what article numbers exist
    article_nums = set(a.get('article_number') for a in articles[:20])
    print(f'First 20 articles have article_number values: {sorted(article_nums)}')
