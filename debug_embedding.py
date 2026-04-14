import json
from pathlib import Path

# Load one embedding from the file to inspect its format
with open('output/embeddings/codedecommerce_embedded.json', 'r', encoding='utf-8') as f:
    articles = json.load(f)

# Check first article
first = articles[0]
print(f'First article keys: {first.keys()}')
print(f'Embedding type: {type(first["embedding"])}')
if isinstance(first['embedding'], list):
    print(f'Embedding length: {len(first["embedding"])}')
    if len(first['embedding']) > 0:
        print(f'First element type: {type(first["embedding"][0])}')
        print(f'First element: {first["embedding"][0]}')
        if len(first['embedding']) > 1:
            print(f'Second element type: {type(first["embedding"][1])}')
            if isinstance(first['embedding'][1], list):
                print(f'Second element length: {len(first["embedding"][1])}')

# Find article 18
for i, article in enumerate(articles):
    if article.get('article') == 18:
        print(f'\nFound Article 18 at index {i}!')
        print(f'Content: {article.get("content", "")[:100]}')
        if 'embedding' in article:
            emb = article['embedding']
            print(f'Embedding structure: {type(emb)} length={len(emb) if isinstance(emb, (list, tuple)) else "N/A"}')
        break
else:
    print('\nArticle 18 not found')
