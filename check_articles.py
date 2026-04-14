import json

with open('output/embeddings/codedecommerce_embedded.json', 'r', encoding='utf-8') as f:
    articles = json.load(f)

# Find articles 488, 489, 499
for a in articles:
    if a.get('article_number') in ['488', '489', '499']:
        print(f"Article {a.get('article_number')}:")
        print(f"  content length: {len(a.get('content', ''))}")
        content_preview = a.get('content', '')[:80] if a.get('content') else 'EMPTY'
        print(f"  content preview: {content_preview}")
        print(f"  pages: {a.get('pages', 'MISSING')}")
        print(f"  page_start: {a.get('page_start', 'MISSING')}")
        print(f"  page_end: {a.get('page_end', 'MISSING')}")
        print()
