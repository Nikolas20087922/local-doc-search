"""Offline TF-IDF document retrieval with deterministic ranking and source snippets."""
import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

TOKEN = re.compile(r'[^\W_]+', re.UNICODE)


def tokens(text):
    return TOKEN.findall(text.casefold())


def chunks(text, size=120, overlap=20):
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError('Require chunk size > overlap >= 0')
    words = text.split()
    for start in range(0, len(words), size - overlap):
        yield start, ' '.join(words[start:start + size])
        if start + size >= len(words):
            break


def build_index(directory, size=120, overlap=20):
    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError('Document directory does not exist')
    # Validate configuration even when there are no documents.
    list(chunks('', size, overlap))
    documents = []
    for path in sorted(root.rglob('*')):
        if path.suffix.lower() not in {'.txt', '.md'} or not path.is_file() or path.is_symlink():
            continue
        if any(part.startswith('.') for part in path.relative_to(root).parts):
            continue
        if not path.resolve().is_relative_to(root):
            continue
        text = path.read_text(encoding='utf-8')
        for start, snippet in chunks(text, size, overlap):
            counts = Counter(tokens(snippet))
            if counts:
                documents.append({'source': path.relative_to(root).as_posix(), 'word_offset': start, 'text': snippet, 'counts': dict(counts)})
    return {'version': 1, 'documents': documents}


def rank(index, query, limit=5):
    if limit <= 0:
        raise ValueError('Result limit must be positive')
    if index.get('version') != 1 or not isinstance(index.get('documents'), list):
        raise ValueError('Unsupported index format')
    docs = index['documents']
    frequencies = Counter(term for doc in docs for term in doc['counts'])
    idf = {term: 1 + math.log((1 + len(docs)) / (1 + frequency)) for term, frequency in frequencies.items()}

    def vector(counts):
        return {term: (1 + math.log(count)) * idf[term] for term, count in counts.items() if term in idf}

    def norm(v):
        return math.sqrt(sum(x * x for x in v.values()))

    q = vector(Counter(tokens(query)))
    qnorm = norm(q)
    if not qnorm:
        return []
    results = []
    for doc in docs:
        v = vector(doc['counts'])
        denominator = qnorm * norm(v)
        score = sum(weight * v.get(term, 0) for term, weight in q.items()) / denominator if denominator else 0
        if score > 0:
            results.append({'source': doc['source'], 'word_offset': doc['word_offset'], 'score': round(score, 6), 'snippet': doc['text']})
    return sorted(results, key=lambda item: (-item['score'], item['source'], item['word_offset']))[:limit]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('index')
    build.add_argument('directory', type=Path)
    build.add_argument('--output', type=Path, required=True)
    build.add_argument('--chunk-size', type=int, default=120)
    build.add_argument('--overlap', type=int, default=20)
    query = sub.add_parser('query')
    query.add_argument('index', type=Path)
    query.add_argument('text')
    query.add_argument('--limit', type=int, default=5)
    args = parser.parse_args(argv)
    try:
        if args.command == 'index':
            result = build_index(args.directory, args.chunk_size, args.overlap)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            print(json.dumps({'chunks_indexed': len(result['documents'])}))
        else:
            result = rank(json.loads(args.index.read_text(encoding='utf-8')), args.text, args.limit)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f'error: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
