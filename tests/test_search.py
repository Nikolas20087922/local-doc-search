import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from search import build_index, chunks, rank, tokens

ROOT = Path(__file__).resolve().parents[1]


class RetrievalTests(unittest.TestCase):
    def test_relevant_source_ranks_first(self):
        result = rank(build_index(ROOT / 'examples'), 'cosine TF-IDF')
        self.assertEqual(result[0]['source'], 'retrieval.md')
        self.assertIn('Cosine', result[0]['snippet'])

    def test_unknown_and_empty_queries_return_nothing(self):
        index = build_index(ROOT / 'examples')
        self.assertEqual(rank(index, 'quokka'), [])
        self.assertEqual(rank(index, ''), [])

    def test_chunk_boundaries(self):
        self.assertEqual(list(chunks('a b c d e f', 4, 1)), [(0, 'a b c d'), (3, 'd e f')])
        with self.assertRaises(ValueError):
            list(chunks('x', 2, 2))

    def test_unicode_casefold(self):
        self.assertEqual(tokens('ŻÓŁĆ, Python_3!'), ['żółć', 'python', '3'])

    def test_hidden_files_and_symlinks_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'visible.txt').write_text('alpha beta')
            (root / '.hidden.md').write_text('private')
            (root / 'link.txt').symlink_to(root / 'visible.txt')
            index = build_index(root)
            self.assertEqual([d['source'] for d in index['documents']], ['visible.txt'])

    def test_stable_ties_and_empty_index(self):
        docs = [{'source': p, 'word_offset': 0, 'text': 'alpha', 'counts': {'alpha': 1}} for p in ['b.txt', 'a.txt']]
        self.assertEqual(rank({'version': 1, 'documents': docs}, 'alpha')[0]['source'], 'a.txt')
        self.assertEqual(rank({'version': 1, 'documents': []}, 'alpha'), [])

    def test_cli_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            index = Path(directory) / 'index.json'
            first = subprocess.run([sys.executable, str(ROOT / 'search.py'), 'index', str(ROOT / 'examples'), '--output', str(index)], capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            second = subprocess.run([sys.executable, str(ROOT / 'search.py'), 'query', str(index), 'regression testing'], capture_output=True, text=True)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(json.loads(second.stdout)[0]['source'], 'testing.md')


if __name__ == '__main__':
    unittest.main()
