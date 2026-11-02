# Local Document Search

An offline search CLI for Markdown and text notes. It splits documents into
overlapping passages and ranks them with TF-IDF and cosine similarity. Results
include a source filename, word offset, score and the original passage.

## Try it

Requires Python 3.10 or newer; no third-party packages or API keys.

```sh
python3 search.py index examples --output /tmp/doc-search-index.json
python3 search.py query /tmp/doc-search-index.json "cosine TF-IDF"
python3 search.py query /tmp/doc-search-index.json "regression testing" --limit 2
python3 -m unittest discover -s tests -v
```

The first query ranks `retrieval.md` first. The second ranks `testing.md` first.
Unknown terms or an empty query return an empty JSON list rather than an answer
without evidence. Successful commands exit `0`; invalid inputs exit `2`.

## How it works

1. Read UTF-8 `.md` and `.txt` files below the chosen directory in sorted order.
2. Skip hidden paths and symbolic links. Split text into 120-word chunks with
   20 words of overlap; both values are configurable when indexing.
3. Tokenize Unicode words with case folding. For each chunk store term counts.
4. At query time compute `idf = 1 + log((1 + N) / (1 + df))`, where N is the
   number of chunks. Term frequency uses `1 + log(count)`.
5. Rank by cosine similarity; break ties by source path and word offset.

This is **lexical retrieval**, not a language model, semantic embedding search,
or an answer-generating RAG system. It provides a testable baseline for studying
the retrieval stage of a future RAG project.

## Limits and privacy

Everything runs locally. The generated index includes the full document passages
and must be treated as private if the source documents are private. Do not commit
indexes built from personal documents. Only the synthetic sample notes are included.
The CLI handles trusted local indexes, not arbitrary untrusted JSON uploads.

There is no stemming, PDF parsing, multilingual translation, incremental update,
or persistent vector database. Each query scans all chunks; this is intended for
small collections. Rebuild the index after changing your documents.

Tests cover relevant-source ranking, unknown queries, Unicode, chunk boundaries,
stable ties, hidden files/symlinks and an end-to-end CLI roundtrip.

## Learning notes

Portfolio learning project created for Nikolas Lutsenko with AI assistance.
It is not commercial work. Topics to explain: tokenization, sparse vectors,
TF-IDF, cosine normalization, retrieval vs generation and evaluation.
Possible next step: create a labelled query set and measure recall@k before
comparing this baseline against an embedding model.
