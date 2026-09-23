# Document Similarity with the Vector Space Model

A Week 3 assignment: given a collection of documents, compute a
similarity score between every pair of them using the vector space
model from Chapter 6 of *An Introduction to Information Retrieval*
(Manning, Raghavan & Schütze).

No external libraries are required -- tf-idf weighting and cosine
similarity are implemented directly from the textbook's formulas using
only the Python standard library.

## Document collection

The 8 documents in `documents/` are real excerpts from official U.S.
federal government sources -- genuine news and press releases, not
Wikipedia. Works produced by the U.S. federal government are not
subject to copyright domestically (17 U.S.C. § 105), so they can be
reproduced in full without any licensing concern:

| Topic | Source |
|---|---|
| Renewable energy | U.S. Energy Information Administration (eia.gov) |
| Artificial intelligence | National Institute of Standards and Technology (nist.gov) |
| Football | U.S. Army / DVIDS (dvidshub.net) |
| Nutrition | USDA / HHS (usda.gov) |
| Financial markets | Securities and Exchange Commission (sec.gov) |
| Space exploration | NASA (science.nasa.gov) |
| Wildlife conservation | U.S. Fish and Wildlife Service (fws.gov) |
| Smartphones / tech | Federal Communications Commission (fcc.gov) |

Each file starts with a `Source:` and `License:` header recording
where it came from (the loader strips this header before indexing, so
it doesn't affect the vocabulary or similarity scores). If you'd
rather use your own gathered articles instead, just replace any file's
body text and re-run `python demo.py` -- no code changes needed.

## Project structure

```
ir_similarity/
├── documents/          your document collection (.txt files)
├── preprocessing.py     tokenization, stop word removal, stemming (from Week 2)
├── vector_space.py      tf-idf weighting + cosine similarity (the core of this week's work)
├── demo.py              scripted demo: similarity matrix, most similar pairs, top terms
├── compare_cli.py        interactive tool to compare any two documents by hand
├── test_vector_space.py test suite
└── README.md
```

## Running it

```bash
python demo.py                                   # full similarity report
python compare_cli.py                            # compare any two documents interactively
python -m unittest test_vector_space.py -v       # unit tests
```

## How it works

Each document is turned into a vector of tf-idf weights, one dimension
per vocabulary term (Section 6.2-6.3):

- **Term frequency**: log-weighted, `wf(t,d) = 1 + log(tf(t,d))`, so
  that (per Section 6.4.1) ten occurrences of a word don't count as
  literally ten times more important than one occurrence.
- **Inverse document frequency**: `idf(t) = log(N / df(t))`, which
  down-weights terms that appear in most documents in the collection.
- **Cosine normalization**: each document's weight vector is divided
  by its own length, so similarity scores aren't biased by document
  length.

This weighting combination is `ltc` in the textbook's SMART notation
(Section 6.4.3). The similarity between two documents is then just the
dot product of their two (already unit-length) vectors -- the standard
cosine similarity formula (Equation 6.11).

## Sample output

Running `demo.py` on this collection produces an 8x8 similarity
matrix, the single most similar pair in the collection, the top-2 most
similar documents for every document, and the top-5 highest-weighted
(most distinguishing) terms per document. The renewable-energy article
(EIA) and the space-mission article (NASA) come out as the most
similar pair, both being technical government reports that share
numerical/measurement-heavy vocabulary -- a reminder that similarity
here is about shared wording, not necessarily shared subject matter.
