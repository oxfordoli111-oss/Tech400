# Basic Information Retrieval System

A small, dependency-free Information Retrieval system built in Python for
the Week 2 assignment, based on Chapters 3 and 4 of *An Introduction to
Information Retrieval* (Manning, Raghavan & Schütze).

The system can:

1. **Add documents** to a collection (from raw text or from a folder of
   `.txt` files).
2. Build a **basic dictionary** (vocabulary + document frequencies).
3. Build an **inverted index** (term → postings list of document IDs).
4. Run **Boolean retrieval** queries using `AND`, `OR`, `NOT`, and
   parentheses.

No external libraries are required — everything runs with the Python 3
standard library.

## Project structure

```
ir_system/
├── documents/              8 sample .txt documents (the test collection)
├── preprocessing.py        tokenization, stop word removal, stemming
├── dictionary.py           the Dictionary class (vocabulary + doc frequency)
├── inverted_index.py       the InvertedIndex class (term -> postings list)
├── boolean_retrieval.py    Boolean query parser and evaluator (AND/OR/NOT)
├── ir_system.py            top-level IRSystem class tying everything together
├── demo.py                 scripted demo: builds the index and runs sample queries
├── query_cli.py            interactive command line query tool
├── test_ir_system.py       unit tests
└── README.md
```

## Running it

```bash
# Run the full scripted demo (builds the index, prints the dictionary,
# the inverted index, and the results of several sample queries)
python demo.py

# Or try your own queries interactively
python query_cli.py

# Run the unit tests
python -m unittest test_ir_system.py -v
```

## Design choices

- **Dictionary**: implemented as a Python `dict` (hash table), which
  Chapter 3 identifies as the right choice for a small, in-memory
  vocabulary. Each entry maps a term to its document frequency.
- **Inverted index**: postings lists are stored internally as Python
  `set` objects so that Boolean `AND` / `OR` / `NOT` can be evaluated
  as fast set intersection / union / difference, then returned to the
  caller as sorted lists (the standard docID-ordered representation
  used in the textbook).
- **Preprocessing**: a minimal pipeline of tokenization, stop word
  removal, and a simplified suffix-stripping stemmer (not the full
  Porter algorithm, but the same underlying idea: merging related word
  forms such as "brew"/"brewed" into a single indexed term).
- **Boolean query parser**: a small recursive-descent parser that
  supports the standard precedence `NOT > AND > OR` and arbitrary
  parenthesised grouping, e.g. `python AND (snake OR language)`.

## Sample collection

The `documents/` folder contains 8 short documents on varied topics
(Python the programming language, pythons the snake, dogs, cats,
machine learning, coffee, space exploration, and football). The topics
were chosen so that some terms overlap in interesting ways — most
notably the word "python", which appears both in the programming
article and in the article about the snake, which is a good
illustration of a limitation of pure keyword matching: the index
cannot distinguish between the two senses of the word.
