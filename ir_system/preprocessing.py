"""
preprocessing.py
-----------------
Text preprocessing utilities for the Information Retrieval system.

This module implements the three preprocessing steps discussed in
Chapters 2-3 of "An Introduction to Information Retrieval" (Manning,
Raghavan & Schutze):

1. Tokenization  - splitting raw text into individual terms.
2. Stop word removal - discarding very common words that carry little
   discriminative value for retrieval (e.g. "the", "is", "and").
3. Stemming - reducing inflected words to a common root form so that,
   for example, "running" and "run" are treated as the same term.

No external libraries (e.g. nltk) are used, so the system has zero
external dependencies and can run anywhere Python 3 is installed.
"""

import re

# A small, standard list of English stop words. This is not exhaustive,
# but it covers the most frequent function words in English text.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "for",
    "of", "in", "on", "at", "to", "from", "by", "with", "as", "is",
    "are", "was", "were", "be", "been", "being", "it", "its", "this",
    "that", "these", "those", "there", "their", "they", "them", "we",
    "you", "he", "she", "his", "her", "i", "not", "no", "do", "does",
    "did", "has", "have", "had", "will", "would", "can", "could",
    "such", "than", "too", "very", "into", "over", "some", "many",
    "most", "more", "other", "which", "who", "whom", "so", "up",
    "down", "out", "about", "also", "each",
}

# A short list of common suffixes, longest first, used by the simple
# stemmer below. This is a lightweight, rule based stemmer rather than
# a full implementation of the Porter algorithm referenced in the
# textbook, but it demonstrates the same underlying idea: stripping
# inflectional endings so that related word forms collapse to one term.
_SUFFIXES = ["ational", "ational", "ing", "edly", "ed", "ly", "es", "s"]


def tokenize(text):
    """Split raw text into lowercase alphabetic tokens.

    Punctuation and digits are treated as token boundaries, mirroring
    the simple tokenization approach described in Chapter 2 of the
    textbook.
    """
    text = text.lower()
    tokens = re.findall(r"[a-z]+", text)
    return tokens


def remove_stopwords(tokens):
    """Remove tokens that appear in the STOPWORDS list."""
    return [t for t in tokens if t not in STOPWORDS]


def stem(token):
    """Apply a small set of suffix-stripping rules to a single token.

    This is intentionally simple (a handful of rules) rather than a
    full Porter stemmer implementation, but it is enough to merge
    common inflected forms, e.g. "runs"/"running" -> "run",
    "brewed"/"brewing" -> "brew".
    """
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def preprocess(text, use_stemming=True):
    """Run the full preprocessing pipeline on a piece of text.

    Returns a list of processed terms ready to be added to the
    dictionary / inverted index.
    """
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    if use_stemming:
        tokens = [stem(t) for t in tokens]
    return tokens
