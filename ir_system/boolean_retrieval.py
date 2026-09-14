"""
boolean_retrieval.py
---------------------
Implements Boolean retrieval on top of the InvertedIndex, as described
in Chapter 1 of the textbook and assumed throughout Chapters 3-4.

Supported query syntax:
    AND, OR, NOT   (case-insensitive keywords)
    parentheses for grouping, e.g. "(cat OR dog) AND NOT snake"

Queries are parsed with a small recursive-descent parser implementing
the standard operator precedence NOT > AND > OR, which is the same
precedence used by most Boolean search systems and by the worked
examples in the textbook.

Each operator is evaluated as a set operation over postings lists:
    AND -> set intersection
    OR  -> set union
    NOT -> set difference against the universe of all document IDs
"""

import re

from preprocessing import stem

_TOKEN_RE = re.compile(r"\(|\)|[A-Za-z]+")


class QueryParseError(Exception):
    pass


class BooleanRetrieval:
    def __init__(self, inverted_index, use_stemming=True):
        self.index = inverted_index
        self.use_stemming = use_stemming

    # ---------- public API ----------

    def search(self, query):
        """Evaluate a Boolean query string and return a sorted list of
        matching document IDs."""
        tokens = _TOKEN_RE.findall(query)
        if not tokens:
            return []
        parser = _Parser(tokens, self)
        result_set = parser.parse_expression()
        parser.expect_end()
        return sorted(result_set)

    # ---------- helpers used by the parser ----------

    def term_postings(self, word):
        term = word.lower()
        if self.use_stemming:
            term = stem(term)
        return self.index.postings_set(term)

    def universe(self):
        return set(self.index.all_doc_ids)


class _Parser:
    """Recursive-descent parser implementing:

        expression := term (OR term)*
        term       := factor (AND factor)*
        factor     := NOT factor | '(' expression ')' | WORD
    """

    def __init__(self, tokens, engine):
        self.tokens = tokens
        self.pos = 0
        self.engine = engine

    def _peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _advance(self):
        tok = self._peek()
        self.pos += 1
        return tok

    def expect_end(self):
        if self._peek() is not None:
            raise QueryParseError(f"Unexpected token near '{self._peek()}'")

    def parse_expression(self):
        result = self.parse_term()
        while self._peek() and self._peek().upper() == "OR":
            self._advance()
            result = result | self.parse_term()
        return result

    def parse_term(self):
        result = self.parse_factor()
        while self._peek() and self._peek().upper() == "AND":
            self._advance()
            result = result & self.parse_factor()
        return result

    def parse_factor(self):
        tok = self._peek()
        if tok is None:
            raise QueryParseError("Unexpected end of query")

        if tok.upper() == "NOT":
            self._advance()
            return self.engine.universe() - self.parse_factor()

        if tok == "(":
            self._advance()
            result = self.parse_expression()
            if self._peek() != ")":
                raise QueryParseError("Missing closing parenthesis")
            self._advance()
            return result

        if tok in ("AND", "OR", ")"):
            raise QueryParseError(f"Unexpected operator '{tok}'")

        # otherwise: a plain search term
        self._advance()
        return self.engine.term_postings(tok)
