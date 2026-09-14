"""
test_ir_system.py
------------------
A small set of unit tests covering preprocessing, the dictionary, the
inverted index, and Boolean retrieval. Run with:

    python -m unittest test_ir_system.py -v
"""

import unittest

from preprocessing import tokenize, remove_stopwords, stem, preprocess
from ir_system import IRSystem


class TestPreprocessing(unittest.TestCase):
    def test_tokenize_lowercases_and_splits(self):
        self.assertEqual(
            tokenize("Cats, Dogs and PYTHONS!"),
            ["cats", "dogs", "and", "pythons"],
        )

    def test_remove_stopwords(self):
        self.assertEqual(
            remove_stopwords(["the", "cat", "is", "on", "the", "mat"]),
            ["cat", "mat"],
        )

    def test_stem_basic_suffixes(self):
        self.assertEqual(stem("running"), "runn")
        self.assertEqual(stem("brewed"), "brew")
        self.assertEqual(stem("cats"), "cat")

    def test_preprocess_pipeline(self):
        terms = preprocess("The cats are running and playing")
        self.assertNotIn("the", terms)
        self.assertNotIn("are", terms)
        self.assertIn("cat", terms)


class TestIRSystem(unittest.TestCase):
    def setUp(self):
        self.system = IRSystem()
        self.system.add_document("d1", "Python is a programming language")
        self.system.add_document("d2", "The python snake lives in Africa")
        self.system.add_document("d3", "Dogs and cats are popular pets")
        self.system.build_index()

    def test_dictionary_built(self):
        self.assertGreater(self.system.dictionary.size(), 0)
        self.assertIn("python", self.system.dictionary.vocabulary())

    def test_inverted_index_postings(self):
        postings = self.system.index.postings("python")
        self.assertEqual(postings, ["d1", "d2"])

    def test_boolean_and(self):
        self.assertEqual(self.system.search("python AND language"), ["d1"])

    def test_boolean_or(self):
        self.assertEqual(sorted(self.system.search("cat OR snake")), ["d2", "d3"])

    def test_boolean_not(self):
        result = self.system.search("python AND NOT language")
        self.assertEqual(result, ["d2"])

    def test_boolean_parentheses(self):
        result = self.system.search("python AND (snake OR language)")
        self.assertEqual(result, ["d1", "d2"])

    def test_no_match_returns_empty(self):
        self.assertEqual(self.system.search("nonexistentterm"), [])

    def test_duplicate_document_id_raises(self):
        with self.assertRaises(ValueError):
            self.system.add_document("d1", "duplicate")


if __name__ == "__main__":
    unittest.main()
