"""
test_vector_space.py
---------------------
Run with:  python -m unittest test_vector_space.py -v
"""

import math
import unittest

from vector_space import VectorSpaceModel


class TestVectorSpaceModel(unittest.TestCase):
    def setUp(self):
        self.docs = {
            "d1": "The cat sat on the mat. The cat was happy.",
            "d2": "The dog sat on the mat. The dog was happy.",
            "d3": "Stock markets rose today after new economic data.",
        }
        self.vsm = VectorSpaceModel(self.docs)

    def test_vector_is_unit_length(self):
        for doc_id in self.docs:
            vec = self.vsm.vector(doc_id)
            length = math.sqrt(sum(w * w for w in vec.values()))
            self.assertAlmostEqual(length, 1.0, places=6)

    def test_self_similarity_is_one(self):
        for doc_id in self.docs:
            self.assertAlmostEqual(
                self.vsm.cosine_similarity(doc_id, doc_id), 1.0, places=6
            )

    def test_similarity_is_symmetric(self):
        self.assertAlmostEqual(
            self.vsm.cosine_similarity("d1", "d2"),
            self.vsm.cosine_similarity("d2", "d1"),
            places=9,
        )

    def test_similar_documents_score_higher_than_unrelated(self):
        # d1 and d2 share most of their (non-stop) vocabulary structure
        # ("sat", "mat", "happy"); d3 is about an unrelated topic.
        sim_d1_d2 = self.vsm.cosine_similarity("d1", "d2")
        sim_d1_d3 = self.vsm.cosine_similarity("d1", "d3")
        self.assertGreater(sim_d1_d2, sim_d1_d3)

    def test_most_similar_pair(self):
        pair, score = self.vsm.most_similar_pair()
        self.assertEqual(set(pair), {"d1", "d2"})

    def test_idf_is_zero_for_terms_in_every_document(self):
        docs = {"a": "cat dog", "b": "cat bird", "c": "cat fish"}
        vsm = VectorSpaceModel(docs)
        self.assertAlmostEqual(vsm.idf("cat"), 0.0, places=9)

    def test_top_terms_returns_requested_count(self):
        terms = self.vsm.top_terms("d3", n=3)
        self.assertLessEqual(len(terms), 3)


if __name__ == "__main__":
    unittest.main()
