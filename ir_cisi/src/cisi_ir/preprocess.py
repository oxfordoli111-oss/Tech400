"""Text preprocessing pipeline (Manning et al., Ch. 2): tokenise -> normalise -> stop -> stem."""
import re
from nltk.stem import PorterStemmer

# Compact English stop list (function words with very high collection frequency, Sec. 2.2.2)
STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have
having he her here hers him his how i if in into is it its itself just me more most my no nor not now
of off on once only or other our out over own same she should so some such than that the their theirs
them then there these they this those through to too under until up very was we were what when where
which while who whom why will with would you your yours also may might shall upon via using use used
""".split())

_TOKEN = re.compile(r"[a-z0-9]+")


class Preprocessor:
    """Configurable analyzer so that documents and queries share the *same* pipeline."""

    def __init__(self, lowercase=True, remove_stopwords=True, stem=True):
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.stem = stem
        self._stemmer = PorterStemmer()
        self._cache = {}

    def _stem(self, tok):
        s = self._cache.get(tok)
        if s is None:
            s = self._cache[tok] = self._stemmer.stem(tok)
        return s

    def __call__(self, text):
        if self.lowercase:
            text = text.lower()
        tokens = _TOKEN.findall(text)
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in STOPWORDS]
        if self.stem:
            tokens = [self._stem(t) for t in tokens]
        return tokens

    def describe(self):
        return f"lower={self.lowercase}, stop={self.remove_stopwords}, stem={self.stem}"
