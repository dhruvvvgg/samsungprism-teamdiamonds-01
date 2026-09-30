"""A tiny in-memory search index, for the demo."""
from examples.textkit.stats import cosine_similarity, tfidf
from examples.textkit.tokenizing import tokenize


class InvertedIndex:
    """Maps each term to the documents containing it, for fast candidate lookup."""

    def __init__(self):
        self.postings = {}
        self.documents = []

    def add(self, doc_id, text):
        """Add one document to the index."""
        self.documents.append((doc_id, text))
        for term in set(tokenize(text)):
            self.postings.setdefault(term, set()).add(doc_id)

    def candidates(self, query):
        """Document ids sharing at least one term with the query."""
        found = set()
        for term in set(tokenize(query)):
            found |= self.postings.get(term, set())
        return found

    def search(self, query, top_k=5):
        """Rank candidate documents against the query by TF-IDF cosine similarity."""
        docs = [tokenize(text) for _, text in self.documents]
        weights = tfidf(docs + [tokenize(query)])
        query_weights = weights[-1]
        scored = []
        for (doc_id, _), doc_weights in zip(self.documents, weights[:-1]):
            scored.append((doc_id, cosine_similarity(query_weights, doc_weights)))
        scored.sort(key=lambda pair: -pair[1])
        return scored[:top_k]


def highlight(text, query, marker="**"):
    """Wrap every query term found in the text with a marker."""
    terms = set(tokenize(query))
    out = []
    for word in text.split():
        stripped = "".join(c for c in word.lower() if c.isalnum() or c == "_")
        out.append(f"{marker}{word}{marker}" if stripped in terms else word)
    return " ".join(out)
