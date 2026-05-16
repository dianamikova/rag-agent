"""
expander.py — Zero-cost query expansion strategies (Level 1)

No API calls.
Uses:
- Synonym expansion via WordNet
- Stemming via NLTK PorterStemmer
- Tokenization via NLTK word_tokenize
- Query decomposition via NLTK sent_tokenize + POS tagging
"""

import re
import nltk

# Download required NLTK data
for _resource in ['punkt_tab', 'wordnet', 'averaged_perceptron_tagger_eng']:
    try:
        nltk.data.find(_resource)
    except LookupError:
        nltk.download(_resource, quiet=True)

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet

# Module-level singletons
_stemmer = PorterStemmer()

try:
    wordnet.synsets("test")
    WORDNET_AVAILABLE = True
except Exception:
    WORDNET_AVAILABLE = False


# Functions

def get_synonyms(word: str) -> list[str]:
    """Return synonyms for a word using WordNet."""
    if not WORDNET_AVAILABLE:
        return []
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name().replace("_", " "))
    return list(set(synonyms))


def simple_stem(word: str) -> str:
    """Stem a word using NLTK PorterStemmer."""
    try:
        return _stemmer.stem(word)
    except Exception:
        return word.lower()

def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphabetic words longer than 2 chars."""
    try:
        return [t.lower() for t in word_tokenize(text) if t.isalpha() and len(t) > 2]
    except Exception:
        import string
        text = text.lower().translate(str.maketrans("", "", string.punctuation))
        return [t for t in text.split() if len(t) > 2]

def decompose_query(query: str) -> list[str]:
    """
    Split compound questions into sub-queries using NLTK.
    First tries sentence splitting, then coordinating conjunctions (CC tags).
    e.g. "What is BM25 and how does HyDE work?" → ["What is BM25", "how does HyDE work"]
    """
    try:
        # 1. Try sentence splitting first
        sentences = sent_tokenize(query)
        if len(sentences) > 1:
            return [s.strip() for s in sentences if len(s.strip()) > 10]

        # 2. Split on coordinating conjunctions via POS tagging
        tokens = word_tokenize(query)
        tagged = nltk.pos_tag(tokens)
        split_indices = [i for i, (_, tag) in enumerate(tagged) if tag == 'CC']

        if not split_indices:
            return [query]

        parts = []
        prev = 0
        for idx in split_indices:
            part = " ".join(tokens[prev:idx]).strip()
            if len(part) > 10:
                parts.append(part)
            prev = idx + 1

        last = " ".join(tokens[prev:]).strip()
        if len(last) > 10:
            parts.append(last)

        return parts if len(parts) > 1 else [query]

    except Exception:
        return [query]

def expand_query(query: str) -> list[str]:
    """
    Level 1 expansion. Returns a list of query variants to try.
    All operations are local — zero API cost.
    """
    variants = [query]
    tokens = tokenize(query)

    # 1. Synonym expansion — swap key terms with synonyms
    for token in tokens:
        syns = get_synonyms(token)
        for syn in syns[:2]:  # max 2 synonyms per word to avoid explosion
            variant = re.sub(r"\b" + re.escape(token) + r"\b", syn, query, flags=re.IGNORECASE)
            if variant != query and variant not in variants:
                variants.append(variant)

    # 2. Stemmed variant
    stemmed_query = " ".join(simple_stem(t) for t in tokens)
    if stemmed_query != query.lower() and stemmed_query not in variants:
        variants.append(stemmed_query)

    # 3. Decomposed sub-queries
    for sq in decompose_query(query):
        if sq not in variants:
            variants.append(sq)

    return variants