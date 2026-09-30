"""
preprocess.py
-------------
NLP preprocessing for the FAQ chatbot: lowercasing, punctuation removal,
tokenization, stopword removal, and lemmatization.

Tries to use NLTK (as requested). If NLTK or its corpora aren't available
in the current environment (e.g. no internet to download them), it falls
back to a small built-in stopword list and a lightweight regex tokenizer,
so the rest of the pipeline still works without interruption.
"""

import re
import string

_NLTK_READY = False

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    # Make sure the required corpora are present; download quietly if not.
    _required = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, pkg in _required:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)

    _stop_words = set(stopwords.words("english"))
    _lemmatizer = WordNetLemmatizer()
    _NLTK_READY = True

except Exception:
    # NLTK not installed, or no network to fetch corpora -> use fallback below.
    _NLTK_READY = False


# --- Fallback resources (used only if NLTK isn't available) ---------------

_FALLBACK_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "could", "did", "do", "does", "doing", "down", "for", "from", "had",
    "has", "have", "having", "how", "i", "if", "in", "into", "is", "it",
    "its", "just", "me", "my", "of", "on", "or", "should", "so", "some",
    "such", "than", "that", "the", "their", "them", "then", "there",
    "these", "they", "this", "to", "too", "up", "very", "was", "we",
    "were", "what", "when", "where", "which", "who", "whom", "why",
    "will", "with", "would", "you", "your", "yours",
}


def _fallback_tokenize(text: str):
    return re.findall(r"[a-z0-9']+", text.lower())


def _simple_stem(word: str) -> str:
    """Very small rule-based suffix stripper, used only as a fallback
    when NLTK's WordNetLemmatizer isn't available."""
    for suffix in ("ing", "edly", "ed", "ies", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


# --- Public API -------------------------------------------------------------

def clean_text(text: str) -> str:
    """Lowercase and strip punctuation/extra whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(text: str) -> str:
    """
    Full pipeline: clean -> tokenize -> remove stopwords -> lemmatize/stem.
    Returns a single space-joined string, ready to be fed to TfidfVectorizer.
    """
    cleaned = clean_text(text)

    if _NLTK_READY:
        tokens = word_tokenize(cleaned)
        tokens = [t for t in tokens if t not in _stop_words and len(t) > 1]
        tokens = [_lemmatizer.lemmatize(t) for t in tokens]
    else:
        tokens = _fallback_tokenize(cleaned)
        tokens = [t for t in tokens if t not in _FALLBACK_STOPWORDS and len(t) > 1]
        tokens = [_simple_stem(t) for t in tokens]

    return " ".join(tokens)


def using_nltk() -> bool:
    """Lets callers report which preprocessing backend is active."""
    return _NLTK_READY


if __name__ == "__main__":
    sample = "How do I track my order after it has shipped?"
    print(f"NLTK backend active: {using_nltk()}")
    print(f"Original : {sample}")
    print(f"Processed: {preprocess(sample)}")
