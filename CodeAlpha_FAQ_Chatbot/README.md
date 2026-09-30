# FAQ Chatbot

A small FAQ chatbot that matches a user's question against a stored set of
Q&A pairs using NLP preprocessing (NLTK) and TF-IDF + cosine similarity
(scikit-learn), served through a Flask API and a simple chat UI.

## How it works

1. **`faqs.json`** — the FAQ knowledge base: a list of `{"question", "answer"}`
   pairs. This demo uses a plant-shop support theme; swap in your own.
2. **`preprocess.py`** — cleans and normalizes text: lowercasing, punctuation
   removal, tokenization, stopword removal, and lemmatization via NLTK. If
   NLTK or its corpora aren't available, it automatically falls back to a
   lightweight built-in stopword list and stemmer so the app still runs.
3. **`matcher.py`** — fits a `TfidfVectorizer` on all preprocessed FAQ
   questions, then for each user message computes cosine similarity against
   every FAQ and returns the best match (if its score clears a confidence
   threshold, default `0.25`).
4. **`app.py`** — a Flask server exposing `POST /api/chat` and serving the
   chat UI in `templates/index.html` / `static/`.
5. **`cli_chat.py`** — a terminal-only version, for testing the matching
   logic without running the web server.

## Setup

```bash
pip install -r requirements.txt
```

The first run downloads a few small NLTK corpora (punkt, stopwords, wordnet)
automatically — this needs an internet connection once. If it can't reach
the internet, the app still runs using the built-in fallback preprocessing.

## Run it

**Web chat UI:**
```bash
python app.py
```
Then open http://localhost:5000

**Terminal version:**
```bash
python cli_chat.py
```

## Using your own FAQs

Replace the contents of `faqs.json` with your own list:

```json
[
  { "question": "How do I reset my password?", "answer": "Go to Settings > Security > Reset Password." },
  { "question": "What are your business hours?", "answer": "We're open Monday-Friday, 9am-5pm EST." }
]
```

More FAQs generally improve matching, since TF-IDF has more vocabulary and
phrasing variety to compare against.

## Tuning

- **`threshold`** in `FAQMatcher(...)` (in `app.py` / `cli_chat.py`) controls
  how confident a match must be before it's shown; lower it if the bot is
  too often saying it doesn't know, raise it if it's giving wrong answers
  too confidently.
- **`ngram_range`** in `matcher.py`'s `TfidfVectorizer` — `(1, 2)` by default
  captures both single words and two-word phrases.

## Known limitation & optional upgrade

TF-IDF + cosine similarity matches on *shared wording*, not meaning — so
"can I send this back?" may not match a FAQ that only says "return policy"
if there's no NLTK/vocabulary overlap. For better semantic (intent-based)
matching, swap the vectorizer in `matcher.py` for sentence embeddings, e.g.:

```bash
pip install sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer("all-MiniLM-L6-v2")
faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)
query_embedding = model.encode(user_query, convert_to_tensor=True)
scores = util.cos_sim(query_embedding, faq_embeddings)[0]
```

This trades a larger model download for much better handling of paraphrased
or differently-worded questions.
