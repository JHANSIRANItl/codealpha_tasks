"""
app.py
------
Flask server for the FAQ chatbot. Serves a simple chat UI and exposes
a POST /api/chat endpoint that returns the best-matching FAQ answer.
"""

from flask import Flask, jsonify, render_template, request

from matcher import FALLBACK_RESPONSE, FAQMatcher
from preprocess import using_nltk

app = Flask(__name__)
matcher = FAQMatcher("faqs.json", threshold=0.25)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()

    if not user_message:
        return jsonify({"error": "Empty message."}), 400

    result = matcher.get_best_match(user_message)

    if result.matched:
        return jsonify({
            "reply": result.answer,
            "matched_question": result.question,
            "confidence": round(result.score, 3),
        })
    else:
        return jsonify({
            "reply": FALLBACK_RESPONSE,
            "matched_question": None,
            "confidence": round(result.score, 3),
        })


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "nltk_active": using_nltk(), "faq_count": len(matcher.faqs)})


if __name__ == "__main__":
    print(f"NLTK preprocessing active: {using_nltk()}")
    print(f"Loaded {len(matcher.faqs)} FAQs.")
    app.run(debug=True, port=5000)
