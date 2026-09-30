"""
cli_chat.py
-----------
Minimal terminal chat loop for testing the matcher without the Flask UI.
Run with: python cli_chat.py
"""

from matcher import FALLBACK_RESPONSE, FAQMatcher
from preprocess import using_nltk


def main():
    matcher = FAQMatcher("faqs.json", threshold=0.25)
    print("Fern & Co. FAQ bot (NLTK backend active: {})".format(using_nltk()))
    print(f"Loaded {len(matcher.faqs)} FAQs. Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if user_input.lower() in {"quit", "exit"}:
            print("Bye!")
            break
        if not user_input:
            continue

        result = matcher.get_best_match(user_input)
        if result.matched:
            print(f"Bot ({result.score:.2f}): {result.answer}\n")
        else:
            print(f"Bot: {FALLBACK_RESPONSE}\n")


if __name__ == "__main__":
    main()
