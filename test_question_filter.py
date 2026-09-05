"""
test_question_filter.py — is_question() against the things it gets wrong.

This is the cheapest place in the pipeline to be wrong twice over. A false
positive costs a transcription request, an LLM call, and an answer landing
over the panel while the interviewer is still speaking. A false negative
costs one press of the grab hotkey. So the filter is deliberately strict —
but it was strict in the wrong places, and loose in others.

Run: python test_question_filter.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transcribe import is_question


# Things that must be answered.
SHOULD_PASS = [
    # Ordinary openers
    ("What's the difference between a process and a thread?", "plain starter"),
    ("How would you scale this?", "starter"),
    ("Why Python?", "two words is enough with a starter"),
    ("Tell me about a time you disagreed with a manager", "starter, no qmark"),
    ("Walk me through your approach", "starter, no qmark"),
    ("Describe a system you designed", "starter"),
    ("Is that thread safe?", "'is that' starter"),
    ("Was the migration your idea?", "'was' with a real boundary after it"),
    ("Does that scale to a million rows?", "'does' with a real boundary"),

    # Four-word questions that the old five-word floor threw away
    ("Any questions for me?", "no starter, four words, qmark"),
    ("Big O of that?", "no starter, four words, qmark"),

    # Buried asks
    ("Okay that makes sense. Can you explain the tradeoff?", "embedded ask"),
    ("Right, so tell me about the caching layer", "embedded ask"),
]

# Things that must NOT cost an answer.
SHOULD_FAIL = [
    # The bug this file exists for: prefix matching without a word boundary
    ("Washington is where I grew up", "'was' is a prefix of Washington"),
    ("Washington", "same, single word"),
    ("doesn't matter, let's move on", "'does' is a prefix of doesn't"),
    ("Hashing collisions are handled by chaining", "'has' is a prefix of Hashing"),
    ("Whenever you're ready", "'when' is a prefix of Whenever"),

    # Filler, the reason the filter exists
    ("Mm-hmm", "filler"),
    ("Right", "filler"),
    ("Thank you.", "filler with punctuation"),
    ("Okay?", "filler with a question mark"),
    ("So", "filler"),
    ("", "empty"),

    # Short fragments with a question mark but nothing else
    ("You too, right?", "three words, no starter"),
    ("Hmm?", "one word"),
]


def main():
    failures = []

    for text, why in SHOULD_PASS:
        if not is_question(text):
            failures.append(f"  MISSED  {text!r}\n          (should pass: {why})")

    for text, why in SHOULD_FAIL:
        if is_question(text):
            failures.append(f"  ANSWERED {text!r}\n          (should fail: {why})")

    total = len(SHOULD_PASS) + len(SHOULD_FAIL)
    if failures:
        print(f"{len(failures)} of {total} cases wrong:\n")
        print("\n".join(failures))
        return 1

    print(f"All {total} cases correct "
          f"({len(SHOULD_PASS)} answered, {len(SHOULD_FAIL)} ignored).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
