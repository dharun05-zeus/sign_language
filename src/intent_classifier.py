"""
Lightweight rule-based intent classifier for question / command / statement.

A learned classifier would be more robust, but for a first production pass
this rule-based tagger is fast, has zero extra VRAM cost, and is easy to
debug - which matters since it runs on every single output before it reaches
the agentic AI. Swap in a small fine-tuned classifier later if rule-based
accuracy proves insufficient on real user data.

Usage:
    from intent_classifier import classify_intent
    classify_intent("Where is the nearest hospital?")  -> "question"
    classify_intent("Schedule a doctor appointment.")  -> "command"
    classify_intent("I have a meeting at 3 PM.")        -> "statement"
"""

import re

QUESTION_WORDS = {
    "who", "what", "when", "where", "why", "how", "which", "whose",
    "is", "are", "am", "do", "does", "did", "can", "could", "will",
    "would", "should", "shall", "has", "have", "had",
}

COMMAND_VERBS = {
    "schedule", "set", "remind", "call", "send", "create", "add", "delete",
    "remove", "cancel", "book", "find", "search", "show", "open", "close",
    "start", "stop", "play", "pause", "turn", "set up", "make", "give",
    "tell", "get", "put", "move", "check", "update", "save", "send",
}


def classify_intent(text):
    text = text.strip()
    if not text:
        return "statement"

    if text.endswith("?"):
        return "question"

    first_word = re.split(r"\s+", text.lower())[0].strip(".,!?")
    if first_word in QUESTION_WORDS:
        return "question"

    if first_word in COMMAND_VERBS:
        return "command"

    # imperative heuristic: sentence starts with a bare verb and has no
    # explicit subject pronoun (I, you, he, she, they, we, it) immediately after
    words = re.split(r"\s+", text.lower())
    if len(words) >= 2 and words[1] not in {"i", "you", "he", "she", "they", "we", "it"}:
        if first_word.endswith(("e", "y")) or first_word in COMMAND_VERBS:
            # weak signal alone; only escalate to command if verb-like AND
            # no clear subject - otherwise default to statement
            pass

    return "statement"


if __name__ == "__main__":
    tests = [
        "Where is the nearest hospital?",
        "Schedule a doctor appointment.",
        "I have a meeting at 3 PM.",
        "Call my mom",
        "What time is it",
    ]
    for t in tests:
        print(f"{t!r:45s} -> {classify_intent(t)}")
