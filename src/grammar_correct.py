"""
Grammar correction layer using Mistral 7B served locally via Ollama.

ASL grammar differs from English (topic-comment structure, dropped articles,
omitted copula, skipped pronouns). T5's raw output inherits this from the
training data. This module sends the raw translation to Mistral with a
strict "fix grammar only, do not change meaning" system prompt.

Requires Ollama running locally:
    ollama pull mistral
    ollama serve

Usage as a library:
    from grammar_correct import correct_grammar
    clean_text, confidence = correct_grammar("me want water drink now please")

Usage as CLI (for quick testing):
    python src/grammar_correct.py "bathroom where"
"""

import argparse
import json
import sys

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"

SYSTEM_PROMPT = (
    "You are a grammar correction assistant for ASL-to-English translation output. "
    "The input text is a raw, telegraphic translation from American Sign Language "
    "that may be missing articles, pronouns, or proper verb conjugation, and may "
    "use topic-comment word order instead of standard English syntax.\n\n"
    "Your ONLY job is to rewrite the input into grammatically correct, natural "
    "English. Do NOT add new information. Do NOT change the meaning. Do NOT "
    "answer questions, do NOT have a conversation, do NOT add commentary.\n\n"
    "Respond with ONLY the corrected sentence and nothing else - no quotes, "
    "no explanation, no preamble."
)


def correct_grammar(raw_text, timeout=15):
    """Returns (corrected_text, confidence). confidence is a heuristic in [0,1]
    based on response validity, since Ollama's /api/generate does not expose
    true log-prob confidence by default."""

    if not raw_text or not raw_text.strip():
        return "", 0.0

    prompt = f"{SYSTEM_PROMPT}\n\nInput: {raw_text}\nOutput:"

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        corrected = data.get("response", "").strip()
    except requests.exceptions.RequestException as e:
        print(f"WARNING: Ollama request failed ({e}). Falling back to raw text.", file=sys.stderr)
        return raw_text, 0.0

    if not corrected:
        return raw_text, 0.3

    confidence = estimate_confidence(raw_text, corrected)
    return corrected, confidence


FUNCTIONAL_WORDS = {
    # articles
    "a", "an", "the",
    # pronouns
    "i", "me", "my", "myself", "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "we", "us", "our", "ours", "ourselves", "they", "them", "their", "theirs", "themselves",
    "who", "whom", "whose", "which", "what", "that", "this", "these", "those",
    # prepositions
    "in", "on", "at", "to", "for", "from", "of", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "up", "down",
    "over", "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    # copulas/auxiliaries
    "is", "am", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "can", "could", "shall", "should", "will", "would", "may",
    "might", "must", "go", "going", "went",
    # conjunctions
    "and", "but", "or", "because", "as", "until", "while", "if", "s"
}


def get_stem(word):
    word = word.lower().strip(".,?!;:-'\"()[]{}")
    if len(word) <= 3:
        return word
    # Strip common suffixes
    if word.endswith("ing"):
        return word[:-3]
    if word.endswith("ed"):
        if word.endswith("ied"):
            return word[:-3] + "y"
        return word[:-2]
    if word.endswith("es"):
        if word.endswith("ies"):
            return word[:-3] + "y"
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    if word.endswith("ly"):
        return word[:-2]
    return word


def estimate_confidence(raw_text, corrected_text):
    """Estimate translation confidence with a length-ratio check and a content-word overlap check.
    
    Penalizes corrections that introduce content words absent from the raw translation.
    """
    if not corrected_text.strip():
        return 0.0

    raw_words = [w.lower().strip(".,?!;:-'\"()[]{}") for w in raw_text.split()]
    corrected_words = [w.lower().strip(".,?!;:-'\"()[]{}") for w in corrected_text.split()]

    raw_words = [w for w in raw_words if w]
    corrected_words = [w for w in corrected_words if w]

    raw_len = len(raw_words)
    corrected_len = len(corrected_words)

    if raw_len == 0:
        return 0.0

    # 1. Length ratio check
    length_ratio = corrected_len / raw_len
    if length_ratio > 3.0 or length_ratio < 0.3:
        return 0.4  # likely degenerate / hallucinated output

    # Base score
    score = 0.9
    if length_ratio > 1.0:
        score = 0.85  # mild expansion is expected (adding articles/pronouns)

    # 2. Content word overlap check
    raw_content_stems = {get_stem(w) for w in raw_words if w not in FUNCTIONAL_WORDS}
    corrected_content_stems = {get_stem(w) for w in corrected_words if w not in FUNCTIONAL_WORDS}

    new_content_stems = corrected_content_stems - raw_content_stems

    # Allow some common acceptable semantic expansions / helper content words
    acceptable_new = {
        "like", "would", "right", "please", "want", "go", "get", "take", "make", "let", 
        "show", "tell", "speak", "talk", "say", "need", "should", "could", "must", "can",
        "please", "thank", "hello", "hi", "hey", "good", "morning", "afternoon", "evening"
    }
    
    unacceptable_new = {stem for stem in new_content_stems if stem not in acceptable_new}

    # Penalize score for each unacceptable new content word introduced
    if unacceptable_new:
        penalty = 0.2 * len(unacceptable_new)
        score = max(0.2, score - penalty)
        print(f"DEBUG: introduced unauthorized content words {unacceptable_new}, applying penalty of {penalty:.2f}", file=sys.stderr)

    return round(score, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="Raw translation text to correct")
    args = parser.parse_args()

    corrected, confidence = correct_grammar(args.text)
    print(json.dumps({
        "raw": args.text,
        "corrected": corrected,
        "confidence": confidence,
    }, indent=2))


if __name__ == "__main__":
    main()
