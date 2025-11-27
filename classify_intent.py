#!/usr/bin/env python3
"""
intent_classifier.py

Lightweight hybrid intent classifier for navigation vs scene description.
- Step 1: fast regex to catch obvious cases
- Step 2: tiny embedding model (all-MiniLM-L6-v2, ~90MB) for fuzzy/zero-shot decisions

Outputs (stdout):
  1, <destination>    # navigation
  2, <original input> # description
  0                   # unclear

Usage:
  python3 intent_classifier.py "I want to go to the EECS building"
  python3 intent_classifier.py --interactive

Dependencies:
  pip install sentence-transformers regex

Notes:
  * No training required. Works offline once the model is cached.
  * Memory footprint << 500MB (model ~90MB + Python overhead)
"""

import argparse
import sys
import re
import regex as re2
from typing import Tuple, Optional, List

# ---- Regex rules -----------------------------------------------------------
# Keep these concise & portable. We prioritize nav if both match.
NAV_KEYWORDS = [
    r"\b(nav|navigate|navigation)\b",
    r"\b(go|going|head|heading|walk|walking|drive|driving|ride|moving)\b",
    r"\b(take\s+me|bring\s+me|get\s+me)\b",
    r"\b(to|toward|towards|into|onto)\b",  # used with others
    r"\b(home|office|school|campus|dorm|lab|library|eecs|eecs\s+building|supermarket|grocery|target|walmart|starbucks|cafe|hospital|airport|station)\b",
]

# Phrases that strongly indicate description requests
DESC_KEYWORDS = [
    r"\b(what'?s|what\s+is|describe|explain|look|see|show|identify|recognize|detect)\b",
    r"\b(in\s+front\s+of\s+me|around\s+me|near\s+me|on\s+the\s+shelf|ahead\s+of\s+me|to\s+my\s+left|to\s+my\s+right|behind\s+me)\b",
    r"\b(read\s+the\s+sign|read\s+this|price|how\s+much|label|text|count|how\s+many)\b",
]

# Destination extraction patterns. We try them in order.
DEST_PATTERNS = [
    re.compile(r"\bwhere\s+is\s+(?:the\s+)?(?P<dest>.+)$", re.IGNORECASE),
    re.compile(r"\bhow\s+do\s+i\s+get\s+to\s+(?P<dest>.+)$", re.IGNORECASE),
    re.compile(r"\b(?:take|bring|get)\s+me\s+(?:to\s+)?(?P<dest>.+)$", re.IGNORECASE),
    re.compile(r"\b(?:go|head|walk|drive)\s+back\s+(?:to\s+)?(?P<dest>.+)$", re.IGNORECASE),
    re.compile(r"\b(?:go|head(?:ing)?|walk(?:ing)?|drive(?:ing)?|ride|move(?:ing)?)\s+(?:me\s+)?(?:to\s+)?(?P<dest>.+)$", re.IGNORECASE),
    re.compile(r"\b(?:nav(?:igate|igation)?|navigate)\s+(?:to\s+)?(?P<dest>.+)$", re.IGNORECASE),
    re.compile(r"\bto\s+(?P<dest>[^,.!?]+)$", re.IGNORECASE),
    re.compile(r"\b(?:go|head|navigate)\s+(?P<dest>home)\b", re.IGNORECASE),
]

# Words to strip from destination spans
DEST_STRIP = re.compile(r"\b(?:to|the|a|an|my|me|please|now|right\s+now|immediately|quickly|fast|quick)\b", re.IGNORECASE)
TRAIL_PUNCT = re.compile(r"^[\s,:;\-\"]+|[\s,:;\-\"]+$")


def matches_any(patterns: List[str], text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def extract_destination(text: str) -> Optional[str]:
    # Normalize whitespace
    t = text.strip()
    for pat in DEST_PATTERNS:
        m = pat.search(t)
        if m:
            dest = m.group("dest") if "dest" in m.groupdict() else None
            if not dest:
                continue
            # Clean up destination
            dest = TRAIL_PUNCT.sub("", dest)
            dest = DEST_STRIP.sub(" ", dest)
            dest = re.sub(r"\s+", " ", dest).strip()
            # If the dest still contains a leading verb like 'go', trim it
            dest = re.sub(r"^(go|head|navigate|walk|drive)\s+", "", dest, flags=re.IGNORECASE)
            # Trim trailing generic words
            dest = re.sub(r"\b(?:please|thanks?)$", "", dest, flags=re.IGNORECASE).strip()
            # Short heuristics: drop leading 'to'
            dest = re.sub(r"^to\s+", "", dest, flags=re.IGNORECASE).strip()
            if dest:
                return dest
    return None


# ---- Embedding model (tiny, zero-shot via prototypes) ---------------------
# We avoid heavy MNLI zero-shot (often >1GB). Instead we embed the input
# and compare to small prototype sets for each intent using cosine similarity.
try:
    from sentence_transformers import SentenceTransformer, util
    _SBERT_AVAILABLE = True
except Exception:
    _SBERT_AVAILABLE = False

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # ~90MB

NAV_PROTOTYPES = [
    "navigate to the destination",
    "go to <place>",
    "head to <place>",
    "take me to <place>",
    "bring me to <place>",
    "drive to <place>",
    "where is the <place>",
    "how do i get to <place>",
]

DESC_PROTOTYPES = [
    "what is around me",
    "describe what you see",
    "what is in front of me",
    "what's on the shelf",
    "read the sign",
    "how many items do you see",
    "what is the price",
    "are there any seats nearby",
    "is there anything near me",
]

# Thresholds for the embedding decision
SIM_THRESHOLD = 0.28       # minimal similarity to any prototype
MARGIN = 0.04              # require this gap between top-1 and runner-up


class MiniLMJudge:
    def __init__(self):
        if not _SBERT_AVAILABLE:
            self.model = None
            return
        self.model = SentenceTransformer(MODEL_NAME)
        self.nav_emb = self.model.encode(NAV_PROTOTYPES, normalize_embeddings=True)
        self.desc_emb = self.model.encode(DESC_PROTOTYPES, normalize_embeddings=True)

    def judge(self, text: str) -> Tuple[int, Optional[str]]:
        """Return (intent_id, payload or None) following the required format.
        1 => navigation, payload is destination
        2 => description, payload is original input
        0 => unclear
        """
        print("Judging with MiniLM")
        if not self.model:
            return 0, None
        q = self.model.encode([text], normalize_embeddings=True)
        # Cosine similarity vs prototype banks
        nav_score = float(util.cos_sim(q, self.nav_emb).max())
        desc_score = float(util.cos_sim(q, self.desc_emb).max())
        # Decide with thresholding
        if max(nav_score, desc_score) < SIM_THRESHOLD:
            return 0, None
        if nav_score - desc_score > MARGIN:
            dest = extract_destination(text) or infer_destination_fallback(text)
            return (1, dest) if dest else (0, None)
        if desc_score - nav_score > MARGIN:
            return 2, text
        # Too close -> unclear
        return 0, None


def infer_destination_fallback(text: str) -> Optional[str]:
    """If NLP thinks it's navigation but regex couldn't extract the destination,
    try a very lightweight fallback: take the span after the last 'to'/'toward(s)'.
    """
    m = re.search(r"\b(?:to|toward|towards)\s+([^,.!?]+)", text, flags=re.IGNORECASE)
    if m:
        dest = m.group(1)
        dest = TRAIL_PUNCT.sub("", dest)
        dest = DEST_STRIP.sub(" ", dest)
        dest = re.sub(r"\s+", " ", dest).strip()
        return dest or None
    # Single-word homes
    if re.search(r"\bhome\b", text, flags=re.IGNORECASE):
        return "home"
    return None


# ---- Top-level decision policy -------------------------------------------

def classify(text: str, judge: Optional[MiniLMJudge] = None) -> Tuple[int, Optional[str]]:
    t = text.strip()
    if not t:
        return 0, None

    # Primary: rule-based quick paths
    nav_hit = matches_any(NAV_KEYWORDS, t)
    desc_hit = matches_any(DESC_KEYWORDS, t)

    # If both hit, prefer navigation if destination is extractable
    if nav_hit:
        dest = extract_destination(t)
        if dest:
            return 1, dest
        # If nav keywords present but no destination, delay to NLP judge
        # unless it's clearly a pure description too.
        if desc_hit:
            # fall through to judge
            pass
        else:
            # try fallback
            dest = infer_destination_fallback(t)
            if dest:
                return 1, dest
            # else judge

    if desc_hit and not nav_hit:
        return 2, t

    # Secondary: fuzzy judge (embeddings)
    judge = judge or MiniLMJudge()
    return judge.judge(t)


# ---- CLI ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Lightweight intent classifier (regex + MiniLM)")
    ap.add_argument("prompt", nargs="?", help="user input to classify")
    ap.add_argument("--interactive", action="store_true", help="REPL mode")
    args = ap.parse_args()

    judge = MiniLMJudge()

    def emit(intent: int, payload: Optional[str]):
        if intent in (1, 2) and payload:
            print(f"{intent}, {payload}")
        else:
            print("0")

    if args.interactive:
        try:
            while True:
                sys.stdout.write(">> ")
                sys.stdout.flush()
                line = sys.stdin.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    print("0")
                    continue
                intent, payload = classify(line, judge)
                emit(intent, payload)
        except KeyboardInterrupt:
            pass
        return

    if args.prompt is None:
        ap.error("provide a prompt or use --interactive")

    intent, payload = classify(args.prompt, judge)
    if intent == 1 and payload:
        # final cleanup: compress common phrases like "go to the supermarket" -> "supermarket"
        payload = re.sub(r"^(?:go|head|navigate)\s+(?:to\s+)?", "", payload, flags=re.IGNORECASE)
        payload = re.sub(r"\b(?:the|a|an)\b\s*", "", payload, flags=re.IGNORECASE).strip()
    print(f"{intent}, {payload}" if payload else "0")


# ---- Quick tests ----------------------------------------------------------
if __name__ == "__main__":
    # Uncomment to run basic sanity checks when executing the file directly.
    tests = [
        "navigate to the EECS building",
        "I want to go to the supermarket",
        "head home",
        "what's in front of me",
        "what's on the shelf",
        "how much is the apple",
        "could you help",
    ]
    for s in tests:
        print(s)
        print(classify(s))
    main()
