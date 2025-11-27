#!/usr/bin/env python3
"""
intent_classifier.py

Lightweight hybrid intent classifier for navigation vs scene description.
- Step 1: fast regex to catch obvious cases
- Step 2: tiny embedding model (all-MiniLM-L6-v2, ~90MB) for fuzzy/zero-shot decisions

Outputs (stdout):
  1, <destination>    # navigation
  2, <question>       # description with a specific question
  2                   # description (general, no specific question)
  0                   # unclear
"""

import argparse
import sys
import re
import logging
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# ---- Regex rules -----------------------------------------------------------
# Keep these concise & portable. We prioritize nav if both match.
NAV_KEYWORDS = [
    r"\b(nav|navigate|navigation|navigated|navigating)\b",
    r"\b(go|going|head|heading|walk|walking|drive|driving|ride|moving)\b",
    r"\b(take\s+me|bring\s+me|get\s+me)\b",
    r"\b(to|toward|towards|into|onto)\b",  # used with others
    r"\b(home|office|school|campus|dorm|lab|library|eecs|eecs\s+building|supermarket|grocery|target|walmart|starbucks|cafe|hospital|airport|station)\b",
]

# Description regex is now ONLY for specific questions (not general “describe/what do you see”)
DESC_KEYWORDS = [
    r"\b(how\s+much|price|how\s+many|count)\b",
    r"\b(are\s+there|is\s+there)\b",
    r"\bread\s+(?:the\s+)?(sign|label|text)\b",
]

# Destination extraction patterns. We try them in order.
DEST_PATTERNS: List[re.Pattern] = [
    # question forms
    re.compile(r"\bwhere\s+is\s+(?:the\s+)?(?P<dest>.+)$", re.IGNORECASE),
    re.compile(r"\bhow\s+do\s+i\s+get\s+to\s+(?P<dest>.+)$", re.IGNORECASE),
    # take me / bring me / get me to X
    re.compile(r"\b(?:take|bring|get)\s+me\s+(?:to\s+)?(?P<dest>.+)$", re.IGNORECASE),
    # go back to X
    re.compile(r"\b(?:go|head|walk|drive)\s+back\s+(?:to\s+)?(?P<dest>.+)$", re.IGNORECASE),
    # go/heading/walk/drive to X
    re.compile(r"\b(?:go|head(?:ing)?|walk(?:ing)?|drive(?:ing)?|ride|move(?:ing)?)\s+(?:me\s+)?(?:to\s+)?(?P<dest>.+)$", re.IGNORECASE),
    # navigate to X / navigation to X
    re.compile(r"\b(?:nav(?:igate|igation)?|navigate)\s+(?:to\s+)?(?P<dest>.+)$", re.IGNORECASE),
    # just "to X" at the end
    re.compile(r"\bto\s+(?P<dest>[^,.!?]+)$", re.IGNORECASE),
    # "head home" / "go home"
    re.compile(r"\b(?:go|head|navigate)\s+(?P<dest>home)\b", re.IGNORECASE),
]

# Phrases that often appear after a destination but are not part of it (purpose/time/companions)
TAIL_CUT_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bfor\s+(?:breakfast|lunch|dinner|brunch|drinks|coffee|dessert)\b.*$", re.IGNORECASE),
    re.compile(r"\bfor\s+(?:a|the)?\s*(?:meeting|class|appointment|pickup|take\s*out|takeout|to[- ]go)\b.*$", re.IGNORECASE),
    re.compile(r"\bwith\s+.+$", re.IGNORECASE),
    re.compile(r"\bat\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)\b.*$", re.IGNORECASE),
    re.compile(r"\b(?:tonight|today|tomorrow|this\s+(?:morning|afternoon|evening|weekend|week))\b.*$", re.IGNORECASE),
    re.compile(r"\b(?:please|now|right\s+now|asap)\b.*$", re.IGNORECASE),
]

def _strip_tail_phrases(s: str) -> str:
    out = s
    for pat in TAIL_CUT_PATTERNS:
        out = pat.sub("", out).strip()
    return out


# Words to strip from destination spans
DEST_STRIP = re.compile(
    r"\b(?:to|the|a|an|my|me|please|now|right\s+now|immediately|quickly|fast|quick|back|let'?s|lets|we|us)\b",
    re.IGNORECASE,
)
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
            dest = _strip_tail_phrases(dest)
            dest = re.sub(r"\s+", " ", dest).strip()
            # If the dest still contains a leading verb like 'go', trim it
            dest = re.sub(r"^(go|head|navigate|walk|drive)\s+", "", dest, flags=re.IGNORECASE)
            # Trim trailing polite words
            dest = re.sub(r"\b(?:please|thanks?)$", "", dest, flags=re.IGNORECASE).strip()
            # Drop leading 'to'
            dest = re.sub(r"^to\s+", "", dest, flags=re.IGNORECASE).strip()
            # Drop leading articles
            dest = re.sub(r"^(?:the|a|an)\s+", "", dest, flags=re.IGNORECASE).strip()
            if dest:
                return dest
    return None


logger = logging.getLogger("smart_glass.intent_classifier")


# ---- Embedding model (tiny, zero-shot via prototypes) ---------------------
# Avoid sklearn/scipy: manual cosine via dot on normalized embeddings.
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _SBERT_AVAILABLE = True
except Exception:
    _SBERT_AVAILABLE = False

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # ~90MB


def _resolve_local_minilm_snapshot() -> Optional[Path]:
    """Locate a fully downloaded MiniLM snapshot for offline use."""
    cache_root = Path.home() / ".cache/huggingface/hub"
    repo_dir = cache_root / "models--sentence-transformers--all-MiniLM-L6-v2"
    snap_root = repo_dir / "snapshots"
    if not snap_root.is_dir():
        return None
    ref_main = repo_dir / "refs" / "main"
    if ref_main.is_file():
        snapshot = ref_main.read_text().strip()
        candidate = snap_root / snapshot
        if candidate.is_dir():
            return candidate
    snapshots = sorted(p for p in snap_root.iterdir() if p.is_dir())
    return snapshots[-1] if snapshots else None

# Navigation prototypes (add question forms)
NAV_PROTOTYPES = [
    "navigating <place>"
    "navigate to the destination",
    "go to <place>",
    "head to <place>",
    "take me to <place>",
    "bring me to <place>",
    "drive to <place>",
    "where is the <place>",
    "how do i get to <place>",
]

# Specific scene questions → return 2, <prototype>
DESC_QUESTION_PROTOTYPES = [
    "how much is the item",
    "what is the price",
    "are there any seats nearby",
    "how many people are around",
    "read the sign",
    "read the label",
    "read the text",
]

# General descriptions (no explicit question) → return 2
GENERAL_PROTOTYPES = [
    "describe what's in front of me",
    "what do you see",
    "describe the surroundings",
    "what is around me",
    "what's going on here"
]

# Thresholds for the embedding decision (slightly loosened)
SIM_THRESHOLD = 0.25       # minimal similarity to any prototype
MARGIN = 0.06              # require this gap between top-1 and runner-up


class MiniLMJudge:
    def __init__(self):
        if not _SBERT_AVAILABLE:
            self.model = None
            return
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path: str
        local_snapshot = _resolve_local_minilm_snapshot()
        if local_snapshot:
            model_path = str(local_snapshot)
        else:
            model_path = MODEL_NAME
        try:
            self.model = SentenceTransformer(
                model_path,
                device=device,
                local_files_only=bool(local_snapshot),
            )
        except Exception as exc:
            logger.warning(
                "MiniLM Judge fallback to regex only; SentenceTransformer unavailable: %s",
                exc,
            )
            self.model = None
            return
        self.nav_emb   = self.model.encode(NAV_PROTOTYPES, normalize_embeddings=True)
        self.descq_emb = self.model.encode(DESC_QUESTION_PROTOTYPES, normalize_embeddings=True)
        self.descg_emb = self.model.encode(GENERAL_PROTOTYPES, normalize_embeddings=True)

    @staticmethod
    def _max_sim_and_idx(q: np.ndarray, bank: np.ndarray) -> tuple[float, int]:
        # cosine == dot because embeddings are L2-normalized
        sims = (q @ bank.T).flatten()
        i = int(np.argmax(sims))
        return float(sims[i]), i

    def judge(self, text: str) -> Tuple[int, Optional[str]]:
        """
        1 => navigation  -> payload is destination
        2 => description -> payload is question prototype for specific-Q; None for general
        0 => unclear
        """
        print("[MiniLMJudge] judging ...")
        if not self.model:
            return 0, None
        q = self.model.encode([text], normalize_embeddings=True)

        nav_s,  _   = self._max_sim_and_idx(q, self.nav_emb)
        dq_s,  dq_i = self._max_sim_and_idx(q, self.descq_emb)
        dg_s,  _    = self._max_sim_and_idx(q, self.descg_emb)

        top = max(nav_s, dq_s, dg_s)
        if top < SIM_THRESHOLD:
            return 0, None

        # Determine top class with margin
        scores = sorted([(nav_s, 'nav'), (dq_s, 'dq'), (dg_s, 'dg')], reverse=True)
        (s1, k1), (s2, _k2) = scores[0], scores[1]
        if s1 - s2 <= MARGIN:
            return 0, None

        if k1 == 'nav':
            dest = extract_destination(text) or infer_destination_fallback(text)
            return (1, dest) if dest else (0, None)
        if k1 == 'dq':
            return 2, DESC_QUESTION_PROTOTYPES[dq_i]
        if k1 == 'dg':
            return 2, None
        return 0, None


_GLOBAL_JUDGE: Optional[MiniLMJudge] = None


def get_judge() -> MiniLMJudge:
    global _GLOBAL_JUDGE
    if _GLOBAL_JUDGE is None:
        _GLOBAL_JUDGE = MiniLMJudge()
    return _GLOBAL_JUDGE


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
        if desc_hit:
            pass  # fall through to LM judge
        else:
            dest = infer_destination_fallback(t)
            if dest:
                return 1, dest

    if desc_hit and not nav_hit:
        # Specific question detected via regex -> echo user's question
        return 2, t

    # Secondary: fuzzy judge (embeddings)
    judge = judge or get_judge()
    return judge.judge(t)


# ---- CLI ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Lightweight intent classifier (regex + MiniLM)")
    ap.add_argument("prompt", nargs="?", help="user input to classify")
    ap.add_argument("--interactive", action="store_true", help="REPL mode")
    args = ap.parse_args()

    judge = get_judge()

    def emit(intent: int, payload: Optional[str]):
        if intent == 1 and payload:
            print(f"1, {payload}")
        elif intent == 2 and payload:
            print(f"2, {payload}")
        elif intent == 2 and payload is None:
            print("2")
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

    # final emit
    if intent == 1 and payload:
        print(f"1, {payload}")
    elif intent == 2 and payload:
        print(f"2, {payload}")
    elif intent == 2 and payload is None:
        print("2")
    else:
        print("0")


if __name__ == "__main__":
    main()
