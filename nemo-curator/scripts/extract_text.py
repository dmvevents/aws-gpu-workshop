#!/usr/bin/env python3
"""
Phase 02: Text Extraction and Language Identification.

Reads raw JSONL documents, cleans HTML/markup, normalizes Unicode,
identifies language using fastText, and enriches each record with
metadata (word_count, char_count, language, language_score).

Usage (standalone):
    python extract_text.py \
        --input /data/raw/wikipedia_50k.jsonl \
        --output /data/extracted/extracted.jsonl

Usage (in Docker/K8s):
    Env vars INPUT_PATH and OUTPUT_PATH override CLI defaults.
"""

import argparse
import html
import json
import os
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------

# Matches HTML tags including attributes
_HTML_TAG_RE = re.compile(r"<[^>]+>")

# Matches runs of 3+ whitespace characters
_MULTI_SPACE_RE = re.compile(r"[ \t]{3,}")

# Matches 3+ consecutive newlines
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

# Common boilerplate patterns found in web-scraped text
_BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*cookie\s+(policy|notice|consent)", re.IGNORECASE),
    re.compile(r"^\s*privacy\s+policy", re.IGNORECASE),
    re.compile(r"^\s*terms\s+(of\s+)?(service|use)", re.IGNORECASE),
    re.compile(r"^\s*all\s+rights\s+reserved", re.IGNORECASE),
    re.compile(r"^\s*subscribe\s+to\s+our\s+newsletter", re.IGNORECASE),
]


def strip_html(text: str) -> str:
    """Remove HTML tags and decode HTML entities."""
    text = _HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return text


def normalize_unicode(text: str) -> str:
    """Apply NFKC Unicode normalization and strip control characters."""
    text = unicodedata.normalize("NFKC", text)
    # Remove control characters except newline and tab
    text = "".join(
        ch for ch in text
        if ch in ("\n", "\t") or not unicodedata.category(ch).startswith("C")
    )
    return text


def clean_whitespace(text: str) -> str:
    """Collapse excessive whitespace and normalize newlines."""
    text = _MULTI_SPACE_RE.sub("  ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    text = text.strip()
    return text


def remove_boilerplate_lines(text: str) -> str:
    """Remove lines that match common boilerplate patterns."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        if any(pat.search(line) for pat in _BOILERPLATE_PATTERNS):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def clean_text(text: str) -> str:
    """Full text cleaning pipeline."""
    text = strip_html(text)
    text = normalize_unicode(text)
    text = remove_boilerplate_lines(text)
    text = clean_whitespace(text)
    return text


# ---------------------------------------------------------------------------
# Language identification
# ---------------------------------------------------------------------------

def load_language_model(model_path: str):
    """Load the fastText language-identification model.

    Returns (model, True) on success or (None, False) if unavailable.
    """
    try:
        import fasttext

        # Suppress the warning about loading a model with the old API
        fasttext.FastText.eprint = lambda *args, **kwargs: None
        model = fasttext.load_model(model_path)
        return model, True
    except Exception as exc:
        print(
            f"WARNING: Could not load fastText model at {model_path}: {exc}",
            file=sys.stderr,
        )
        print(
            "  Language identification will be skipped. "
            "Install fasttext-wheel and download lid.176.ftz.",
            file=sys.stderr,
        )
        return None, False


def identify_language(model, text: str):
    """Return (language_code, confidence_score) for the given text.

    Uses the first 500 characters (single line) for speed.
    """
    # fastText expects a single line of text
    sample = text[:500].replace("\n", " ").strip()
    if not sample:
        return "unknown", 0.0

    predictions = model.predict(sample, k=1)
    # predictions = (('__label__en',), array([0.98]))
    label = predictions[0][0].replace("__label__", "")
    score = float(predictions[1][0])
    return label, score


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 02: Extract clean text and identify language."
    )
    parser.add_argument(
        "--input", type=str,
        default=os.environ.get("INPUT_PATH", "/data/raw/wikipedia_50k.jsonl"),
        help="Input JSONL file path.",
    )
    parser.add_argument(
        "--output", type=str,
        default=os.environ.get("OUTPUT_PATH", "/data/extracted/extracted.jsonl"),
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--min-length", type=int, default=50,
        help="Minimum character length after cleaning (default: 50).",
    )
    parser.add_argument(
        "--lang-threshold", type=float, default=0.7,
        help="Minimum language confidence to keep a document (default: 0.7).",
    )
    parser.add_argument(
        "--target-lang", type=str, default="en",
        help="Target language ISO code (default: en). Set to 'any' to keep all.",
    )
    parser.add_argument(
        "--fasttext-model", type=str,
        default=os.environ.get("FASTTEXT_MODEL", "/models/lid.176.ftz"),
        help="Path to fastText lid model (default: /models/lid.176.ftz).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 02: Text Extraction and Language ID")
    print("=" * 60)
    print(f"  Input:          {args.input}")
    print(f"  Output:         {args.output}")
    print(f"  Min length:     {args.min_length} chars")
    print(f"  Target lang:    {args.target_lang}")
    print(f"  Lang threshold: {args.lang_threshold}")
    print("=" * 60)

    # Load language model
    lang_model, lang_available = load_language_model(args.fasttext_model)

    t_start = time.time()
    stats = {
        "input_count": 0,
        "output_count": 0,
        "skipped_empty": 0,
        "skipped_short": 0,
        "skipped_lang": 0,
        "skipped_low_confidence": 0,
    }
    now_iso = datetime.now(timezone.utc).isoformat()

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:

        for line_no, line in enumerate(fin, 1):
            stats["input_count"] += 1
            line = line.strip()
            if not line:
                stats["skipped_empty"] += 1
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["skipped_empty"] += 1
                continue

            raw_text = record.get("text", "")
            if not raw_text:
                stats["skipped_empty"] += 1
                continue

            # Clean text
            cleaned = clean_text(raw_text)

            # Length filter
            if len(cleaned) < args.min_length:
                stats["skipped_short"] += 1
                continue

            # Language identification
            lang = "unknown"
            lang_score = 0.0
            if lang_available:
                lang, lang_score = identify_language(lang_model, cleaned)

                # Filter by target language
                if args.target_lang != "any" and lang != args.target_lang:
                    stats["skipped_lang"] += 1
                    continue

                # Filter by confidence
                if lang_score < args.lang_threshold:
                    stats["skipped_low_confidence"] += 1
                    continue

            # Compute metadata
            words = cleaned.split()
            word_count = len(words)
            char_count = len(cleaned)

            # Build output record
            output_record = {
                "id": record.get("id", str(line_no)),
                "title": record.get("title", ""),
                "text": cleaned,
                "url": record.get("url", ""),
                "language": lang,
                "language_score": round(lang_score, 4),
                "word_count": word_count,
                "char_count": char_count,
                "extraction_timestamp": now_iso,
            }

            fout.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            stats["output_count"] += 1

            # Progress
            if stats["input_count"] % 5000 == 0:
                elapsed = time.time() - t_start
                rate = stats["input_count"] / elapsed if elapsed > 0 else 0
                print(
                    f"  [{stats['input_count']:>7,}] "
                    f"kept={stats['output_count']:,}  "
                    f"rate={rate:,.0f} docs/s"
                )

    # Final report
    elapsed = time.time() - t_start
    print()
    print("=" * 60)
    print("Extraction Complete")
    print("=" * 60)
    print(f"  Input documents:     {stats['input_count']:,}")
    print(f"  Output documents:    {stats['output_count']:,}")
    print(f"  Skipped (empty):     {stats['skipped_empty']:,}")
    print(f"  Skipped (too short): {stats['skipped_short']:,}")
    print(f"  Skipped (wrong lang):{stats['skipped_lang']:,}")
    print(f"  Skipped (low conf):  {stats['skipped_low_confidence']:,}")
    print(f"  Elapsed:             {elapsed:,.1f}s")
    print(f"  Output file:         {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
