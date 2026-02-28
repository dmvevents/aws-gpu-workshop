#!/usr/bin/env python3
"""
Phase 05: Domain Classification and Quality Scoring.

Classifies each document into one of 8 technology domains using
keyword matching, and computes a heuristic quality score based on
text features (sentence count, vocabulary diversity, avg sentence
length).

This is a lightweight, dependency-free alternative to NeMo Curator's
GPU-based neural classifiers. It is suitable for workshop-scale data
and can be replaced with a fine-tuned classifier for production use.

Usage (standalone):
    python classify_domains.py \
        --input /data/filtered/filtered.jsonl \
        --output /data/classified/classified.jsonl

    # Filter out documents below a quality threshold:
    python classify_domains.py --input in.jsonl --output out.jsonl --min-quality-score 0.4
"""

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# Domain keywords
# ---------------------------------------------------------------------------
# Each domain has a set of keywords. A document is assigned to the domain
# with the highest keyword hit count. Ties are broken by the order below.

DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "artificial_intelligence": [
        "machine learning", "deep learning", "neural network", "nlp",
        "natural language processing", "computer vision", "reinforcement learning",
        "transformer", "language model", "llm", "gpt", "generative ai",
        "artificial intelligence", " ai ", "training data", "fine-tun",
        "classification", "embeddings", "diffusion model", "chatbot",
    ],
    "cybersecurity": [
        "cybersecurity", "security", "vulnerability", "encryption",
        "malware", "ransomware", "phishing", "firewall", "zero-day",
        "authentication", "data breach", "penetration test", "infosec",
        "cyber attack", "threat intelligence", "zero trust", "siem",
    ],
    "cloud_computing": [
        "cloud computing", "aws", "azure", "google cloud", "gcp",
        "kubernetes", "docker", "serverless", "lambda function",
        "cloud-native", "microservice", "container orchestrat",
        "terraform", "infrastructure as code", "saas", "paas", "iaas",
        "load balancer", "auto-scaling", "cloud migration",
    ],
    "software_engineering": [
        "software engineer", "programming", "code review", "git",
        "ci/cd", "continuous integration", "agile", "scrum",
        "test-driven", "refactor", "api design", "design pattern",
        "open source", "typescript", "python", "rust", "golang",
        "devops", "debugging", "software architecture",
    ],
    "hardware": [
        "semiconductor", "chip", "processor", "cpu", "gpu", "fpga",
        "quantum comput", "5g", "network", "iot", "internet of things",
        "robotics", "sensor", "arm architecture", "risc-v", "silicon",
        "motherboard", "embedded system", "wearable", "lidar",
    ],
    "data_science": [
        "data science", "data analysis", "data engineer", "analytics",
        "big data", "data pipeline", "etl", "data warehouse",
        "visualization", "pandas", "spark", "hadoop", "sql",
        "feature engineering", "a/b test", "statistical",
        "machine learning pipeline", "data lake",
    ],
    "business_tech": [
        "startup", "funding", "acquisition", "ipo", "venture capital",
        "digital transformation", "enterprise", "saas", "b2b",
        "revenue", "market share", "ceo", "cto", "product launch",
        "tech industry", "valuation", "series a", "series b",
    ],
    "other": [],  # Catch-all -- assigned when no keywords match
}

# Pre-compile domain patterns for speed
_DOMAIN_PATTERNS: dict[str, list[re.Pattern]] = {}
for domain, keywords in DOMAIN_KEYWORDS.items():
    _DOMAIN_PATTERNS[domain] = [
        re.compile(re.escape(kw), re.IGNORECASE) for kw in keywords
    ]


# ---------------------------------------------------------------------------
# Domain classification
# ---------------------------------------------------------------------------

def classify_domain(text: str) -> tuple[str, float]:
    """Return (domain, confidence) based on keyword matching.

    Confidence is normalized: keyword_hits / max_possible_hits.
    """
    text_lower = text.lower()
    scores: dict[str, int] = {}

    for domain, patterns in _DOMAIN_PATTERNS.items():
        if domain == "other":
            continue
        hits = sum(1 for pat in patterns if pat.search(text_lower))
        if hits > 0:
            scores[domain] = hits

    if not scores:
        return "other", 0.0

    best_domain = max(scores, key=scores.get)
    max_keywords = len(DOMAIN_KEYWORDS[best_domain])
    confidence = scores[best_domain] / max_keywords if max_keywords > 0 else 0.0
    return best_domain, round(min(confidence, 1.0), 4)


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------

_SENTENCE_END = re.compile(r"[.!?]+")


def compute_quality_score(text: str) -> float:
    """Heuristic quality score from 0.0 to 1.0 based on text features.

    Components (equally weighted):
      - sentence_score:   Rewards docs with >= 5 sentences (capped at 1.0)
      - length_score:     Rewards moderate length (500-5000 words)
      - vocabulary_score: Type-token ratio (vocabulary diversity)
      - structure_score:  Rewards multi-paragraph documents
    """
    words = text.split()
    word_count = len(words)

    if word_count == 0:
        return 0.0

    # Sentence count
    sentences = [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]
    sentence_count = max(len(sentences), 1)
    sentence_score = min(sentence_count / 5.0, 1.0)

    # Length score: peak at 500-5000 words
    if word_count < 50:
        length_score = word_count / 50.0
    elif word_count <= 5000:
        length_score = 1.0
    else:
        # Gentle penalty for very long documents
        length_score = max(0.5, 1.0 - (word_count - 5000) / 50000)

    # Vocabulary diversity (type-token ratio, sampled for long texts)
    sample = words[:2000]  # Cap to avoid bias from very long docs
    unique_words = len(set(w.lower() for w in sample))
    vocabulary_score = unique_words / len(sample)

    # Structure: multi-paragraph bonus
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    para_count = len(paragraphs)
    structure_score = min(para_count / 3.0, 1.0)

    # Weighted average
    score = (
        0.25 * sentence_score
        + 0.25 * length_score
        + 0.30 * vocabulary_score
        + 0.20 * structure_score
    )
    return round(score, 4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 05: Domain classification and quality scoring."
    )
    parser.add_argument(
        "--input", type=str,
        default=os.environ.get("INPUT_PATH", "/data/filtered/filtered.jsonl"),
        help="Input JSONL file path.",
    )
    parser.add_argument(
        "--output", type=str,
        default=os.environ.get("OUTPUT_PATH", "/data/classified/classified.jsonl"),
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--min-quality-score", type=float, default=0.0,
        help="Minimum quality score to keep a document (default: 0.0 = keep all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 05: Domain Classification and Quality Scoring")
    print("=" * 60)
    print(f"  Input:              {args.input}")
    print(f"  Output:             {args.output}")
    print(f"  Min quality score:  {args.min_quality_score}")
    print(f"  Domains:            {len(DOMAIN_KEYWORDS)}")
    print("=" * 60)

    t_start = time.time()
    domain_counts: Counter = Counter()
    quality_sum = 0.0
    input_count = 0
    output_count = 0
    filtered_quality = 0

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            input_count += 1
            text = record.get("text", "")

            # Classify domain
            domain, domain_confidence = classify_domain(text)

            # Score quality
            quality_score = compute_quality_score(text)

            # Filter by quality
            if quality_score < args.min_quality_score:
                filtered_quality += 1
                continue

            # Enrich record
            record["domain"] = domain
            record["domain_confidence"] = domain_confidence
            record["quality_score"] = quality_score

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_count += 1

            domain_counts[domain] += 1
            quality_sum += quality_score

            # Progress
            if input_count % 5000 == 0:
                print(f"  [{input_count:>7,}] classified={output_count:,}")

    # Report
    elapsed = time.time() - t_start
    avg_quality = (quality_sum / output_count) if output_count > 0 else 0.0

    print()
    print("=" * 60)
    print("Classification Complete")
    print("=" * 60)
    print(f"  Input documents:     {input_count:,}")
    print(f"  Output documents:    {output_count:,}")
    print(f"  Filtered (quality):  {filtered_quality:,}")
    print(f"  Avg quality score:   {avg_quality:.3f}")
    print()
    print("  Domain distribution:")
    total = sum(domain_counts.values())
    for domain, count in domain_counts.most_common():
        pct = (count / total * 100) if total > 0 else 0
        bar = "#" * max(1, int(pct / 2))
        print(f"    {domain:<28s}  {count:>6,} ({pct:5.1f}%)  {bar}")
    print()
    print(f"  Elapsed:             {elapsed:,.1f}s")
    print(f"  Output file:         {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
