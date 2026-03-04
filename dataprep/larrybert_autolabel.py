#!/usr/bin/env python3
"""
larrybert_autolabel.py

More rigorous heuristic auto-labeler for LarryBERT claim sentences.

Key fixes vs naive labelers:
- FACTCHECK is NOT "any yes/no question". It's reserved for explicit myth/claim-check framing.
- "this year/this season/right now" are TREND/time-window cues (not FORECAST by themselves).
- Adds a question-start fallback router to prevent everything collapsing into FACTCHECK.
- Uses weighted scoring within each label + priority tie-break.

Outputs JSONL:
{"text": "...", "label": "COMPARE|LEADERBOARD|TREND|FORECAST|FACTCHECK|EXPLAIN", "meta": {...}}

Usage:
  py -m larrybert_autolabel --infile data/txt_files/train.txt --outfile data/json_files/intent_train_autolabeled.jsonl --dedupe --stats --skip_garbage
  py -m larrybert_autolabel --infile data/txt_files/test.txt --outfile data/json_files/intent_test_autolabeled.jsonl --dedupe --stats --skip_garbage
  py -m larrybert_autolabel --infile data/txt_files/val.txt --outfile data/json_files/intent_val_autolabeled.jsonl --dedupe --stats --skip_garbage
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Pattern, Tuple

# Label set (FACTCHECK split + ATTRIBUTE split)
LABELS = [
    "COMPARE",
    "LEADERBOARD",
    "TREND",
    "FORECAST",
    "EXPLAIN",
    "MYTHCHECK",
    "ROLE_OPPORTUNITY",
    "METRIC_CHECK",
    "TRAIT_CHECK",
]

# -----------------------------
# Pattern utilities
# -----------------------------
def rx(p: str) -> Pattern:
    return re.compile(p, re.IGNORECASE)

def normalize_line(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"([?!\.])\1{1,}$", r"\1", s)
    return s


# -----------------------------
# Garbage detector (LESS aggressive)
# -----------------------------
QUESTION_START = rx(r"^\s*(is|are|was|were|do|does|did|has|have|can|should|would)\b")
WEIRD_ENCODING = rx(r"[�]")             # common broken-char marker
ONLY_SYMBOLS = rx(r"^[^A-Za-z0-9]+$")   # just punctuation/symbols

# Words that often indicate an incomplete fragment at the end
FRAGMENT_END = rx(r"\b(shooting|minutes?|rotation|defender|defense|scoring|rebounding|assists?)\b$")

# Minimal NBA/stat/role vocabulary (small + safe)
SIGNAL_WORDS = rx(
    r"\b("
    r"3|three|threes|three[-\s]*point|midrange|rim|paint|perimeter|"
    r"points?|reb(ounds?)?|assist(s)?|block(s)?|steal(s)?|"
    r"ts%|efg%|fg%|3p%|ft%|true\s+shooting|"
    r"rate|per\s+game|per\s+36|"
    r"defen(se|sive|der)|offen(se|sive)|"
    r"minutes?|rotation|starter|bench|usage|role|opportunit(y|ies)|"
    r"efficient|efficiency|volume|attempts?|accuracy|"
    r"improv(e|ed|ing)|declin(e|ed|ing)|regress(ed|ing)|since|lately|recent"
    r")\b"
)

def is_garbage(s: str) -> Tuple[bool, str]:
    """
    Conservative filter: skip only clearly broken lines.
    Keeps short questions because they're valid for intent classification.
    """
    t = s.strip()
    if not t:
        return True, "empty"
    if WEIRD_ENCODING.search(t):
        return True, "bad_encoding_char"
    if ONLY_SYMBOLS.match(t):
        return True, "symbols_only"

    words = t.split()
    if len(words) < 2:
        return True, "too_few_words"

    # If it looks like a question fragment like "Is Jordan shooting."
    # keep it ONLY if there's at least one signal word elsewhere.
    if QUESTION_START.search(t):
        t_no_punct = re.sub(r"[?.!]+$", "", t).strip()
        if len(t_no_punct.split()) <= 4 and FRAGMENT_END.search(t_no_punct):
            if not SIGNAL_WORDS.search(t_no_punct):
                return True, "dangling_fragment"

    return False, ""


# -----------------------------
# Scored rule sets
# -----------------------------
@dataclass(frozen=True)
class ScoredRule:
    label: str
    patterns: Tuple[Pattern, ...]
    weight: int
    name: str


def build_scored_rules() -> List[ScoredRule]:
    rules: List[ScoredRule] = []

    # FORECAST
    rules += [
        ScoredRule("FORECAST", (rx(r"\bwill\b"),), 3, "will"),
        ScoredRule("FORECAST", (rx(r"\bon\s*pace\b"), rx(r"\bpace\s+to\b")), 4, "pace"),
        ScoredRule("FORECAST", (rx(r"\bproject(ed|ion)\b"), rx(r"\bprojection\b")), 3, "projection"),
        ScoredRule("FORECAST", (rx(r"\bby\s+the\s+end\s+of\b"), rx(r"\bby\s+season'?s?\s+end\b")), 4, "by_end"),
        ScoredRule("FORECAST", (rx(r"\bbreak\s+(the\s+)?(all[-\s]*time\s+)?record\b"),), 5, "break_record"),
        ScoredRule("FORECAST", (rx(r"\bcareer\b.*\b(record|total|milestone)\b"),), 3, "career_record"),
        ScoredRule("FORECAST", (rx(r"\breach\s+\d+\b"), rx(r"\bfinish\s+with\s+\d+\b")), 3, "reach_finish_number"),
        ScoredRule("FORECAST", (rx(r"\bnext\s+season\b"), rx(r"\bin\s+\d+\s+years?\b")), 2, "future_window"),
        ScoredRule("FORECAST", (rx(r"\bdevelop\s+into\b"), rx(r"\bbecome\b"), rx(r"\beventually\b")), 2, "develop_become_eventually"),
    ]

    # EXPLAIN
    rules += [
        ScoredRule("EXPLAIN", (rx(r"\bwhy\b"), rx(r"\bhow\s+come\b")), 5, "why_howcome"),
        ScoredRule("EXPLAIN", (rx(r"\bdue\s+to\b"), rx(r"\bbecause\b"), rx(r"\bcaused\s+by\b")), 4, "causal"),
        ScoredRule("EXPLAIN", (rx(r"\bexplain\b"), rx(r"\breason\b"), rx(r"\bwhat\s+caused\b")), 4, "explain_reason"),
        ScoredRule("EXPLAIN", (rx(r"\bimpact\s+of\b"), rx(r"\beffect\s+of\b"), rx(r"\bdriven\s+by\b")), 3, "impact_effect"),
    ]

    # COMPARE
    rules += [
        ScoredRule("COMPARE", (rx(r"\bbetter\s+than\b"), rx(r"\bworse\s+than\b")), 5, "better_worse"),
        ScoredRule("COMPARE", (rx(r"\bmore\s+\w+\s+than\b"), rx(r"\bless\s+\w+\s+than\b")), 3, "more_less"),
        ScoredRule("COMPARE", (rx(r"\bhigher\s+than\b"), rx(r"\blower\s+than\b")), 4, "higher_lower"),
        ScoredRule("COMPARE", (rx(r"\bvs\.?\b"), rx(r"\bversus\b"), rx(r"\bcompared\s+to\b")), 4, "vs_versus"),
        ScoredRule("COMPARE", (rx(r"\bthan\b"),), 2, "than_loose"),
        ScoredRule("COMPARE", (rx(r"\bcompare\b"), rx(r"\boutperform(s|ed)?\b")), 3, "compare_outperform"),
    ]

    # LEADERBOARD
    rules += [
        ScoredRule("LEADERBOARD", (rx(r"\blead(s|ing)?\s+the\s+league\b"), rx(r"\bin\s+the\s+league\b")), 4, "league_leading"),
        ScoredRule("LEADERBOARD", (rx(r"\bmost\b"), rx(r"\bbest\b"), rx(r"\bworst\b")), 4, "most_best_worst"),
        ScoredRule("LEADERBOARD", (rx(r"\btop\s+\d+\b"), rx(r"\brank(s|ed|ing)?\b")), 4, "top_rank"),
        ScoredRule("LEADERBOARD", (rx(r"#\s*1\b"), rx(r"\bno\.\s*1\b"), rx(r"\bnumber\s+one\b")), 5, "number_one"),
        ScoredRule("LEADERBOARD", (rx(r"\bhighest\b"), rx(r"\blowest\b"), rx(r"\bamong\b")), 3, "highest_lowest_among"),
    ]

    # TREND
    rules += [
        ScoredRule("TREND", (rx(r"\blately\b"), rx(r"\brecent(ly)?\b")), 4, "lately_recent"),
        ScoredRule("TREND", (rx(r"\blast\s+\d+\s+games?\b"), rx(r"\bover\s+the\s+last\s+\d+\s+games?\b")), 5, "last_n_games"),
        ScoredRule("TREND", (rx(r"\bsince\b"),), 3, "since"),
        ScoredRule("TREND", (rx(r"\bimprov(e|ed|ing|ement)\b"), rx(r"\bdeclin(e|ed|ing)\b"), rx(r"\bregress(ed|ing|ion)?\b")), 4, "improve_decline_regress"),
        ScoredRule("TREND", (rx(r"\bslump\b"), rx(r"\bheating\s+up\b"), rx(r"\bcooled\s+off\b")), 4, "slump_heat_cool"),
        ScoredRule("TREND", (rx(r"\bstill\b"), rx(r"\banymore\b"), rx(r"\bright\s+now\b")), 3, "still_anymore_now"),
        ScoredRule("TREND", (rx(r"\bthis\s+season\b"), rx(r"\bthis\s+year\b"), rx(r"\blast\s+season\b")), 2, "season_window"),
        ScoredRule("TREND", (rx(r"\bcompared\s+to\s+last\s+season\b"), rx(r"\bvs\s+last\s+season\b")), 4, "vs_last_season"),
    ]

    # MYTHCHECK
    rules += [
        ScoredRule("MYTHCHECK", (rx(r"\bis\s+it\s+(true|false)\b"), rx(r"\btrue\s+or\s+false\b")), 5, "true_false"),
        ScoredRule("MYTHCHECK", (rx(r"\bmyth\b"), rx(r"\bdebunk\b")), 5, "myth_debunk"),
        ScoredRule("MYTHCHECK", (rx(r"\bpeople\s+(say|think|claim)\b"), rx(r"\bnarrative\b"), rx(r"\bstigma\b")), 4, "people_say_narrative"),
        ScoredRule("MYTHCHECK", (rx(r"\bcap\b"), rx(r"\bfacts?\b")), 3, "cap_facts"),
        ScoredRule("MYTHCHECK", (rx(r"\breally\b"), rx(r"\bactually\b")), 1, "really_actually_weak"),
    ]

    # ROLE_OPPORTUNITY
    rules += [
        ScoredRule("ROLE_OPPORTUNITY", (rx(r"\bminutes?\b"),), 4, "minutes"),
        ScoredRule("ROLE_OPPORTUNITY", (rx(r"\brotation\b"), rx(r"\bsee\s+the\s+floor\b")), 5, "rotation_floor"),
        ScoredRule("ROLE_OPPORTUNITY", (rx(r"\bstarting\b"), rx(r"\bstarter\b"), rx(r"\bbench\b"), rx(r"\bsixth\s+man\b")), 4, "starting_bench"),
        ScoredRule("ROLE_OPPORTUNITY", (rx(r"\bopportunit(y|ies)\b"), rx(r"\bconsistent\s+opportunit(y|ies)\b")), 5, "opportunity"),
        ScoredRule("ROLE_OPPORTUNITY", (rx(r"\bplaying\s+time\b"), rx(r"\bappear(ing)?\s+in\s+games\b")), 5, "playing_time_appearing"),
        ScoredRule("ROLE_OPPORTUNITY", (rx(r"\busage\b"), rx(r"\blow[-\s]*usage\b"), rx(r"\bhigh[-\s]*usage\b")), 4, "usage"),
        ScoredRule("ROLE_OPPORTUNITY", (rx(r"\brole\b"), rx(r"\brole\s+player\b"), rx(r"\bbackup\b")), 3, "role_backup"),
        ScoredRule("ROLE_OPPORTUNITY", (rx(r"\bhaven't\s+gotten\s+consistent\b"), rx(r"\bhasn't\s+gotten\s+consistent\b")), 5, "hasnt_gotten_consistent"),
        ScoredRule("ROLE_OPPORTUNITY", (rx(r"\bhasn't\s+broken\s+through\b"), rx(r"\bhasn'?t\s+cracked\b")), 5, "hasnt_broken_through"),
        ScoredRule("ROLE_OPPORTUNITY", (rx(r"\brarely\s+shoots?\b"),), 3, "rarely_shoots"),
        ScoredRule("ROLE_OPPORTUNITY", (rx(r"\breduced\s+(role|minutes|usage)\b"), rx(r"\bprimary\s+offensive\s+role\b")), 3, "reduced_role"),
    ]

    # METRIC_CHECK
    rules += [
        ScoredRule("METRIC_CHECK", (rx(r"\b(ts%|efg%|fg%|3p%|ft%|true\s+shooting|effective\s+field\s+goal)\b"),), 6, "efficiency_pct"),
        ScoredRule("METRIC_CHECK", (rx(r"\b(points?|ppg|reb(ounds?)?|rpg|ast(s)?|apg|blk(s)?|bpg|stl(s)?|spg|tov|turnovers?)\b"),), 4, "boxstats_words"),
        ScoredRule("METRIC_CHECK", (rx(r"\brate(s)?\b"), rx(r"\bper\s+game\b"), rx(r"\bper[-\s]*minute\b"), rx(r"\bper\s+36\b")), 4, "rate_per"),
        ScoredRule("METRIC_CHECK", (rx(r"\bthree[-\s]*point\b"), rx(r"\b3[-\s]*poin(t|ters?)\b"), rx(r"\bthrees?\b")), 3, "three_point"),
        ScoredRule("METRIC_CHECK", (rx(r"\bvolume\b"), rx(r"\battempts?\b"), rx(r"\baccuracy\b"), rx(r"\bpercentage\b")), 3, "volume_attempts_accuracy"),
        ScoredRule("METRIC_CHECK", (rx(r"\b(defensive\s+rating|offensive\s+rating|net\s+rating)\b"),), 4, "ratings"),
        ScoredRule("METRIC_CHECK", (rx(r"\bsteal\s+rate\b"), rx(r"\bblock\s+rate\b"), rx(r"\busage\s+rate\b")), 4, "rates_named"),
        ScoredRule("METRIC_CHECK", (rx(r"\bopen\s+three\b"), rx(r"\bwide\s+open\b")), 2, "shot_quality_proxy"),
    ]

    # TRAIT_CHECK
    rules += [
        ScoredRule("TRAIT_CHECK", (rx(r"\bprotect(ing)?\s+(the\s+)?rim\b"), rx(r"\brim\s+protector\b")), 5, "rim_protection"),
        ScoredRule("TRAIT_CHECK", (rx(r"\bperimeter\s+defen(ce|se|der)\b"), rx(r"\bon[-\s]*ball\b")), 4, "perimeter_onball_defense"),
        ScoredRule("TRAIT_CHECK", (rx(r"\bversatility\b"), rx(r"\btwo[-\s]*way\b")), 3, "versatility_two_way"),
        ScoredRule("TRAIT_CHECK", (rx(r"\bball[-\s]*handling\b"), rx(r"\bhandle\b")), 4, "ball_handling"),
        ScoredRule("TRAIT_CHECK", (rx(r"\bclutch\b"),), 2, "clutch_trait"),
        ScoredRule("TRAIT_CHECK", (rx(r"\binconsisten(t|cy)\b"),), 2, "inconsistency_trait"),
        ScoredRule("TRAIT_CHECK", (rx(r"\bdefender\b"), rx(r"\bdefensive\b")), 2, "defender_general"),
        # generic adjectives are weak; keep low weight
        ScoredRule("TRAIT_CHECK", (rx(r"\bsolid\b"), rx(r"\beffective\b"), rx(r"\bgood\b"), rx(r"\belite\b")), 1, "generic_trait_words"),
    ]

    return rules


SCORED_RULES = build_scored_rules()

# Priority order tie-break
PRIORITY = [
    "FORECAST",
    "EXPLAIN",
    "COMPARE",
    "LEADERBOARD",
    "TREND",
    "MYTHCHECK",
    "ROLE_OPPORTUNITY",
    "METRIC_CHECK",
    "TRAIT_CHECK",
]
PRIORITY_RANK = {lab: i for i, lab in enumerate(PRIORITY)}


def score_sentence(s: str) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
    scores: Dict[str, int] = {lab: 0 for lab in PRIORITY}
    hits: Dict[str, List[str]] = {lab: [] for lab in PRIORITY}

    for rule in SCORED_RULES:
        if any(p.search(s) for p in rule.patterns):
            scores[rule.label] += rule.weight
            hits[rule.label].append(rule.name)

    return scores, hits


def choose_label(scores: Dict[str, int]) -> Tuple[str, int]:
    best_label = None
    best_score = -1

    for lab in PRIORITY:
        sc = scores.get(lab, 0)
        if sc > best_score:
            best_score = sc
            best_label = lab
        elif sc == best_score and best_label is not None:
            if PRIORITY_RANK[lab] < PRIORITY_RANK[best_label]:
                best_label = lab

    assert best_label is not None
    return best_label, best_score


def fallback_route_question(s: str) -> str:
    """
    If nothing scored, but it's a yes/no style question, route based on light cues.
    Default should be METRIC_CHECK.
    """
    if not QUESTION_START.search(s):
        return "METRIC_CHECK"

    if re.search(r"\b(record|milestone|on pace|pace to|projected|projection)\b", s, re.I):
        return "FORECAST"
    if re.search(r"\b(why|how come|because|due to|caused by|reason|explain)\b", s, re.I):
        return "EXPLAIN"
    if re.search(r"\b(vs|versus|than|compared to)\b", s, re.I):
        return "COMPARE"
    if re.search(r"\b(best|worst|most|top|rank|leading|highest|lowest|among|in the league)\b", s, re.I):
        return "LEADERBOARD"
    if re.search(r"\b(lately|recent|since|last\s+\d+|this\s+season|this\s+year|right now|still|anymore|slump|heating up|cooled off)\b", s, re.I):
        return "TREND"
    if re.search(r"\b(minutes?|rotation|opportunity|playing time|starter|starting|bench|backup|low[-\s]*usage|usage|role)\b", s, re.I):
        return "ROLE_OPPORTUNITY"
    if re.search(r"\b(is it true|is it false|true or false|myth|debunk|people say|narrative|stigma|cap)\b", s, re.I):
        return "MYTHCHECK"
    if re.search(r"\b(rim|protect(ing)?|perimeter|on[-\s]*ball|versatility|two[-\s]*way|ball[-\s]*handling|handle)\b", s, re.I):
        return "TRAIT_CHECK"

    return "METRIC_CHECK"


def label_sentence(s: str) -> Tuple[str, Dict]:
    scores, hits = score_sentence(s)
    label, best_score = choose_label(scores)

    if best_score == 0:
        label = fallback_route_question(s)

    meta = {
        "scores": scores,
        "hits": {k: v for k, v in hits.items() if v},
    }
    return label, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True, help="Input .txt file (1 sentence per line)")
    ap.add_argument("--outfile", required=True, help="Output .jsonl file")
    ap.add_argument("--dedupe", action="store_true", help="Remove exact duplicates (after normalization)")
    ap.add_argument("--stats", action="store_true", help="Print label distribution")
    ap.add_argument("--include_meta", action="store_true", help="Include meta (scores/hits) in JSONL output")
    ap.add_argument("--skip_garbage", action="store_true", help="Skip low-quality/garbage lines")
    args = ap.parse_args()

    in_path = Path(args.infile)
    out_path = Path(args.outfile)

    raw_lines = in_path.read_text(encoding="utf-8").splitlines()
    cleaned = [normalize_line(x) for x in raw_lines]
    cleaned = [x for x in cleaned if x]

    if args.dedupe:
        seen = set()
        deduped = []
        for x in cleaned:
            if x not in seen:
                seen.add(x)
                deduped.append(x)
        cleaned = deduped

    counts = Counter()
    garbage_counts = Counter()

    with out_path.open("w", encoding="utf-8") as f:
        for s in cleaned:
            if args.skip_garbage:
                bad, reason = is_garbage(s)
                if bad:
                    garbage_counts[reason] += 1
                    continue

            lab, meta = label_sentence(s)
            counts[lab] += 1
            obj = {"text": s, "label": lab}
            if args.include_meta:
                obj["meta"] = meta
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    total = sum(counts.values())
    print(f"Wrote {total} labeled rows to {out_path}")

    if args.stats:
        print("Label distribution:")
        for lab in LABELS:
            n = counts.get(lab, 0)
            pct = (n / total * 100) if total else 0
            print(f"  {lab:16s} {n:5d}  ({pct:5.1f}%)")

        if args.skip_garbage and garbage_counts:
            skipped = sum(garbage_counts.values())
            print(f"\nSkipped {skipped} garbage lines:")
            for reason, n in garbage_counts.most_common():
                print(f"  {reason:28s} {n}")

        if total:
            top_lab, top_n = counts.most_common(1)[0]
            if top_n / total > 0.70:
                print(
                    f"\nWARNING: '{top_lab}' is {100*top_n/total:.1f}% of labels. "
                    "The corpus may be skewed or the rules may still be too weak for other classes."
                )


if __name__ == "__main__":
    main()