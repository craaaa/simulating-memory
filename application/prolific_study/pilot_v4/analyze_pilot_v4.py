#!/usr/bin/env python3
"""
Pilot v4 Qualtrics data analysis.
Birds: Ashen Skreel (Plumbea vorax) and Copperhook Finch (Fringilla rufa).

Qualtrics CSV format:
  row 0 = column labels (field names)
  row 1 = display labels
  row 2 = importIds
  data starts at row 3

Multi-select questions: Qualtrics concatenates selected option texts with a comma.
Because some option texts themselves contain commas, we enumerate the full option
universe per question and use greedy left-to-right matching to parse selections.
"""

import csv
import statistics
from collections import Counter, defaultdict

DATA_FILE = (
    "/Users/cl5625/simulating-memory/.claude/worktrees/compare-task/"
    "application/prolific_study/pilot_v4/"
    "results.csv"
)

# ── Option universes (order matters for greedy parsing) ──────────────────────
# Each list is ordered longest-first / most-specific-first so that greedy
# prefix matching does not accidentally consume part of a longer option.
#
# Source: Qualtrics QSF + unique singleton values observed in the export.
# NOTE: The live survey used different option texts than pilot_v4.qsf.
# pilot_v4.qsf (QP04) showed diet-matching options; the actual data contains
# Latin-name-matching options.  All option texts below are derived from the
# data export, verified against control.md.

OPTION_UNIVERSES = {
    # QP01 — Ashen Skreel plumage (1 correct: option A)
    "QP01": [
        "Mostly gray, with a lighter patch at the throat",   # A — CORRECT
        "Rust orange, with a lighter patch at the throat",   # B — false (Copperhook colour)
        "Faint streaking along the back",                    # C — false
        "A reddish wash across the breast",                  # D — false
        "None of the above",                                 # E — false
    ],
    # QP02 — Copperhook Finch diet (3 correct: Fruit, Seeds, Grubs)
    "QP02": [
        "Fruit",        # A — CORRECT
        "Seeds",        # B — CORRECT
        "Grubs",        # C — CORRECT
        "None of the above",  # D — false (there are correct answers)
    ],
    # QP03 — differences between birds (2 correct: A and B)
    "QP03": [
        "The ashen skreel eats carrion, while the Copperhook finch eats seeds",    # A — CORRECT
        "The ashen skreel is mostly gray, whereas the Copperhook finch is mostly orange",  # B — CORRECT
        "The Copperhook finch eats seeds, unlike the fruit-eating ashen skreel",   # C — false (Skreel eats carrion, not fruit)
        "The ashen skreel has black wingtips, while the Copperhook finch has a pale throat",  # D — false (reversed)
        "None of the above",  # E — false
    ],
    # QP04 — Latin name matching (correct = "None of the above")
    # Ashen Skreel = Plumbea vorax; Copperhook Finch = Fringilla rufa.
    # None of the four distractor pairings are correct, so answer = E.
    "QP04": [
        "Ashen skreel -- Fringilla rufa",          # A — false (wrong species)
        "Ashen skreel -- Plumbea rufa",            # B — false (wrong epithet: vorax, not rufa)
        "Copperhook finch -- Fringilla vorax",     # C — false (wrong epithet: rufa, not vorax)
        "Copperhook finch -- Plumbea rufa",        # D — false (wrong genus)
        "None of the above",                       # E — CORRECT
    ],
    # QP05 — Copperhook Finch plumage (2 correct: A and B)
    "QP05": [
        "A rust-orange body",        # A — CORRECT
        "Dark tips to the wings",    # B — CORRECT
        "A bright yellow crest",     # C — false
        "Bluish bars across the tail",  # D — false
        # Note: no "None of the above" option seen in data for QP05
    ],
}

# Correct option sets per question (use exact strings from OPTION_UNIVERSES)
CORRECT_ANSWERS = {
    "QP01": {"Mostly gray, with a lighter patch at the throat"},
    "QP02": {"Fruit", "Seeds", "Grubs"},
    "QP03": {
        "The ashen skreel eats carrion, while the Copperhook finch eats seeds",
        "The ashen skreel is mostly gray, whereas the Copperhook finch is mostly orange",
    },
    "QP04": {"None of the above"},
    "QP05": {"A rust-orange body", "Dark tips to the wings"},
}

QUESTIONS = ["QP01", "QP02", "QP03", "QP04", "QP05"]


def parse_multiselect(raw: str, universe: list[str]) -> set[str]:
    """
    Parse a Qualtrics multi-select field into the set of selected options.

    Qualtrics concatenates selected display texts with a comma.  Because some
    options themselves contain commas we use greedy prefix matching against the
    known option universe.
    """
    if not raw.strip():
        return set()

    selected = set()
    remaining = raw
    while remaining:
        matched = False
        for opt in universe:
            if remaining == opt:
                selected.add(opt)
                remaining = ""
                matched = True
                break
            if remaining.startswith(opt + ","):
                selected.add(opt)
                remaining = remaining[len(opt) + 1:]  # skip the comma
                matched = True
                break
        if not matched:
            # Fallback: treat the rest as one option (should not happen with
            # a complete universe, but prevents infinite loops)
            selected.add(remaining)
            remaining = ""
    return selected


def hamming_score(selected: set[str], correct: set[str], universe: list[str]) -> int:
    """
    Per-statement (Hamming) score: 1 point per option correctly handled
    (selected when correct, not selected when incorrect).
    Max = len(universe).
    """
    score = 0
    for opt in universe:
        is_correct = opt in correct
        is_selected = opt in selected
        if is_correct == is_selected:
            score += 1
    return score


def all_or_nothing(selected: set[str], correct: set[str]) -> int:
    return 1 if selected == correct else 0


def count_non_empty(row: dict, questions: list[str]) -> int:
    return sum(1 for q in questions if row.get(q, "").strip())


# ── Load data ────────────────────────────────────────────────────────────────
with open(DATA_FILE, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    all_rows = list(reader)

# Qualtrics: row 0 = display labels, row 1 = importIds, data from row 2
data_rows = all_rows[2:]
total_raw = len(data_rows)

# ── Step 1: keep rows with non-empty PROLIFIC_PID ───────────────────────────
pid_rows = [r for r in data_rows if r.get("PROLIFIC_PID", "").strip()]
n_with_pid = len(pid_rows)

# ── Step 2: deduplicate by PROLIFIC_PID ─────────────────────────────────────
pid_groups = defaultdict(list)
for r in pid_rows:
    pid_groups[r["PROLIFIC_PID"].strip()].append(r)

deduped = []
n_dupes_removed = 0
for pid, group in pid_groups.items():
    if len(group) == 1:
        deduped.append(group[0])
    else:
        # Keep the row with the most non-empty question answers
        best = max(group, key=lambda r: count_non_empty(r, QUESTIONS))
        deduped.append(best)
        n_dupes_removed += len(group) - 1

n_after_dedup = len(deduped)

# ── Step 3: attention check filtering ───────────────────────────────────────
AT1_CORRECT = "Green versus black"
AT2_CORRECT = "Social structure"

passed = []
failed_at1_only = 0
failed_at2_only = 0
failed_both = 0

for r in deduped:
    at1 = r.get("AT1", "").strip()
    at2 = r.get("AT2", "").strip()
    ok_at1 = at1 == AT1_CORRECT
    ok_at2 = at2 == AT2_CORRECT
    if ok_at1 and ok_at2:
        passed.append(r)
    elif not ok_at1 and not ok_at2:
        failed_both += 1
    elif not ok_at1:
        failed_at1_only += 1
    else:
        failed_at2_only += 1

n_passed = len(passed)
n_failed_att = n_after_dedup - n_passed

# ── Score each included participant ─────────────────────────────────────────
for r in passed:
    for q in QUESTIONS:
        raw = r.get(q, "")
        universe = OPTION_UNIVERSES[q]
        correct = CORRECT_ANSWERS[q]
        selected = parse_multiselect(raw, universe)
        r[f"_{q}_selected"] = selected
        r[f"_{q}_hamming"] = hamming_score(selected, correct, universe)
        r[f"_{q}_aon"] = all_or_nothing(selected, correct)

# ── Output ───────────────────────────────────────────────────────────────────

def fmt(x: float, decimals: int = 2) -> str:
    return f"{x:.{decimals}f}"


def section(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


# 1. Raw counts
section("1. RAW COUNTS")
print(f"  Total rows in CSV (excl. header rows):  {total_raw}")
print(f"  With non-empty PROLIFIC_PID:            {n_with_pid}")
print(f"  After deduplication:                    {n_after_dedup}  (removed {n_dupes_removed} duplicate(s))")
print(f"  After attention check filter:           {n_passed}  (excluded {n_failed_att})")
print()
print(f"  Attention check failures:")
print(f"    Failed AT1 only ('Green versus black'): {failed_at1_only}")
print(f"    Failed AT2 only ('Social structure'):  {failed_at2_only}")
print(f"    Failed both:                           {failed_both}")

# 2. Per-condition counts
section("2. PER-CONDITION COUNTS (included participants)")
cond_counts = Counter(r.get("assigned_doc", "").strip() for r in passed)
for cond in sorted(cond_counts):
    print(f"  {cond:<20} {cond_counts[cond]}")
print(f"  {'TOTAL':<20} {sum(cond_counts.values())}")

# 3. Q_timing_Page Submit stats
section("3. QUESTION PAGE TIMING (Q_timing_Page Submit, seconds)")
timing_vals = []
for r in passed:
    t = r.get("Q_timing_Page Submit", "").strip()
    if t:
        try:
            timing_vals.append(float(t))
        except ValueError:
            pass

if timing_vals:
    print(f"  N with timing data: {len(timing_vals)}")
    print(f"  Mean:   {fmt(statistics.mean(timing_vals))} s")
    print(f"  Median: {fmt(statistics.median(timing_vals))} s")
    print(f"  Min:    {fmt(min(timing_vals))} s")
    print(f"  Max:    {fmt(max(timing_vals))} s")
else:
    print("  No timing data found.")

# 4. Per-question accuracy (all included participants)
section("4. PER-QUESTION ACCURACY (all included participants)")
max_scores = {q: len(OPTION_UNIVERSES[q]) for q in QUESTIONS}

header = f"  {'Q':<6} {'n correct opts':<16} {'max pts':<9} {'mean Hamming':<15} {'mean Hamming%':<16} {'AoN rate'}"
print(header)
print("  " + "-" * (len(header) - 2))

for q in QUESTIONS:
    n_correct_opts = len(CORRECT_ANSWERS[q])
    max_pts = max_scores[q]
    hamming_vals = [r[f"_{q}_hamming"] for r in passed]
    aon_vals = [r[f"_{q}_aon"] for r in passed]
    if hamming_vals:
        mean_h = statistics.mean(hamming_vals)
        mean_aon = statistics.mean(aon_vals)
        print(
            f"  {q:<6} {n_correct_opts:<16} {max_pts:<9} "
            f"{fmt(mean_h):<15} {fmt(100*mean_h/max_pts)+'%':<16} {fmt(100*mean_aon)+'%'}"
        )

# 5. Per-question accuracy broken down by condition
section("5. PER-QUESTION ACCURACY BY CONDITION")
conditions = sorted(set(r.get("assigned_doc", "").strip() for r in passed))

for q in QUESTIONS:
    max_pts = max_scores[q]
    print(f"\n  {q} (max {max_pts} pts per-statement, {len(CORRECT_ANSWERS[q])} correct option(s))")
    print(f"  {'Condition':<20} {'N':<5} {'Mean Hamming':<15} {'Hamming%':<12} {'AoN rate'}")
    print("  " + "-" * 60)
    for cond in conditions:
        cond_rows = [r for r in passed if r.get("assigned_doc", "").strip() == cond]
        h_vals = [r[f"_{q}_hamming"] for r in cond_rows]
        a_vals = [r[f"_{q}_aon"] for r in cond_rows]
        if h_vals:
            mh = statistics.mean(h_vals)
            ma = statistics.mean(a_vals)
            print(
                f"  {cond:<20} {len(h_vals):<5} {fmt(mh):<15} "
                f"{fmt(100*mh/max_pts)+'%':<12} {fmt(100*ma)+'%'}"
            )

# 6. Notes on discrepancies vs task description
section("6. NOTES AND DISCREPANCIES")
print("""
  a) Bird names: The task spec referenced "Skreel" and "Copperhook" (short forms).
     The actual birds are the Ashen Skreel and Copperhook Finch. The old YAML
     (birds_q_multiselect.yaml) used placeholder names "crestling" and "sunwhistle".
     In pilot v4 these have been fully replaced by the Ashen Skreel and Copperhook
     Finch throughout both texts and questions.

  b) QP04 — Latin name matching vs diet matching:
     The task spec stated QP04 tested "diet matching" (Ashen Skreel -- bits of
     carrion; Copperhook Finch -- fruit). The LIVE SURVEY actually tested Latin
     name matching.  The pilot_v4.qsf on disk shows the diet-matching version,
     but the collected data shows Latin-name options.  Likely the QSF on disk is
     an earlier draft; the Qualtrics study was updated before data collection.

     Correct answer for QP04 (Latin name matching):
       Ashen Skreel  scientific name = Plumbea vorax  (from control.md)
       Copperhook Finch scientific name = Fringilla rufa  (from control.md)
     Options offered: Ashen skreel -- Fringilla rufa (wrong genus/epithet swap),
       Ashen skreel -- Plumbea rufa (wrong epithet: vorax not rufa),
       Copperhook finch -- Fringilla vorax (wrong epithet: rufa not vorax),
       Copperhook finch -- Plumbea rufa (wrong genus entirely).
     Therefore: NONE of the options are correct; the keyed answer is "None of
     the above" (option E).

  c) QP03 — option texts vs task spec:
     The task spec described correct options as "Skreel eats carrion / Copperhook
     eats fruit and seeds" and "Skreel mostly gray / Copperhook mostly orange".
     The actual text in the survey is:
       A: "The ashen skreel eats carrion, while the Copperhook finch eats seeds"
       B: "The ashen skreel is mostly gray, whereas the Copperhook finch is mostly orange"
     These correspond to the task spec's intended options 1 and 2 — scoring is
     unaffected.

  d) QP01 — unexpected option:
     "Rust orange, with a lighter patch at the throat" appears in the data but
     is NOT listed in pilot_v4.qsf (which has "Mostly gray, with a lighter patch
     at the throat" and "A pale ring around the eye" instead).  This is additional
     evidence that the live survey differed from the QSF on disk.

  e) QP05 — no "None of the above" observed:
     Unlike QP01-04, no participant selected "None of the above" for QP05, and
     the QSF does not list such an option.  The option universe for QP05 contains
     only 4 options (A-D), giving a maximum Hamming score of 4.
""")
