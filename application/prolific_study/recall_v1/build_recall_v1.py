#!/usr/bin/env python3
"""Build `recall_v1.qsf` from `../multi_v6/multi_v6.qsf`.

recall_v1 keeps multi_v6's study phase exactly as it is -- same audio, same four
passage conditions, same 24-group assignment, same topic-order randomisation --
and replaces the recognition QA task with a production measure:

    passage -> 60s math task (filled delay) -> free recall

per topic. The math task is a Brown-Peterson style filled delay: self-paced
arithmetic that occupies working memory and blocks rehearsal of the passage.

Usage:
    python build_recall_v1.py                # writes recall_v1.qsf
    python build_recall_v1.py --check-only   # validate, write nothing

Why a builder and not a hand-edited QSF: multi_v6.qsf is 159 KB of minified
single-line JSON with a 160-node survey flow. Hand-editing it is unreviewable.
This follows the convention of ../multi_v5/build_v5_survey.py, which is the
script multi_v6 itself descends from.

Naming note: multi_v6 already uses "distractor" for a *passage condition*
(audio/distractor.mp3). The arithmetic task is called the **math task**
everywhere here so that name stays unambiguous.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE.parent / "multi_v6" / "multi_v6.qsf"
OUTPUT = HERE / "recall_v1.qsf"

MATH_JS = HERE / "qualtrics_math_question.js"
RECALL_JS = HERE / "qualtrics_recall_question.js"

# Topic key -> the four `_qs` blocks to remove, resolved by block ID. Block
# *descriptions* are unreliable in the template: two carry stray Qualtrics
# duplicate-block suffixes ("astronomy_qs - Jun 29, 2026"), so always match by ID.
QS_BLOCK_ID = {
    "martial_arts": "BL_8iCv8qNS7eti0e2",
    "fruits_v2": "BL_23Q5vwV4c8Qhjj8",
    "astronomy": "BL_bw7fxbJW7IkF3GS",
    "fabrics": "BL_2aAYirFCVeDDo6q",
}

PASSAGE_BLOCK_ID = {
    "martial_arts": "BL_W65xDI9qbbnsC7hU",
    "fruits_v2": "BL_mv7lLyv86Ur6nq6a",
    "astronomy": "BL_PQWoxw4EVWwzZiqI",
    "fabrics": "BL_V1mkkrfSJWqHBI40",
}

TOPICS = ["martial_arts", "fruits_v2", "astronomy", "fabrics"]

# The topic key is `fruits_v2` but its content directory is `fruits`. multi_v6's
# data_prep.py carries the same map as TOPIC_CONTENT_DIRNAME; do not try to
# derive one from the other.
TOPIC_CONTENT_DIRNAME = {
    "martial_arts": "martial_arts",
    "fruits_v2": "fruits",
    "astronomy": "astronomy",
    "fabrics": "fabrics",
}

# Recall prompts name only the topic, never passage content, so they cannot cue
# specific facts. Phrasing follows the per-topic intros already in the trial JS.
TOPIC_BLURB = {
    "martial_arts": "a type of combat sport",
    "fruits_v2": "a type of fruit",
    "astronomy": "an astronomical object",
    "fabrics": "a type of textile",
}

# Attention-check answer keys to drop from flow node FL_143. The
# musical_instruments key stays: its topic is already orphaned in the template
# and cleaning that up is not this change's business.
AT_CORRECT_TO_DROP = {f"{t}_AT_correct" for t in TOPICS}

# The PoorQuality screen-out branch and its EndSurvey child. FL_160's logic
# references the four attention-check QIDs this build deletes, so it must go too
# or the QSF ships with branch logic pointing at nonexistent questions.
FLOW_IDS_TO_DROP = {"FL_160", "FL_142"}

# Flow-unreachable in the template (a 5th topic dropped after v5, plus the Trash
# block). Pre-existing dead weight, left alone -- but validation must not trip
# over the QA/AT questions they still contain.
ORPHAN_BLOCK_IDS = {
    "BL_musinstr_passage",
    "BL_musinstr_qs",
    "BL_musinstr_feedback",
    "BL_a4XQDZWLQJLQ11Q",
}

# New QIDs live in an unused 4000 band, verified free of collisions at build time.
QID_BASE = 4000

SURVEY_NAME = "recall_v1"

MIN_RECALL_SECONDS = 60
MATH_DURATION_SECONDS = 60


# ----------------------------------------------------------------------------
# small helpers over the QSF structure
# ----------------------------------------------------------------------------

def elements_by_type(qsf: dict, element: str) -> list[dict]:
    return [e for e in qsf["SurveyElements"] if e["Element"] == element]


def sole_element(qsf: dict, element: str) -> dict:
    matches = elements_by_type(qsf, element)
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one {element} element, found {len(matches)}")
    return matches[0]


def block_payloads(qsf: dict) -> dict[str, dict]:
    """Blocks keyed by the BL payload's own string index."""
    return sole_element(qsf, "BL")["Payload"]


def blocks_by_id(qsf: dict) -> dict[str, dict]:
    return {b["ID"]: b for b in block_payloads(qsf).values()}


def questions_by_id(qsf: dict) -> dict[str, dict]:
    out = {}
    for e in elements_by_type(qsf, "SQ"):
        out[e["Payload"]["QuestionID"]] = e["Payload"]
    return out


def walk_flow(nodes: list[dict]):
    """Yield (node, parent_list) for every node in the flow tree."""
    for node in nodes:
        yield node, nodes
        if "Flow" in node:
            yield from walk_flow(node["Flow"])


def flow_root(qsf: dict) -> dict:
    return sole_element(qsf, "FL")["Payload"]


def block_question_ids(block: dict) -> list[str]:
    return [
        be["QuestionID"]
        for be in block.get("BlockElements", [])
        if be.get("Type") == "Question"
    ]


# ----------------------------------------------------------------------------
# element constructors
# ----------------------------------------------------------------------------

def make_sq_element(survey_id: str, qid: str, secondary: str, payload: dict) -> dict:
    return {
        "SurveyID": survey_id,
        "Element": "SQ",
        "PrimaryAttribute": qid,
        "SecondaryAttribute": secondary,
        "TertiaryAttribute": None,
        "Payload": payload,
    }


def make_timing_question(survey_id: str, qid: str, export_tag: str) -> dict:
    """A PageTimer, mirroring the template's timers exactly.

    MinSeconds/MaxSeconds are "0" like every other timer in this survey: these
    are pure measurement, not gating. All page gating is done in JS.
    """
    payload = {
        "QuestionText": "Timing",
        "DefaultChoices": False,
        "DataExportTag": export_tag,
        "QuestionID": qid,
        "QuestionType": "Timing",
        "Selector": "PageTimer",
        "DataVisibility": {"Private": False, "Hidden": False},
        "Configuration": {
            "QuestionDescriptionOption": "UseText",
            "MinSeconds": "0",
            "MaxSeconds": "0",
        },
        "QuestionDescription": "Timing",
        "Choices": {
            "1": {"Display": "First Click"},
            "2": {"Display": "Last Click"},
            "3": {"Display": "Page Submit"},
            "4": {"Display": "Click Count"},
        },
        "GradingData": [],
        "Language": [],
        "NextChoiceId": 20,
        "NextAnswerId": 1,
        "QuestionJS": None,
    }
    return make_sq_element(survey_id, qid, "Timing", payload)


def make_math_question(survey_id: str, qid: str, export_tag: str, js: str) -> dict:
    """Text/Graphic shell whose JS builds the whole task, like <topic>_trial."""
    payload = {
        "QuestionText": "Loading...",
        "DefaultChoices": False,
        "DataExportTag": export_tag,
        "QuestionType": "DB",
        "Selector": "TB",
        "DataVisibility": {"Private": False, "Hidden": False},
        "Configuration": {"QuestionDescriptionOption": "UseText"},
        "QuestionDescription": "Loading...",
        "ChoiceOrder": [],
        "Validation": {"Settings": {"Type": "None"}},
        "GradingData": [],
        "Language": [],
        "NextChoiceId": 4,
        "NextAnswerId": 1,
        "QuestionID": qid,
        "QuestionJS": js,
    }
    return make_sq_element(survey_id, qid, "Loading...", payload)


def make_recall_question(survey_id: str, qid: str, export_tag: str, prompt: str, js: str) -> dict:
    """Essay text box. Qualtrics owns the textarea so the response exports
    natively as the column <topic>_recall.

    ForceResponse is OFF on purpose: requiring text would pressure participants
    into confabulating when they remember nothing. An empty recall is a real
    datum, excluded (or not) at analysis time.
    """
    payload = {
        "QuestionText": prompt,
        "QuestionType": "TE",
        "Selector": "ESTB",
        "QuestionDescription": prompt[:97] + "..." if len(prompt) > 100 else prompt,
        "Validation": {"Settings": {"Type": "None", "ForceResponse": "OFF"}},
        "Language": [],
        "SearchSource": {"AllowFreeResponse": "false"},
        "DataVisibility": {"Private": False, "Hidden": False},
        "Configuration": {"QuestionDescriptionOption": "UseText"},
        "NextChoiceId": 1,
        "NextAnswerId": 1,
        "DataExportTag": export_tag,
        "QuestionID": qid,
        "QuestionJS": js,
    }
    return make_sq_element(survey_id, qid, prompt[:100], payload)


def make_block(block_id: str, description: str, question_ids: list[str]) -> dict:
    """A block mirroring the passage block's shape.

    One block renders as one page in this survey -- no block in the template
    contains a Page Break -- which is why math and recall each need their own
    block. Putting both in one block would show them on the same page and defeat
    the filled delay.
    """
    return {
        "Type": "Standard",
        "SubType": "",
        "Description": description,
        "ID": block_id,
        "BlockElements": [{"Type": "Question", "QuestionID": q} for q in question_ids],
        "Options": {
            "BlockLocking": "false",
            "RandomizeQuestions": "false",
            "BlockVisibility": "Collapsed",
        },
    }


# ----------------------------------------------------------------------------
# transform
# ----------------------------------------------------------------------------

def load_js(path: Path, replacements: dict[str, str]) -> str:
    js = path.read_text(encoding="utf-8")
    for needle, value in replacements.items():
        if needle not in js:
            raise SystemExit(f"{path.name}: expected placeholder {needle} not found")
        js = js.replace(needle, value)
    return js


def next_flow_id(root: dict) -> int:
    """Allocate above every FlowID present, and above Properties.Count."""
    highest = int(root.get("Properties", {}).get("Count", 0))
    for node, _ in walk_flow(root["Flow"]):
        fid = node.get("FlowID", "")
        m = re.fullmatch(r"FL_(\d+)", fid)
        if m:
            highest = max(highest, int(m.group(1)))
    return highest + 1


def build(template: dict) -> dict:
    qsf = copy.deepcopy(template)
    survey_id = qsf["SurveyEntry"]["SurveyID"]

    # -- 1. rename so the import lands as a distinct, inactive project --------
    qsf["SurveyEntry"]["SurveyName"] = SURVEY_NAME
    qsf["SurveyEntry"]["SurveyStatus"] = "Inactive"
    qsf["SurveyEntry"]["LastActivated"] = "0000-00-00 00:00:00"

    blocks = block_payloads(qsf)
    by_id = blocks_by_id(qsf)

    for topic, bid in QS_BLOCK_ID.items():
        if bid not in by_id:
            raise SystemExit(f"template has no {topic} _qs block {bid}")

    # -- 2. delete the QA blocks and their questions --------------------------
    # Delete by block membership, not by export-tag pattern: the QA tags are not
    # uniform across topics (martial_arts_QMA01, fruits_v2_QF_V2_01,
    # astronomy_QA01, fabrics_QFB01).
    deleted_qids: set[str] = set()
    for bid in QS_BLOCK_ID.values():
        deleted_qids.update(block_question_ids(by_id[bid]))

    doomed_keys = [k for k, b in blocks.items() if b["ID"] in set(QS_BLOCK_ID.values())]
    for k in doomed_keys:
        del blocks[k]

    qsf["SurveyElements"] = [
        e
        for e in qsf["SurveyElements"]
        if not (e["Element"] == "SQ" and e["Payload"]["QuestionID"] in deleted_qids)
    ]

    # -- 3. drop the AT answer keys and the now-dangling PoorQuality branch ---
    root = flow_root(qsf)

    for node, _ in walk_flow(root["Flow"]):
        if node.get("Type") == "EmbeddedData":
            node["EmbeddedData"] = [
                ed for ed in node["EmbeddedData"] if ed.get("Field") not in AT_CORRECT_TO_DROP
            ]

    def prune(nodes: list[dict]) -> list[dict]:
        kept = []
        for node in nodes:
            if node.get("FlowID") in FLOW_IDS_TO_DROP:
                continue
            if "Flow" in node:
                node["Flow"] = prune(node["Flow"])
            kept.append(node)
        return kept

    root["Flow"] = prune(root["Flow"])

    # -- 4. build the new questions and blocks -------------------------------
    existing_qids = set(questions_by_id(qsf))
    qid_counter = QID_BASE
    new_elements: list[dict] = []
    new_blocks: dict[str, dict] = {}

    def alloc_qid() -> str:
        nonlocal qid_counter
        while f"QID{qid_counter}" in existing_qids:
            qid_counter += 1
        qid = f"QID{qid_counter}"
        existing_qids.add(qid)
        qid_counter += 1
        return qid

    for topic in TOPICS:
        math_js = load_js(MATH_JS, {'"__TOPIC__"': json.dumps(topic)})
        # This prompt is the source of truth. It lands in the question's
        # QuestionText (what Qualtrics renders) and is also inlined into the JS as
        # RECALL_PROMPT, which rewrites the stem as belt-and-braces in case the
        # theme's .QuestionText markup differs. Both come from here, so they
        # cannot drift.
        prompt = (
            f"You just heard a passage about {TOPIC_BLURB[topic]}. "
            "Write down everything you can remember about it. "
            "Include as much detail as you can, in any order."
        )
        recall_js = load_js(RECALL_JS, {'"__PROMPT__"': json.dumps(prompt)})

        t_math_qid = alloc_qid()
        math_qid = alloc_qid()
        t_recall_qid = alloc_qid()
        recall_qid = alloc_qid()

        new_elements += [
            make_timing_question(survey_id, t_math_qid, f"{topic}_timing_math"),
            make_math_question(survey_id, math_qid, f"{topic}_math", math_js),
            make_timing_question(survey_id, t_recall_qid, f"{topic}_timing_recall"),
            make_recall_question(survey_id, recall_qid, f"{topic}_recall", prompt, recall_js),
        ]

        new_blocks[topic] = {
            "math": make_block(f"BL_{topic}_math", f"{topic}_math", [t_math_qid, math_qid]),
            "recall": make_block(
                f"BL_{topic}_recall", f"{topic}_recall", [t_recall_qid, recall_qid]
            ),
        }

    qsf["SurveyElements"].extend(new_elements)

    next_block_key = max(int(k) for k in blocks) + 1
    for topic in TOPICS:
        for kind in ("math", "recall"):
            blocks[str(next_block_key)] = new_blocks[topic][kind]
            next_block_key += 1

    # -- 5. rewire the flow: swap each _qs node for math then recall ----------
    fid = next_flow_id(root)
    swapped = set()

    for node, parent in walk_flow(root["Flow"]):
        if node.get("Type") not in ("Standard", "Block"):
            continue
        for topic, qs_bid in QS_BLOCK_ID.items():
            if node.get("ID") != qs_bid:
                continue
            idx = parent.index(node)
            replacement = []
            for kind in ("math", "recall"):
                replacement.append(
                    {
                        "Type": "Standard",
                        "ID": new_blocks[topic][kind]["ID"],
                        "FlowID": f"FL_{fid}",
                        "Autofill": [],
                    }
                )
                fid += 1
            parent[idx : idx + 1] = replacement
            swapped.add(topic)
            break

    if swapped != set(TOPICS):
        raise SystemExit(f"flow rewire missed topics: {set(TOPICS) - swapped}")

    root.setdefault("Properties", {})["Count"] = fid

    # -- 6. declare the math task's embedded data ----------------------------
    # Recall needs no embedded data: the recall question is a real TE, so its text
    # exports natively as <topic>_recall, and its duration comes from
    # <topic>_timing_recall_Page Submit.
    math_fields = []
    for topic in TOPICS:
        for suffix in ("n_attempted", "n_correct", "trials"):
            field = f"{topic}_math_{suffix}"
            math_fields.append(
                {
                    "Description": field,
                    "Type": "Custom",
                    "Field": field,
                    "VariableType": "String",
                    "DataVisibility": [],
                    "AnalyzeText": False,
                    "Value": "",
                }
            )

    ed_node = {"Type": "EmbeddedData", "FlowID": f"FL_{fid}", "EmbeddedData": math_fields}
    fid += 1
    root["Properties"]["Count"] = fid

    # Place it alongside the other top-level declarations, before any block runs.
    insert_at = 0
    for i, node in enumerate(root["Flow"]):
        if node.get("Type") == "EmbeddedData":
            insert_at = i + 1
        else:
            break
    root["Flow"].insert(insert_at, ed_node)

    # -- 7. reword the feedback slider ---------------------------------------
    # It rated "Passage" and "Questions"; there are no questions any more.
    # Scoped to the four live topics: the orphaned musical_instruments slider is
    # left exactly as the template has it.
    live_difficulty_tags = {f"{t}_difficulty" for t in TOPICS}
    stem = "How difficult did you find the passage and the recall task?"
    for e in elements_by_type(qsf, "SQ"):
        payload = e["Payload"]
        if payload.get("DataExportTag") not in live_difficulty_tags:
            continue
        payload["QuestionText"] = stem
        payload["QuestionDescription"] = stem
        choices = payload.get("Choices", {})
        if choices.get("2", {}).get("Display") == "Questions":
            choices["2"]["Display"] = "Recall task"
        e["SecondaryAttribute"] = stem

    return qsf


# ----------------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------------

def duplicate_export_tags(qsf: dict) -> set[str]:
    tags = [
        e["Payload"].get("DataExportTag")
        for e in elements_by_type(qsf, "SQ")
        if e["Payload"].get("DataExportTag")
    ]
    return {t for t in tags if tags.count(t) > 1}


def undeclared_piped_fields(qsf: dict) -> set[str]:
    """Embedded fields referenced via ${e://Field/...} but never declared."""
    root = flow_root(qsf)

    declared = set()
    for node, _ in walk_flow(root["Flow"]):
        for ed in node.get("EmbeddedData", []) or []:
            declared.add(ed.get("Field"))

    piped = set()
    for payload in questions_by_id(qsf).values():
        haystack = (payload.get("QuestionJS") or "") + (payload.get("QuestionText") or "")
        piped.update(re.findall(r"\$\{e://Field/([^}]+)\}", haystack))
    for node, _ in walk_flow(root["Flow"]):
        for ed in node.get("EmbeddedData", []) or []:
            piped.update(re.findall(r"\$\{e://Field/([^}]+)\}", str(ed.get("Value") or "")))

    return piped - declared


def validate(qsf: dict, template: dict, deleted_qids: set[str]) -> list[str]:
    """Internal-consistency checks. These prove the QSF is self-consistent; they
    do not prove Qualtrics accepts it. Importing is the only real test.

    Two checks are measured *relative to the template*, because multi_v6 already
    violates them and inherited defects are not this build's business:
      - duplicate DataExportTags: the template reuses "Q1"/"Q2" across the
        consent, audio-check, instructions and debrief questions.
      - undeclared piped fields: the orphaned musical_instruments trial JS pipes
        `musical_instruments_cond`, which no surviving branch declares.
    Only *newly introduced* violations of these two are errors.
    """
    errors: list[str] = []

    questions = questions_by_id(qsf)
    blocks = blocks_by_id(qsf)
    root = flow_root(qsf)

    # 1. every block element resolves to a real question
    for bid, block in blocks.items():
        for qid in block_question_ids(block):
            if qid not in questions:
                errors.append(f"block {bid} references missing question {qid}")

    # 2. flow integrity: block nodes resolve, and no surviving branch logic
    #    points at a deleted question (this is the check that catches FL_160).
    for node, _ in walk_flow(root["Flow"]):
        if node.get("Type") in ("Standard", "Block") and "ID" in node:
            if node["ID"] not in blocks:
                errors.append(f"flow {node.get('FlowID')} references missing block {node['ID']}")
        logic = node.get("BranchLogic")
        if logic:
            for qid in re.findall(r"QID\d+", json.dumps(logic)):
                if qid not in questions:
                    errors.append(
                        f"flow {node.get('FlowID')} branch logic references missing {qid}"
                    )

    # 3. unique QIDs and export tags
    qids = [e["Payload"]["QuestionID"] for e in elements_by_type(qsf, "SQ")]
    if len(qids) != len(set(qids)):
        errors.append("duplicate QuestionIDs")
    for tag in sorted(duplicate_export_tags(qsf) - duplicate_export_tags(template)):
        errors.append(f"newly duplicated DataExportTag: {tag}")

    # 4. every piped embedded field is declared somewhere in the flow
    for field in sorted(undeclared_piped_fields(qsf) - undeclared_piped_fields(template)):
        errors.append(f"piped embedded field never declared: {field}")

    # 5. per-topic shape, scoped to flow-reachable blocks (the orphaned musinstr
    #    and Trash blocks are pre-existing and exempt)
    reachable_bids = {
        node["ID"]
        for node, _ in walk_flow(root["Flow"])
        if node.get("Type") in ("Standard", "Block") and "ID" in node
    }
    for bid in ORPHAN_BLOCK_IDS:
        if bid in reachable_bids:
            errors.append(f"orphan block {bid} unexpectedly reachable")

    reachable_tags = set()
    for bid in reachable_bids:
        if bid not in blocks:
            continue
        for qid in block_question_ids(blocks[bid]):
            if qid in questions:
                reachable_tags.add(questions[qid].get("DataExportTag"))

    for topic in TOPICS:
        for suffix in ("trial", "math", "recall", "difficulty", "timing_passage",
                       "timing_math", "timing_recall"):
            tag = f"{topic}_{suffix}"
            if tag not in reachable_tags:
                errors.append(f"missing reachable question {tag}")

    for qid in deleted_qids:
        if qid in questions:
            errors.append(f"deleted QA/AT question still present: {qid}")

    # 6. the study-phase machinery must survive untouched
    tmpl_root = flow_root(template)
    tmpl_nodes = {n.get("FlowID"): n for n, _ in walk_flow(tmpl_root["Flow"])}
    out_nodes = {n.get("FlowID"): n for n, _ in walk_flow(root["Flow"])}

    for fid in ("FL_74", "FL_76"):
        if fid not in out_nodes:
            errors.append(f"screening branch {fid} was lost")
        elif json.dumps(out_nodes[fid], sort_keys=True) != json.dumps(
            tmpl_nodes[fid], sort_keys=True
        ):
            errors.append(f"screening branch {fid} was modified")

    group_branches = [
        n
        for n in out_nodes.values()
        if n.get("Type") == "Branch" and str(n.get("Description", "")).startswith("Group ")
    ]
    if len(group_branches) != 24:
        errors.append(f"expected 24 group branches, found {len(group_branches)}")

    blob = json.dumps(qsf)
    for code in ("COY4SWKW", "CEFEXCUH", "C1IVKJCG"):
        if code not in blob:
            errors.append(f"Prolific completion code {code} missing")
    if "C1ICGI9P" in blob:
        errors.append("PoorQuality code C1ICGI9P should be gone with FL_160")

    for fid in FLOW_IDS_TO_DROP:
        if fid in out_nodes:
            errors.append(f"{fid} should have been removed")

    # 7. round-trips as JSON with the expected element count
    reparsed = json.loads(json.dumps(qsf))
    if len(reparsed["SurveyElements"]) != len(qsf["SurveyElements"]):
        errors.append("SurveyElements count changed on round-trip")

    return errors


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--template", default=str(TEMPLATE))
    ap.add_argument("--out", default=str(OUTPUT))
    ap.add_argument("--check-only", action="store_true", help="validate, write nothing")
    args = ap.parse_args()

    template = json.loads(Path(args.template).read_text(encoding="utf-8"))

    by_id = blocks_by_id(template)
    deleted_qids: set[str] = set()
    for bid in QS_BLOCK_ID.values():
        deleted_qids.update(block_question_ids(by_id[bid]))

    qsf = build(template)
    errors = validate(qsf, template, deleted_qids)

    if errors:
        print(f"FAILED with {len(errors)} problem(s):", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        raise SystemExit(1)

    n_sq = len(elements_by_type(qsf, "SQ"))
    n_blocks = len(block_payloads(qsf))
    print(f"validation passed: {n_sq} questions, {n_blocks} blocks")
    print(f"removed {len(deleted_qids)} QA/attention-check/timing questions")

    if args.check_only:
        print("--check-only: nothing written")
        return

    out = Path(args.out)
    out.write_text(json.dumps(qsf, ensure_ascii=True), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
