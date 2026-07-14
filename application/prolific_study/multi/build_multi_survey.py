#!/usr/bin/env python3
"""
Build the full multi-topic Qualtrics survey QSF.

Takes multi_study_conditions.qsf as base (Qualtrics-exported; contains all internal
metadata + 24 condition-assignment branches + birds blocks in sequential flow).

Adds blocks for the remaining topics, replaces the 3 sequential birds Standard blocks
with a BlockRandomizer containing all 4 topic groups (passage → questions → feedback).

Usage:
    python build_multi_survey.py
    python build_multi_survey.py --base multi_study_conditions.qsf --output multi_study.qsf
"""

import json
import argparse
import copy
import sys
import hashlib
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML required: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

MULTI_DIR = Path(__file__).parent
BASE_URL = "https://craaaa.github.io/simulating-memory/multi"

# Topics to include (with audio + questions)
TOPICS = ["birds", "fruits", "astronomy", "musical_instruments"]

# Block IDs already in the base QSF for birds
BIRDS_BLOCK_IDS = {
    "passage":  "BL_3pUIE2wY593YFU2",
    "qs":       "BL_0CYOCQ867W4e09E",
    "feedback": "BL_dpwpOFCj4XQ5jcW",
}

# QID + block ID offsets for new topics (must not collide with existing QIDs 1-99)
TOPIC_CFG = {
    "fruits":               {"qoff": 1100, "bloff": 20},
    "astronomy":            {"qoff": 1200, "bloff": 30},
    "musical_instruments":  {"qoff": 1300, "bloff": 40},
}

SLIDER_CHOICES = [{"Display": str(i)} for i in range(11)]


# ---------------------------------------------------------------------------
# Question builders (same as generate_topic_qsf.py)
# ---------------------------------------------------------------------------

def make_timing_question(q_id, export_tag):
    return {"Element": "SQ", "Payload": {
        "QuestionText": "Timing", "DefaultChoices": False,
        "DataExportTag": export_tag, "QuestionID": q_id,
        "QuestionType": "Timing", "Selector": "PageTimer",
        "DataVisibility": {"Private": False, "Hidden": False},
        "Configuration": {"QuestionDescriptionOption": "UseText", "MinSeconds": "0", "MaxSeconds": "0"},
        "QuestionDescription": "Timing",
        "Choices": {"1": {"Display": "First Click"}, "2": {"Display": "Last Click"},
                    "3": {"Display": "Page Submit"}, "4": {"Display": "Click Count"}},
        "GradingData": [], "Language": [], "NextChoiceId": 20, "NextAnswerId": 1,
    }}


def make_audio_question(q_id, topic, export_tag):
    cond_field = f"{topic}_cond"
    conditions = ["control", "repeat_short", "repeat_long", "distractor"]
    stimuli_lines = "\n".join(
        f'        "{c}": {{ "url": "{BASE_URL}/{topic}/audio/{c}.mp3" }},'
        for c in conditions
    )
    js = f"""Qualtrics.SurveyEngine.addOnload(function () {{
    if (window._audioQuestionLoaded) return;
    window._audioQuestionLoaded = true;
    var qthis = this;

    var STIMULI = {{
{stimuli_lines}
    }};

    var docId = "${{e://Field/{cond_field}}}";
    var doc = STIMULI[docId];

    if (!doc || !doc.url) {{
        qthis.getQuestionContainer().innerHTML =
            "<p>Configuration error: missing audio for condition=" + docId +
            ". Please return this study on Prolific.</p>";
        return;
    }}

    qthis.hideNextButton();
    var container = qthis.getQuestionContainer();
    container.innerHTML =
        '<div id="audio-status" style="text-align:center;font-weight:bold;margin-bottom:12px;">Loading audio...</div>' +
        '<div id="audio-start-wrap" style="text-align:center;margin-bottom:12px;"></div>' +
        '<div id="audio-progress" style="text-align:center;color:#666;margin-bottom:12px;"></div>';

    var statusEl   = document.getElementById("audio-status");
    var startWrap  = document.getElementById("audio-start-wrap");
    var progressEl = document.getElementById("audio-progress");
    var audio = new Audio(doc.url);
    audio.preload = "auto";
    audio.loop    = false;
    audio.addEventListener("contextmenu", function (e) {{ e.preventDefault(); }});

    var startTime = null;
    var progressIv = null;

    function beginPlayback() {{
        startTime = Date.now();
        startWrap.innerHTML = "";
        statusEl.textContent = "Audio is playing. Please listen carefully.";
        progressIv = setInterval(function () {{
            if (!isNaN(audio.duration) && audio.duration > 0) {{
                var remaining = Math.ceil(audio.duration - audio.currentTime);
                progressEl.textContent = remaining > 0 ? remaining + "s remaining" : "";
            }}
        }}, 1000);
        audio.play();
    }}

    audio.addEventListener("canplaythrough", function () {{
        statusEl.textContent = "Audio loaded. When you are ready, click to start listening.";
        startWrap.innerHTML =
            '<button id="audio-start-btn" style="font-size:1em;padding:8px 24px;">Start Audio</button>';
        document.getElementById("audio-start-btn").addEventListener("click", function () {{
            document.getElementById("audio-start-btn").disabled = true;
            beginPlayback();
        }});
    }}, {{ once: true }});

    audio.addEventListener("ended", function () {{
        if (progressIv) {{ clearInterval(progressIv); }}
        var rt = startTime ? (Date.now() - startTime) : null;
        Qualtrics.SurveyEngine.setEmbeddedData("{topic}_listening_rt", rt);
        Qualtrics.SurveyEngine.setEmbeddedData("{topic}_listening_completed", "true");
        statusEl.textContent  = "Audio complete. Click Next to continue.";
        progressEl.textContent = "";
        qthis.showNextButton();
    }});

    audio.addEventListener("error", function () {{
        if (progressIv) {{ clearInterval(progressIv); }}
        statusEl.textContent = "Audio failed to load. Please return this study on Prolific.";
    }});
}});
Qualtrics.SurveyEngine.addOnUnload(function () {{}});"""

    return {"Element": "SQ", "Payload": {
        "QuestionText": "Loading...", "DefaultChoices": False,
        "DataExportTag": export_tag, "QuestionType": "DB", "Selector": "TB",
        "DataVisibility": {"Private": False, "Hidden": False},
        "Configuration": {"QuestionDescriptionOption": "UseText"},
        "QuestionDescription": "Loading...", "ChoiceOrder": [],
        "Validation": {"Settings": {"Type": "None"}},
        "GradingData": [], "Language": [], "NextChoiceId": 4, "NextAnswerId": 1,
        "QuestionID": q_id, "QuestionJS": js,
    }}


def make_mc_question(q_id, q_entry, export_tag):
    options = q_entry["options"]
    n = len(options)
    none_key = str(n)
    choices = {str(k): {"Display": v} for k, v in options.items()}
    rand_keys = [str(k) for k in options if str(k) != none_key]
    fixed_order = ["{~Randomized~}"] * (n - 1) + [none_key]
    return {"Element": "SQ", "Payload": {
        "QuestionText": q_entry["question"],
        "DataExportTag": export_tag, "QuestionType": "MC",
        "Selector": "MAVR", "SubSelector": "TX",
        "DataVisibility": {"Private": False, "Hidden": False},
        "Configuration": {"QuestionDescriptionOption": "UseText"},
        "QuestionDescription": q_entry["question"][:100],
        "Choices": choices, "ChoiceOrder": list(choices.keys()),
        "Validation": {"Settings": {"ForceResponse": "ON", "ForceResponseType": "ON", "Type": "None"}},
        "Language": [], "NextChoiceId": n + 2, "NextAnswerId": 1,
        "QuestionID": q_id,
        "Randomization": {
            "Advanced": {"FixedOrder": fixed_order, "RandomSubSet": [],
                         "RandomizeAll": rand_keys, "ScaleReversal": [],
                         "TotalRandSubset": 0, "Undisplayed": []},
            "ConsistentScaleReversal": False, "EvenPresentation": False,
            "TotalRandSubset": "", "Type": "Advanced",
        },
    }}


def make_slider_question(q_id, text, export_tag):
    return {"Element": "SQ", "Payload": {
        "QuestionText": text, "DefaultChoices": False,
        "DataExportTag": export_tag, "QuestionType": "SS", "Selector": "TA",
        "DataVisibility": {"Private": False, "Hidden": False},
        "Configuration": {"QuestionDescriptionOption": "UseText"},
        "QuestionDescription": text,
        "Validation": {"Settings": {"ForceResponse": "ON", "ForceResponseType": "ON"}},
        "GradingData": [], "Language": [],
        "NextChoiceId": 12, "NextAnswerId": 1,
        "Category": "Gauges", "Scale": "TenGauge",
        "Choices": SLIDER_CHOICES, "QuestionID": q_id, "Direction": "horizontal",
    }}


def make_text_question(q_id, export_tag):
    text = (
        "Is there anything about the study you found notable or wanted to let us know about? "
        "For example: comments about the passage and/or narration, how challenging the "
        "questions were, any distractions or technical issues, or anything else."
    )
    return {"Element": "SQ", "Payload": {
        "QuestionText": text, "DefaultChoices": False,
        "DataExportTag": export_tag, "QuestionType": "TE", "Selector": "ML",
        "DataVisibility": {"Private": False, "Hidden": False},
        "Configuration": {"QuestionDescriptionOption": "UseText"},
        "QuestionDescription": text[:100],
        "Validation": {"Settings": {"ForceResponse": "OFF", "Type": "None"}},
        "GradingData": [], "Language": [], "NextChoiceId": 4, "NextAnswerId": 1,
        "QuestionID": q_id,
    }}


# ---------------------------------------------------------------------------
# Block + question generation per topic
# ---------------------------------------------------------------------------

def make_bl_id(seed):
    """Generate a Qualtrics-style 16-char alphanumeric block ID deterministically."""
    chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    digest = hashlib.sha256(seed.encode()).digest()
    result = ""
    for byte in digest[:16]:
        result += chars[byte % len(chars)]
    return f"BL_{result}"


def generate_topic_elements(topic, qoff, bloff):
    """
    Returns:
      block_entries: dict {key: block_payload} to add to BL
      sq_elements:   list of SQ dicts
      block_ids:     {"passage": id, "qs": id, "feedback": id}
    """
    qyaml = MULTI_DIR / topic / "questions.yaml"
    with open(qyaml) as f:
        qdata = yaml.safe_load(f)
    questions = qdata["questions"]
    content_qs = [q for q in questions if q["metadata"]["type"] == "content"]
    at_qs = [q for q in questions if q["metadata"]["type"] == "attention_check"]

    # QIDs
    passage_timing_qid = f"QID{qoff + 1}"
    passage_audio_qid  = f"QID{qoff + 2}"
    qs_timing_qid      = f"QID{qoff + 10}"
    content_qids = [f"QID{qoff + 11 + i}" for i in range(len(content_qs))]
    at_qids      = [f"QID{qoff + 11 + len(content_qs) + i}" for i in range(len(at_qs))]
    feedback_qids = [f"QID{qoff + 50}", f"QID{qoff + 51}", f"QID{qoff + 52}"]

    # Block IDs — deterministic 16-char alphanumeric (Qualtrics format)
    bl_passage  = make_bl_id(f"{topic}_passage")
    bl_qs       = make_bl_id(f"{topic}_qs")
    bl_feedback = make_bl_id(f"{topic}_feedback")

    # Advanced randomization layout for questions block
    n_content = len(content_qs)
    n_at = len(at_qs)
    fixed_order = [qs_timing_qid]
    if n_at == 0:
        fixed_order += ["{~Randomized~}"] * n_content
    elif n_at == 1:
        split = n_content // 2
        fixed_order += ["{~Randomized~}"] * split + [at_qids[0]] + ["{~Randomized~}"] * (n_content - split)
    elif n_at == 2:
        fixed_order += ["{~Randomized~}", "{~Randomized~}", at_qids[0],
                        "{~Randomized~}", "{~Randomized~}", at_qids[1]]
        fixed_order += ["{~Randomized~}"] * (n_content - 4)
    else:
        fixed_order += ["{~Randomized~}"] * n_content + at_qids

    all_qs_qids = [qs_timing_qid] + content_qids + at_qids

    block_entries = {
        f"bl_{topic}_passage": {
            "Type": "Standard", "SubType": "",
            "Description": f"{topic}_passage", "ID": bl_passage,
            "BlockElements": [
                {"Type": "Question", "QuestionID": passage_timing_qid},
                {"Type": "Question", "QuestionID": passage_audio_qid},
            ],
            "Options": {"BlockLocking": "false", "RandomizeQuestions": "false", "BlockVisibility": "Collapsed"},
        },
        f"bl_{topic}_qs": {
            "Type": "Standard", "SubType": "",
            "Description": f"{topic}_qs", "ID": bl_qs,
            "BlockElements": [{"Type": "Question", "QuestionID": q} for q in all_qs_qids],
            "Options": {
                "BlockLocking": "false", "RandomizeQuestions": "Advanced",
                "Randomization": {
                    "Advanced": {
                        "FixedOrder": fixed_order,
                        "RandomizeAll": content_qids,
                        "RandomSubSet": [], "Undisplayed": [],
                        "TotalRandSubset": 0, "QuestionsPerPage": 0,
                    },
                    "EvenPresentation": False,
                },
                "BlockVisibility": "Collapsed",
            },
        },
        f"bl_{topic}_feedback": {
            "Type": "Standard", "SubType": "",
            "Description": f"{topic}_feedback", "ID": bl_feedback,
            "BlockElements": [{"Type": "Question", "QuestionID": q} for q in feedback_qids],
            "Options": {"BlockLocking": "false", "RandomizeQuestions": "false", "BlockVisibility": "Collapsed"},
        },
    }

    sq_elements = [
        make_timing_question(passage_timing_qid, f"{topic}_timing_passage"),
        make_audio_question(passage_audio_qid, topic, f"{topic}_trial"),
        make_timing_question(qs_timing_qid, f"{topic}_timing_qs"),
    ]
    for i, q in enumerate(content_qs):
        sq_elements.append(make_mc_question(content_qids[i], q, f"{topic}_{q['q_id']}"))
    for i, q in enumerate(at_qs):
        sq_elements.append(make_mc_question(at_qids[i], q, f"{topic}_{q['q_id']}"))
    sq_elements += [
        make_slider_question(feedback_qids[0], "On a scale of 0 (easy) to 10 (difficult), how difficult did you find the passage?", f"{topic}_passage_difficulty"),
        make_slider_question(feedback_qids[1], "On a scale of 0 (easy) to 10 (difficult), how difficult did you find the questions?", f"{topic}_qs_difficulty"),
        make_text_question(feedback_qids[2], f"{topic}_comments"),
    ]

    return block_entries, sq_elements, {"passage": bl_passage, "qs": bl_qs, "feedback": bl_feedback}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="multi_study_conditions.qsf")
    parser.add_argument("--output", default="multi_study.qsf")
    args = parser.parse_args()

    base_path = MULTI_DIR / args.base
    out_path  = MULTI_DIR / args.output

    with open(base_path) as f:
        qsf = json.load(f)

    # --- Locate elements ---
    bl_elem = next(e for e in qsf["SurveyElements"] if e["Element"] == "BL")
    fl_elem = next(e for e in qsf["SurveyElements"] if e["Element"] == "FL")
    bl_payload = bl_elem["Payload"]
    flow = fl_elem["Payload"]["Flow"]

    # --- Generate blocks for non-birds topics ---
    topic_block_ids = {
        "birds": BIRDS_BLOCK_IDS,
    }
    new_sq_elements = []

    for topic in TOPICS:
        if topic == "birds":
            continue  # already in base QSF
        cfg = TOPIC_CFG[topic]
        block_entries, sq_elements, block_ids = generate_topic_elements(
            topic, cfg["qoff"], cfg["bloff"]
        )
        # Add to BL payload (next available int key)
        next_key = str(max(int(k) for k in bl_payload if k.isdigit()) + 1)
        for entry in block_entries.values():
            bl_payload[next_key] = entry
            next_key = str(int(next_key) + 1)

        new_sq_elements.extend(sq_elements)
        topic_block_ids[topic] = block_ids
        print(f"  {topic}: {len(sq_elements)} questions generated")

    # --- Add top-level wrapper fields Qualtrics requires on each SQ element ---
    survey_id = qsf["SurveyEntry"]["SurveyID"]
    for sq in new_sq_elements:
        payload = sq["Payload"]
        qid_val = payload.get("QuestionID", "")
        qtext = payload.get("QuestionText", "")
        sq["SurveyID"] = survey_id
        sq["PrimaryAttribute"] = qid_val
        sq["SecondaryAttribute"] = qtext[:200] if qtext else None
        sq["TertiaryAttribute"] = None

    # --- Add new SQ elements to SurveyElements ---
    qsf["SurveyElements"].extend(new_sq_elements)

    # --- Find next safe FlowID integer ---
    def max_fl(node):
        m = 0
        if isinstance(node, dict):
            fid = node.get("FlowID", "")
            if isinstance(fid, str) and fid.startswith("FL_"):
                try: m = max(m, int(fid[3:]))
                except ValueError: pass
            for v in node.values():
                m = max(m, max_fl(v))
        elif isinstance(node, list):
            for x in node:
                m = max(m, max_fl(x))
        return m
    fl_counter = [max_fl(fl_elem) + 1]

    def next_fl():
        fid = f"FL_{fl_counter[0]}"
        fl_counter[0] += 1
        return fid

    # --- Replace the 3 sequential birds Standard blocks with a BlockRandomizer ---
    # In multi_study_conditions.qsf the birds blocks appear as Standard elements.
    birds_ids_ordered = [
        BIRDS_BLOCK_IDS["passage"],
        BIRDS_BLOCK_IDS["qs"],
        BIRDS_BLOCK_IDS["feedback"],
    ]
    # Find index of first birds block (passage) in flow
    passage_idx = next(
        i for i, fe in enumerate(flow)
        if fe.get("ID") == BIRDS_BLOCK_IDS["passage"]
    )

    groups = []
    for topic in TOPICS:
        bids = topic_block_ids[topic]
        group = {
            "Type": "Group",
            "FlowID": next_fl(),
            "Description": topic,
            "Flow": [
                {"Type": "Standard", "ID": bids["passage"],  "FlowID": next_fl(), "Autofill": []},
                {"Type": "Standard", "ID": bids["qs"],       "FlowID": next_fl(), "Autofill": []},
                {"Type": "Standard", "ID": bids["feedback"], "FlowID": next_fl(), "Autofill": []},
            ],
        }
        groups.append(group)
        print(f"  {topic}: group added to BlockRandomizer")

    block_randomizer = {
        "Type": "BlockRandomizer",
        "FlowID": next_fl(),
        "SubSet": len(TOPICS),
        "Flow": groups,
        "EvenPresentation": False,
    }

    # Remove the 3 sequential birds Standard blocks and insert BlockRandomizer at same position
    birds_ids_set = set(birds_ids_ordered)
    new_flow = []
    inserted = False
    for fe in flow:
        if fe.get("ID") in birds_ids_set:
            if not inserted:
                new_flow.append(block_randomizer)
                inserted = True
            # skip the individual birds Standard blocks
        else:
            new_flow.append(fe)
    flow[:] = new_flow
    print(f"  Replaced 3 sequential birds blocks with BlockRandomizer")

    # Keep original SurveyID — Qualtrics assigns a new server-side ID on import.
    # Changing it here without updating every element's SurveyID field causes rejection.
    qsf["SurveyEntry"]["SurveyName"] = "Multi-Topic Listening Study"

    # --- Update Properties.Count (recursive flow elements + SQ count) ---
    def count_flow(flow_list):
        total = 0
        for fe in flow_list:
            total += 1
            if fe.get("Flow"):
                total += count_flow(fe["Flow"])
        return total

    sq_count = sum(1 for e in qsf["SurveyElements"] if e.get("Element") == "SQ")
    flow_count = count_flow(flow)
    fl_elem["Payload"]["Properties"]["Count"] = flow_count + sq_count
    print(f"  Properties.Count updated: {flow_count} flow + {sq_count} SQ = {flow_count + sq_count}")

    # --- Write output ---
    out_path.write_text(json.dumps(qsf, indent=2, ensure_ascii=True))
    print(f"\nSaved: {out_path}")
    print(f"Total SurveyElements: {len(qsf['SurveyElements'])}")

    # Summary
    print("\nBlockRandomizer groups:")
    for t in TOPICS:
        bids = topic_block_ids[t]
        print(f"  {t}: {bids['passage']} → {bids['qs']} → {bids['feedback']}")


if __name__ == "__main__":
    main()
