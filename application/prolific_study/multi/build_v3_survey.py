#!/usr/bin/env python3
"""
Build multi_v3.qsf from multi_v2.qsf (base/conditions) + multi_v2/ topic content.

Keeps all non-topic structure intact (conditions flow, 24 branches, consent,
audio check, debrief, instructions, BlockRandomizer). Replaces topic SQ elements
(passage audio, content questions, attention checks, feedback) with fresh content
from multi_v2/topic/questions.yaml. Audio URLs updated to multi_v2 gh-pages path.

Usage:
    python build_v3_survey.py
    python build_v3_survey.py --base multi_v2.qsf --content-dir ../multi_v2 --output multi_v3.qsf
"""

import json
import argparse
import hashlib
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML required: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

MULTI_DIR = Path(__file__).parent
DEFAULT_BASE = MULTI_DIR / "multi_v2.qsf"
DEFAULT_CONTENT = MULTI_DIR.parent / "multi_v2"
DEFAULT_OUTPUT = MULTI_DIR / "multi_v3.qsf"
BASE_URL = "https://craaaa.github.io/simulating-memory/multi_v2"

TOPICS = ["birds", "fruits", "astronomy", "musical_instruments"]

# Block IDs from multi_v2.qsf — kept unchanged so FL flow remains valid
TOPIC_BLOCK_IDS = {
    "birds": {
        "passage":  "BL_W65xDI9qbbnsC7hU",
        "qs":       "BL_8iCv8qNS7eti0e2",
        "feedback": "BL_3l3HZvbfkuFcbr0",
    },
    "fruits": {
        "passage":  "BL_mv7lLyv86Ur6nq6a",
        "qs":       "BL_23Q5vwV4c8Qhjj8",
        "feedback": "BL_eFdlaACmEVJFMmG",
    },
    "astronomy": {
        "passage":  "BL_PQWoxw4EVWwzZiqI",
        "qs":       "BL_bw7fxbJW7IkF3GS",
        "feedback": "BL_exp7Y2hLES2qlds",
    },
    "musical_instruments": {
        "passage":  "BL_V1mkkrfSJWqHBI40",
        "qs":       "BL_2aAYirFCVeDDo6q",
        "feedback": "BL_9XM2zWkUnyoqR2C",
    },
}

# New QID ranges — must not collide with existing non-topic QIDs
# Existing non-topic QIDs are in the hundreds (consent, audio check, etc.)
TOPIC_QID_OFFSET = {
    "birds":               2000,
    "fruits":              2100,
    "astronomy":           2200,
    "musical_instruments": 2300,
}

SLIDER_CHOICES = [{"Display": str(i)} for i in range(11)]


# ---------------------------------------------------------------------------
# Question builders
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
    url_lines = "\n".join(
        f'        "{c}":      {{ "url": "{BASE_URL}/{topic}/audio/{c}.mp3" }},'
        for c in conditions
    )
    js = f"""Qualtrics.SurveyEngine.addOnload(function () {{
    var qthis = this;
    var _guardKey = '_audioLoaded_' + this.questionId;
    if (window[_guardKey]) return;
    window[_guardKey] = true;

    var STIMULI = {{
{url_lines}
    }};
    var docId = "${{e://Field/{cond_field}}}";
    var doc = STIMULI[docId];

    if (!doc || !doc.url) {{
		qthis.hideNextButton();
        qthis.getQuestionContainer().innerHTML =
            "<p>Configuration error: missing audio for condition=" + docId +
            ". Please return this study on Prolific.</p>";
        return;
    }}

    qthis.hideNextButton();

    var container = qthis.getQuestionContainer();
    container.innerHTML =
        '<div id="audio-status" style="text-align:center;font-weight:bold;margin-bottom:12px;">Loading audio…</div>' +
        '<div id="audio-start-wrap" style="text-align:center;margin-bottom:12px;"></div>' +
        '<div id="audio-progress" style="text-align:center;color:#666;margin-bottom:12px;"></div>';

    var statusEl   = document.getElementById("audio-status");
    var startWrap  = document.getElementById("audio-start-wrap");
    var progressEl = document.getElementById("audio-progress");

    var audio = new Audio(doc.url);
    audio.preload = "auto";
    audio.loop    = false;

    audio.addEventListener("contextmenu", function (e) {{ e.preventDefault(); }});

    var progressIv = null;

    function beginPlayback() {{
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
            '<button id="audio-start-btn" style="font-size:1em;padding:8px 24px;">▶ Start Audio</button>';
        document.getElementById("audio-start-btn").addEventListener("click", function () {{
            document.getElementById("audio-start-btn").disabled = true;
            beginPlayback();
        }});
    }}, {{ once: true }});

    audio.addEventListener("ended", function () {{
        if (progressIv) {{ clearInterval(progressIv); }}
        statusEl.textContent  = "Audio completed. Click Next to continue.";
        progressEl.textContent = "";
        qthis.showNextButton();
        // Audio is finished and not looped -- no replay possible without page reload.
    }});

    audio.addEventListener("error", function () {{
        if (progressIv) {{ clearInterval(progressIv); }}
        statusEl.textContent =
            "Audio failed to load. Please return this study on Prolific.";
    }});
}});

Qualtrics.SurveyEngine.addOnUnload(function () {{
    // No-op; cleanup handled in ended/error handlers above.
}});"""

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
    """Dual HSLIDER matching v2 format: Passage + Questions on 0-10 scale."""
    return {"Element": "SQ", "Payload": {
        "QuestionText": text, "DefaultChoices": False,
        "DataExportTag": export_tag,
        "QuestionID": q_id,
        "QuestionType": "Slider", "Selector": "HSLIDER",
        "DataVisibility": {"Private": False, "Hidden": False},
        "Configuration": {
            "QuestionDescriptionOption": "UseText",
            "TextPosition": "above",
            "CSSliderMin": 0, "CSSliderMax": 10,
            "GridLines": 10, "NumDecimals": "0",
            "ShowValue": True, "StarCount": 5, "StarType": "discrete",
            "SnapToGrid": False, "CustomStart": False,
            "NotApplicable": False, "MobileFirst": True,
        },
        "QuestionDescription": text,
        "Choices": {"1": {"Display": "Passage"}, "2": {"Display": "Questions"}},
        "ChoiceOrder": [1, 2],
        "Validation": {"Settings": {"ForceResponse": "ON", "Type": "None", "ForceResponseType": "ON"}},
        "GradingData": [], "Language": [],
        "NextChoiceId": 3, "NextAnswerId": 6,
        "Labels": {"1": {"Display": "Easy"}, "2": {"Display": "Difficult"}},
        "ChoiceDataExportTags": False,
    }}


def make_text_question(q_id, export_tag):
    text = (
        "Is there anything about the study you found notable or wanted to let us know about? "
        "For example: comments about the passage and/or narration, how challenging the "
        "questions were, any distractions or technical issues, or anything else."
    )
    desc = text[:100] + "..."
    return {"Element": "SQ", "Payload": {
        "QuestionText": text,
        "QuestionType": "TE", "Selector": "ESTB",
        "QuestionDescription": desc,
        "Validation": {"Settings": {"Type": "None", "ForceResponse": "OFF"}},
        "Language": [],
        "SearchSource": {"AllowFreeResponse": "false"},
        "DataVisibility": {"Private": False, "Hidden": False},
        "Configuration": {"QuestionDescriptionOption": "UseText"},
        "NextChoiceId": 1, "NextAnswerId": 1,
        "DataExportTag": export_tag,
        "QuestionID": q_id,
    }}


# ---------------------------------------------------------------------------
# Per-topic generation
# ---------------------------------------------------------------------------

def generate_topic_elements(topic, content_dir, qoff):
    """
    Returns (new_sq_elements, passage_qids, qs_qids, feedback_qids)
    where passage_qids = [timing_qid, audio_qid]
          qs_qids     = [timing_qid, *content_qids, *at_qids]
          feedback_qids = [slider_qid, text_qid]
    """
    qyaml = content_dir / topic / "questions.yaml"
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
    slider_qid   = f"QID{qoff + 50}"
    text_qid     = f"QID{qoff + 51}"

    # Advanced randomization for qs block
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

    sq_elements = [
        make_timing_question(passage_timing_qid, f"{topic}_timing_passage"),
        make_audio_question(passage_audio_qid, topic, f"{topic}_trial"),
        make_timing_question(qs_timing_qid, f"{topic}_timing_qs"),
    ]
    for i, q in enumerate(content_qs):
        sq_elements.append(make_mc_question(content_qids[i], q, f"{topic}_{q['q_id']}"))
    for i, q in enumerate(at_qs):
        sq_elements.append(make_mc_question(at_qids[i], q, f"{topic}_{q['q_id']}"))
    sq_elements.append(make_slider_question(slider_qid,
        "How difficult did you find the passage and questions?",
        f"{topic}_difficulty"))
    sq_elements.append(make_text_question(text_qid, f"{topic}_comments"))

    passage_qids  = [passage_timing_qid, passage_audio_qid]
    qs_qids       = [qs_timing_qid] + content_qids + at_qids
    feedback_qids = [slider_qid, text_qid]

    return sq_elements, passage_qids, qs_qids, feedback_qids, content_qids, at_qids, fixed_order


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_old_topic_qids(bl_payload):
    """Return set of all QIDs referenced by topic blocks in BL."""
    old_qids = set()
    for v in bl_payload.values():
        desc = v.get("Description", "")
        is_topic = any(
            kw in desc
            for kw in ["birds", "fruits", "astronomy", "instruments"]
        )
        if is_topic:
            for elem in v.get("BlockElements", []):
                qid = elem.get("QuestionID")
                if qid:
                    old_qids.add(qid)
    return old_qids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(DEFAULT_BASE))
    parser.add_argument("--content-dir", default=str(DEFAULT_CONTENT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    base_path    = Path(args.base)
    content_dir  = Path(args.content_dir)
    out_path     = Path(args.output)

    print(f"Base:    {base_path}")
    print(f"Content: {content_dir}")
    print(f"Output:  {out_path}")

    with open(base_path) as f:
        qsf = json.load(f)

    bl_elem = next(e for e in qsf["SurveyElements"] if e["Element"] == "BL")
    bl_payload = bl_elem["Payload"]

    # Collect old topic QIDs to remove
    old_topic_qids = get_old_topic_qids(bl_payload)
    print(f"\nRemoving {len(old_topic_qids)} old topic SQ elements")

    # Remove old topic SQ elements
    qsf["SurveyElements"] = [
        e for e in qsf["SurveyElements"]
        if not (e.get("Element") == "SQ" and
                e.get("Payload", {}).get("QuestionID") in old_topic_qids)
    ]

    # Generate new topic content and update BL blocks
    survey_id = qsf["SurveyEntry"]["SurveyID"]
    new_sq_elements = []

    for topic in TOPICS:
        qoff = TOPIC_QID_OFFSET[topic]
        sq_elems, passage_qids, qs_qids, feedback_qids, content_qids, at_qids, fixed_order = \
            generate_topic_elements(topic, content_dir, qoff)

        # Stamp SurveyID on new SQ elements (Qualtrics requires it)
        for sq in sq_elems:
            payload = sq["Payload"]
            qid_val = payload.get("QuestionID", "")
            qtext = payload.get("QuestionText", "")
            sq["SurveyID"] = survey_id
            sq["PrimaryAttribute"] = qid_val
            sq["SecondaryAttribute"] = qtext[:200] if qtext else None
            sq["TertiaryAttribute"] = None

        new_sq_elements.extend(sq_elems)

        # Update BL blocks
        block_ids = TOPIC_BLOCK_IDS[topic]
        for k, v in bl_payload.items():
            bid = v.get("ID")
            if bid == block_ids["passage"]:
                v["BlockElements"] = [{"Type": "Question", "QuestionID": q} for q in passage_qids]
            elif bid == block_ids["qs"]:
                v["BlockElements"] = [{"Type": "Question", "QuestionID": q} for q in qs_qids]
                v["Options"]["RandomizeQuestions"] = "Advanced"
                v["Options"]["Randomization"] = {
                    "Advanced": {
                        "FixedOrder": fixed_order,
                        "RandomizeAll": content_qids,
                        "RandomSubSet": [], "Undisplayed": [],
                        "TotalRandSubset": 0, "QuestionsPerPage": 0,
                    },
                    "EvenPresentation": False,
                }
            elif bid == block_ids["feedback"]:
                v["BlockElements"] = [{"Type": "Question", "QuestionID": q} for q in feedback_qids]

        n_content = len([q for q in sq_elems if q["Payload"].get("QuestionType") == "MC"
                         and "timing" not in q["Payload"].get("DataExportTag", "")])
        print(f"  {topic}: {len(sq_elems)} SQ elements, {n_content} questions")

    qsf["SurveyElements"].extend(new_sq_elements)

    # Validate: all block-referenced QIDs exist as SQ elements
    all_sq_qids = {e["Payload"]["QuestionID"] for e in qsf["SurveyElements"]
                   if e.get("Element") == "SQ" and "QuestionID" in e.get("Payload", {})}
    missing = []
    for v in bl_payload.values():
        for elem in v.get("BlockElements", []):
            qid = elem.get("QuestionID")
            if qid and qid not in all_sq_qids:
                missing.append((v.get("Description", "?"), qid))
    if missing:
        print(f"\nWARNING: {len(missing)} QIDs in blocks have no SQ: {missing[:5]}", file=sys.stderr)
        sys.exit(1)

    # Validate: no old topic QIDs remain in SQ elements or block elements
    remaining_sq_qids = {e["Payload"]["QuestionID"] for e in qsf["SurveyElements"]
                         if e.get("Element") == "SQ" and "QuestionID" in e.get("Payload", {})}
    block_referenced = set()
    for v in bl_payload.values():
        for elem in v.get("BlockElements", []):
            qid = elem.get("QuestionID")
            if qid:
                block_referenced.add(qid)
    all_referenced = remaining_sq_qids | block_referenced
    leftover = old_topic_qids & all_referenced
    if leftover:
        print(f"\nWARNING: {len(leftover)} old QIDs still in output: {sorted(leftover)[:5]}", file=sys.stderr)
        sys.exit(1)

    qsf["SurveyEntry"]["SurveyName"] = "Multi-Topic Listening Study v3"

    out_path.write_text(json.dumps(qsf, indent=2, ensure_ascii=True))
    print(f"\nSaved: {out_path}")
    sq_count = sum(1 for e in qsf["SurveyElements"] if e.get("Element") == "SQ")
    print(f"Total SQ elements: {sq_count}")
    print("Validation: OK")


if __name__ == "__main__":
    main()
