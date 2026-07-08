#!/usr/bin/env python3
"""
Build multi_v5.qsf from multi_v2/multi_v4.qsf + multi_v5/ topic content.

Keeps the same 4 randomized-order topics as v4 (martial_arts, fruits_v2, astronomy,
fabrics) and the same non-topic structure (conditions flow, 24 branches, consent,
audio check, debrief, instructions, BlockRandomizer). Swaps in v5's revised
texts/questions for those 4.

Adds musical_instruments back as a 5th topic (content unchanged, reused from
multi_v2/musical_instruments). It is NOT part of the 4-topic order randomizer —
it always plays last, right before the Debrief block. Its condition is assigned
independently via its own BlockRandomizer (SubSet=1 of 4 groups), rather than
through the existing 24-branch factorial used for the other 4 topics.

Also adds an EmbeddedData block near the top of the survey flow recording each
topic's correct attention-check answer(s), so the answer key travels with the
survey flow itself rather than living only in questions.yaml.

Usage:
    python build_v5_survey.py
    python build_v5_survey.py --base ../multi_v2/multi_v4.qsf --content-dir . --output multi_v5.qsf
"""

import json
import argparse
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML required: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

V5_DIR = Path(__file__).parent
MULTI_V2_DIR = V5_DIR.parent / "multi_v2"
DEFAULT_BASE = MULTI_V2_DIR / "multi_v4.qsf"
DEFAULT_CONTENT = V5_DIR
DEFAULT_OUTPUT = V5_DIR / "multi_v5.qsf"
BASE_URL = "https://craaaa.github.io/simulating-memory/multi_v5"
MUSIC_BASE_URL = "https://craaaa.github.io/simulating-memory/multi_v2"

# The 4 topics that keep the existing randomized-order / 24-branch-factorial slot
TOPICS = ["martial_arts", "fruits_v2", "astronomy", "fabrics"]

# musical_instruments is fixed-position (always last) with its own independent
# condition randomizer — handled separately from TOPICS throughout this script.
MUSIC_TOPIC = "musical_instruments"
MUSIC_QID_OFFSET = 2400
MUSIC_CONTENT_DIR = MULTI_V2_DIR  # content unchanged, reused from multi_v2/musical_instruments

CONDITIONS = ["control", "repeat_short", "repeat_long", "distractor"]

# fruits_v2's content directory was renamed to "fruits" on disk in multi_v5/
TOPIC_CONTENT_DIRNAME = {
    "martial_arts": "martial_arts",
    "fruits_v2": "fruits",
    "astronomy": "astronomy",
    "fabrics": "fabrics",
}

# Same block IDs as multi_v4.qsf — topic identity/slot in the flow is unchanged
TOPIC_BLOCK_IDS = {
    "martial_arts": {
        "passage":  "BL_W65xDI9qbbnsC7hU",
        "qs":       "BL_8iCv8qNS7eti0e2",
        "feedback": "BL_3l3HZvbfkuFcbr0",
    },
    "fruits_v2": {
        "passage":  "BL_mv7lLyv86Ur6nq6a",
        "qs":       "BL_23Q5vwV4c8Qhjj8",
        "feedback": "BL_eFdlaACmEVJFMmG",
    },
    "astronomy": {
        "passage":  "BL_PQWoxw4EVWwzZiqI",
        "qs":       "BL_bw7fxbJW7IkF3GS",
        "feedback": "BL_exp7Y2hLES2qlds",
    },
    "fabrics": {
        "passage":  "BL_V1mkkrfSJWqHBI40",
        "qs":       "BL_2aAYirFCVeDDo6q",
        "feedback": "BL_9XM2zWkUnyoqR2C",
    },
}

TOPIC_QID_OFFSET = {
    "martial_arts": 3000,
    "fruits_v2":    3100,
    "astronomy":    2200,
    "fabrics":      2300,
}


# ---------------------------------------------------------------------------
# Question builders (identical to build_v4_survey.py)
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


def make_audio_question(q_id, topic, export_tag, base_url=BASE_URL):
    cond_field = f"{topic}_cond"
    conditions = ["control", "repeat_short", "repeat_long", "distractor"]
    url_lines = "\n".join(
        f'        "{c}":      {{ "url": "{base_url}/{topic}/audio/{c}.mp3" }},'
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
\t\tqthis.hideNextButton();
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

def generate_topic_elements(topic, content_dir, qoff, base_url=BASE_URL):
    dirname = TOPIC_CONTENT_DIRNAME.get(topic, topic)
    qyaml = content_dir / dirname / "questions.yaml"
    with open(qyaml) as f:
        qdata = yaml.safe_load(f)
    questions = qdata["questions"]
    content_qs = [q for q in questions if q["metadata"]["type"] == "content"]
    at_qs = [q for q in questions if q["metadata"]["type"] == "attention_check"]

    passage_timing_qid = f"QID{qoff + 1}"
    passage_audio_qid  = f"QID{qoff + 2}"
    qs_timing_qid      = f"QID{qoff + 10}"
    content_qids = [f"QID{qoff + 11 + i}" for i in range(len(content_qs))]
    at_qids      = [f"QID{qoff + 11 + len(content_qs) + i}" for i in range(len(at_qs))]
    slider_qid   = f"QID{qoff + 50}"
    text_qid     = f"QID{qoff + 51}"

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
        make_audio_question(passage_audio_qid, topic, f"{topic}_trial", base_url=base_url),
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

    # Correct attention-check answer text(s), joined with a comma (matches Qualtrics'
    # multi-select export format), for the new AT-answer-key embedded data field.
    at_correct_texts = []
    for q in at_qs:
        ans = q["answer"] if isinstance(q["answer"], list) else [q["answer"]]
        at_correct_texts.append(",".join(q["options"][a] for a in ans))
    at_correct = at_correct_texts[0] if at_correct_texts else ""

    return (sq_elements, passage_qids, qs_qids, feedback_qids,
            content_qids, at_qids, fixed_order, at_correct)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_old_topic_qids(bl_payload):
    old_qids = set()
    for v in bl_payload.values():
        desc = v.get("Description", "")
        is_topic = any(kw in desc for kw in TOPICS + [MUSIC_TOPIC])
        if is_topic:
            for elem in v.get("BlockElements", []):
                qid = elem.get("QuestionID")
                if qid:
                    old_qids.add(qid)
    return old_qids


def add_music_blocks(bl_payload, passage_qids, qs_qids, feedback_qids, content_qids, fixed_order):
    """Create fresh BL block entries for musical_instruments (not present in the
    multi_v4 base, since v4 dropped this topic entirely). Returns the 3 new block IDs."""
    next_key = max(int(k) for k in bl_payload.keys()) + 1
    ids = {
        "passage":  "BL_musinstr_passage",
        "qs":       "BL_musinstr_qs",
        "feedback": "BL_musinstr_feedback",
    }
    bl_payload[str(next_key)] = {
        "Type": "Standard", "SubType": "", "Description": "musical_instruments_passage",
        "ID": ids["passage"],
        "BlockElements": [{"Type": "Question", "QuestionID": q} for q in passage_qids],
        "Options": {"BlockLocking": "false", "RandomizeQuestions": "false", "BlockVisibility": "Expanded"},
    }
    bl_payload[str(next_key + 1)] = {
        "Type": "Standard", "SubType": "", "Description": "musical_instruments_qs",
        "ID": ids["qs"],
        "BlockElements": [{"Type": "Question", "QuestionID": q} for q in qs_qids],
        "Options": {
            "BlockLocking": "false", "RandomizeQuestions": "Advanced", "BlockVisibility": "Expanded",
            "Randomization": {
                "Advanced": {
                    "FixedOrder": fixed_order, "RandomizeAll": content_qids,
                    "RandomSubSet": [], "Undisplayed": [],
                    "TotalRandSubset": 0, "QuestionsPerPage": 0,
                },
                "EvenPresentation": False,
            },
        },
    }
    bl_payload[str(next_key + 2)] = {
        "Type": "Standard", "SubType": "", "Description": "musical_instruments_feedback",
        "ID": ids["feedback"],
        "BlockElements": [{"Type": "Question", "QuestionID": q} for q in feedback_qids],
        "Options": {"BlockLocking": "false", "RandomizeQuestions": "false", "BlockVisibility": "Collapsed"},
    }
    return ids


def build_music_flow_nodes(music_block_ids, break_block_id, next_fl_id):
    """BlockRandomizer (SubSet=1 of 4) assigning musical_instruments_cond at random,
    followed by the passage/qs/feedback/break sequence, always run last (this whole
    chunk is inserted after the 4-topic BlockRandomizer, before Debrief)."""
    fl = [next_fl_id]

    def nid():
        fl[0] += 1
        return f"FL_{fl[0]}"

    cond_groups = []
    for cond in CONDITIONS:
        cond_groups.append({
            "Type": "Group", "FlowID": nid(), "Description": f"musical_instruments_{cond}",
            "Flow": [{
                "Type": "EmbeddedData", "FlowID": nid(),
                "EmbeddedData": [{
                    "Description": "musical_instruments_cond", "Type": "Custom",
                    "Field": "musical_instruments_cond", "VariableType": "String",
                    "DataVisibility": [], "AnalyzeText": False, "Value": cond,
                }],
            }],
        })

    cond_randomizer = {
        "Type": "BlockRandomizer", "FlowID": nid(),
        "SubSet": 1, "EvenPresentation": True,
        "Flow": cond_groups,
    }

    nodes = [
        cond_randomizer,
        {"Type": "Standard", "ID": music_block_ids["passage"], "FlowID": nid(), "Autofill": []},
        {"Type": "Standard", "ID": music_block_ids["qs"], "FlowID": nid(), "Autofill": []},
        {
            "Type": "EmbeddedData", "FlowID": nid(),
            "EmbeddedData": [{
                "Description": "passage_count", "Type": "Custom", "Field": "passage_count",
                "VariableType": "Scale", "DataVisibility": [],
                "Value": "${e://Field/passage_count}+1",
            }],
        },
        {"Type": "Standard", "ID": music_block_ids["feedback"], "FlowID": nid(), "Autofill": []},
        {"Type": "Standard", "ID": break_block_id, "FlowID": nid(), "Autofill": []},
    ]
    return nodes, fl[0]


def insert_at_answer_key(qsf, at_correct_by_topic):
    """Insert an EmbeddedData FL element recording each topic's correct AT answer(s),
    placed right after the existing recipient-field EmbeddedData block near the top
    of the flow, so the answer key travels with the survey flow itself."""
    fl_elem = next(e for e in qsf["SurveyElements"] if e["Element"] == "FL")
    fl_payload = fl_elem["Payload"]

    new_ed_fields = [
        {
            "Description": f"{topic}_AT_correct",
            "Type": "Custom",
            "Field": f"{topic}_AT_correct",
            "VariableType": "String",
            "DataVisibility": [],
            "AnalyzeText": False,
            "Value": at_correct_by_topic[topic],
        }
        for topic in TOPICS + [MUSIC_TOPIC]
    ]
    new_node = {
        "Type": "EmbeddedData",
        "FlowID": f"FL_{fl_payload.get('Properties', {}).get('Count', 0) + 1}",
        "EmbeddedData": new_ed_fields,
    }

    flow = fl_payload["Flow"]
    insert_at = 0
    for i, node in enumerate(flow):
        if node.get("Type") == "EmbeddedData":
            insert_at = i + 1
    flow.insert(insert_at, new_node)

    props = fl_payload.setdefault("Properties", {})
    props["Count"] = props.get("Count", len(flow)) + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(DEFAULT_BASE))
    parser.add_argument("--content-dir", default=str(DEFAULT_CONTENT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    base_path   = Path(args.base)
    content_dir = Path(args.content_dir)
    out_path    = Path(args.output)

    print(f"Base:    {base_path}")
    print(f"Content: {content_dir}")
    print(f"Output:  {out_path}")

    with open(base_path) as f:
        qsf = json.load(f)

    bl_elem = next(e for e in qsf["SurveyElements"] if e["Element"] == "BL")
    bl_payload = bl_elem["Payload"]

    old_topic_qids = get_old_topic_qids(bl_payload)
    print(f"\nRemoving {len(old_topic_qids)} old topic SQ elements")

    qsf["SurveyElements"] = [
        e for e in qsf["SurveyElements"]
        if not (e.get("Element") == "SQ" and
                e.get("Payload", {}).get("QuestionID") in old_topic_qids)
    ]

    survey_id = qsf["SurveyEntry"]["SurveyID"]
    new_sq_elements = []
    at_correct_by_topic = {}

    for topic in TOPICS:
        qoff = TOPIC_QID_OFFSET[topic]
        (sq_elems, passage_qids, qs_qids, feedback_qids,
         content_qids, at_qids, fixed_order, at_correct) = \
            generate_topic_elements(topic, content_dir, qoff)
        at_correct_by_topic[topic] = at_correct

        for sq in sq_elems:
            payload = sq["Payload"]
            qid_val = payload.get("QuestionID", "")
            qtext = payload.get("QuestionText", "")
            sq["SurveyID"] = survey_id
            sq["PrimaryAttribute"] = qid_val
            sq["SecondaryAttribute"] = qtext[:200] if qtext else None
            sq["TertiaryAttribute"] = None

        new_sq_elements.extend(sq_elems)

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

        n_mc = len([q for q in sq_elems if q["Payload"].get("QuestionType") == "MC"
                    and "timing" not in q["Payload"].get("DataExportTag", "")])
        print(f"  {topic}: {len(sq_elems)} SQ elements, {n_mc} questions, AT correct = {at_correct!r}")

    # musical_instruments: 5th topic, fixed position (always last), content reused
    # from multi_v2/musical_instruments (unchanged). Needs its own new BL blocks,
    # since multi_v4's base has none for this topic.
    (music_sq_elems, music_passage_qids, music_qs_qids, music_feedback_qids,
     music_content_qids, music_at_qids, music_fixed_order, music_at_correct) = \
        generate_topic_elements(MUSIC_TOPIC, MUSIC_CONTENT_DIR, MUSIC_QID_OFFSET, base_url=MUSIC_BASE_URL)
    at_correct_by_topic[MUSIC_TOPIC] = music_at_correct

    for sq in music_sq_elems:
        payload = sq["Payload"]
        qid_val = payload.get("QuestionID", "")
        qtext = payload.get("QuestionText", "")
        sq["SurveyID"] = survey_id
        sq["PrimaryAttribute"] = qid_val
        sq["SecondaryAttribute"] = qtext[:200] if qtext else None
        sq["TertiaryAttribute"] = None
    new_sq_elements.extend(music_sq_elems)

    n_mc_music = len([q for q in music_sq_elems if q["Payload"].get("QuestionType") == "MC"
                       and "timing" not in q["Payload"].get("DataExportTag", "")])
    print(f"  {MUSIC_TOPIC}: {len(music_sq_elems)} SQ elements, {n_mc_music} questions, "
          f"AT correct = {music_at_correct!r} (fixed position, own condition randomizer)")

    music_block_ids = add_music_blocks(
        bl_payload, music_passage_qids, music_qs_qids, music_feedback_qids,
        music_content_qids, music_fixed_order)

    qsf["SurveyElements"].extend(new_sq_elements)

    print("\nInserting attention-check answer key into survey flow")
    insert_at_answer_key(qsf, at_correct_by_topic)

    print("Splicing musical_instruments (fixed-position, own condition randomizer) into flow")
    fl_elem = next(e for e in qsf["SurveyElements"] if e["Element"] == "FL")
    fl_payload = fl_elem["Payload"]
    flow = fl_payload["Flow"]

    br_index = next(i for i, node in enumerate(flow) if node.get("Type") == "BlockRandomizer")

    def _collect_fl_ids(node, acc):
        if isinstance(node, dict):
            fid = node.get("FlowID")
            if fid and fid.startswith("FL_"):
                try:
                    acc.append(int(fid[3:]))
                except ValueError:
                    pass
            for v in node.values():
                _collect_fl_ids(v, acc)
        elif isinstance(node, list):
            for x in node:
                _collect_fl_ids(x, acc)

    all_fl_ids = []
    _collect_fl_ids(fl_payload, all_fl_ids)
    next_fl_id = max(all_fl_ids) + 1

    break_block_id = next(v["ID"] for v in bl_payload.values() if v.get("Description") == "Break")

    music_nodes, last_fl_id = build_music_flow_nodes(music_block_ids, break_block_id, next_fl_id)
    flow[br_index + 1:br_index + 1] = music_nodes

    props = fl_payload.setdefault("Properties", {})
    props["Count"] = props.get("Count", 0) + len(music_nodes) + len(CONDITIONS) * 2 + 2

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

    qsf["SurveyEntry"]["SurveyName"] = "Multi-Topic Listening Study v5"

    out_path.write_text(json.dumps(qsf, indent=2, ensure_ascii=True))
    print(f"\nSaved: {out_path}")
    sq_count = sum(1 for e in qsf["SurveyElements"] if e.get("Element") == "SQ")
    print(f"Total SQ elements: {sq_count}")
    print("Validation: OK")


if __name__ == "__main__":
    main()
