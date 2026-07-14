#!/usr/bin/env python3
"""
Generate a Qualtrics QSF for one or all study topics.

Each topic gets 3 blocks:
  1. {topic}_passage  — PageTimer + audio question (JS reads {topic}_cond embedded field)
  2. {topic}_qs       — PageTimer + content questions + AT questions (advanced randomization)
  3. {topic}_feedback — passage difficulty slider + question difficulty slider + open text

Usage:
    python generate_topic_qsf.py birds
    python generate_topic_qsf.py all
    python generate_topic_qsf.py birds --base-url https://craaaa.github.io/simulating-memory/multi
"""

import json
import argparse
import sys
from pathlib import Path

try:
    import yaml
    HAS_YAML = True
except ImportError:
    print("PyYAML required: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

MULTI_DIR = Path(__file__).parent
DEFAULT_BASE_URL = "https://craaaa.github.io/simulating-memory/multi"

# Topic → QID offset (multiples of 100 to leave room)
TOPIC_QID_OFFSET = {
    "birds":               1000,
    "fruits":              1100,
    "astronomy":           1200,
    "musical_instruments": 1300,
    "board_games":         1400,
    "books":               1500,
}
TOPIC_BL_OFFSET = {
    "birds":               10,
    "fruits":              20,
    "astronomy":           30,
    "musical_instruments": 40,
    "board_games":         50,
    "books":               60,
}

SLIDER_CHOICES = [{"Display": str(i)} for i in range(11)]


def qid(offset, n):
    return f"QID{offset + n}"


def blid(offset, n):
    return f"BL_MULTI{offset + n:04d}xxxxxxxx"


def flid(offset, n):
    return f"FL_M{offset + n:04d}"


# --------------------------------------------------------------------------
# Question builders
# --------------------------------------------------------------------------

def make_timing_question(q_id, export_tag):
    return {
        "Element": "SQ",
        "Payload": {
            "QuestionText": "Timing",
            "DefaultChoices": False,
            "DataExportTag": export_tag,
            "QuestionID": q_id,
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
        },
    }


def make_audio_question(q_id, topic, base_url, export_tag):
    cond_field = f"{topic}_cond"
    rt_field = f"{topic}_listening_rt"
    done_field = f"{topic}_listening_completed"
    conditions = ["control", "repeat_short", "repeat_long", "distractor"]
    stimuli_lines = "\n".join(
        f'        "{c}": {{ "url": "{base_url}/{topic}/audio/{c}.mp3" }},'
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
            '<button id="audio-start-btn" style="font-size:1em;padding:8px 24px;">▶ Start Audio</button>';
        document.getElementById("audio-start-btn").addEventListener("click", function () {{
            document.getElementById("audio-start-btn").disabled = true;
            beginPlayback();
        }});
    }}, {{ once: true }});

    audio.addEventListener("ended", function () {{
        if (progressIv) {{ clearInterval(progressIv); }}
        var rt = startTime ? (Date.now() - startTime) : null;
        Qualtrics.SurveyEngine.setEmbeddedData("{rt_field}", rt);
        Qualtrics.SurveyEngine.setEmbeddedData("{done_field}", "true");
        statusEl.textContent  = "Audio complete. Click Next to continue.";
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

    return {
        "Element": "SQ",
        "Payload": {
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
            "QuestionID": q_id,
            "QuestionJS": js,
        },
    }


def make_mc_question(q_id, q_entry, export_tag):
    """Multi-select (MAVR) question from a questions.yaml entry.
    Options 1-4 randomized; option 5 (None of the above) fixed last."""
    options = q_entry["options"]
    n_opts = len(options)
    none_key = str(n_opts)  # assume last option is "None of the above"

    choices = {str(k): {"Display": v} for k, v in options.items()}
    choice_order = [str(k) for k in options.keys()]

    # Randomize all except the last ("None of the above")
    rand_keys = [str(k) for k in options.keys() if str(k) != none_key]
    fixed_order = ["{~Randomized~}"] * (n_opts - 1) + [none_key]

    return {
        "Element": "SQ",
        "Payload": {
            "QuestionText": q_entry["question"],
            "DataExportTag": export_tag,
            "QuestionType": "MC",
            "Selector": "MAVR",
            "SubSelector": "TX",
            "DataVisibility": {"Private": False, "Hidden": False},
            "Configuration": {"QuestionDescriptionOption": "UseText"},
            "QuestionDescription": q_entry["question"][:100],
            "Choices": choices,
            "ChoiceOrder": choice_order,
            "Validation": {
                "Settings": {
                    "ForceResponse": "ON",
                    "ForceResponseType": "ON",
                    "Type": "None",
                }
            },
            "Language": [],
            "NextChoiceId": n_opts + 2,
            "NextAnswerId": 1,
            "QuestionID": q_id,
            "Randomization": {
                "Advanced": {
                    "FixedOrder": fixed_order,
                    "RandomSubSet": [],
                    "RandomizeAll": rand_keys,
                    "ScaleReversal": [],
                    "TotalRandSubset": 0,
                    "Undisplayed": [],
                },
                "ConsistentScaleReversal": False,
                "EvenPresentation": False,
                "TotalRandSubset": "",
                "Type": "Advanced",
            },
        },
    }


def make_slider_question(q_id, text, export_tag):
    return {
        "Element": "SQ",
        "Payload": {
            "QuestionText": text,
            "DefaultChoices": False,
            "DataExportTag": export_tag,
            "QuestionType": "SS",
            "Selector": "TA",
            "DataVisibility": {"Private": False, "Hidden": False},
            "Configuration": {"QuestionDescriptionOption": "UseText"},
            "QuestionDescription": text,
            "Validation": {
                "Settings": {
                    "ForceResponse": "ON",
                    "ForceResponseType": "ON",
                }
            },
            "GradingData": [],
            "Language": [],
            "NextChoiceId": 12,
            "NextAnswerId": 1,
            "Category": "Gauges",
            "Scale": "TenGauge",
            "Choices": SLIDER_CHOICES,
            "QuestionID": q_id,
            "Direction": "horizontal",
        },
    }


def make_text_question(q_id, export_tag):
    text = (
        "Is there anything about the study you found notable or wanted to let us know about? "
        "For example: comments about the passage and/or narration, how challenging the "
        "questions were, any distractions or technical issues, or anything else."
    )
    return {
        "Element": "SQ",
        "Payload": {
            "QuestionText": text,
            "DefaultChoices": False,
            "DataExportTag": export_tag,
            "QuestionType": "TE",
            "Selector": "ML",
            "DataVisibility": {"Private": False, "Hidden": False},
            "Configuration": {"QuestionDescriptionOption": "UseText"},
            "QuestionDescription": text[:100],
            "Validation": {
                "Settings": {"ForceResponse": "OFF", "Type": "None"}
            },
            "GradingData": [],
            "Language": [],
            "NextChoiceId": 4,
            "NextAnswerId": 1,
            "QuestionID": q_id,
        },
    }


# --------------------------------------------------------------------------
# Block builders
# --------------------------------------------------------------------------

def make_passage_block(bl_id, topic, timing_qid, audio_qid, base_url):
    """Returns (block_payload_entry, [sq_elements])"""
    block = {
        "Type": "Standard",
        "SubType": "",
        "Description": f"{topic}_passage",
        "ID": bl_id,
        "BlockElements": [
            {"Type": "Question", "QuestionID": timing_qid},
            {"Type": "Question", "QuestionID": audio_qid},
        ],
        "Options": {
            "BlockLocking": "false",
            "RandomizeQuestions": "false",
            "BlockVisibility": "Collapsed",
        },
    }
    sqs = [
        make_timing_question(timing_qid, f"{topic}_timing_passage"),
        make_audio_question(audio_qid, topic, base_url, f"{topic}_trial"),
    ]
    return block, sqs


def make_questions_block(bl_id, topic, questions, fl_offset):
    """
    questions: list of dicts from questions.yaml (content first, then ATs)
    Returns (block_payload_entry, [sq_elements], timing_qid, content_qids, at_qids)
    """
    content_qs = [q for q in questions if q["metadata"]["type"] == "content"]
    at_qs = [q for q in questions if q["metadata"]["type"] == "attention_check"]

    # Assign QIDs sequentially starting from fl_offset + 10
    qid_start = fl_offset + 10
    timing_qid = f"QID{qid_start}"
    content_qids = [f"QID{qid_start + 1 + i}" for i in range(len(content_qs))]
    at_qids = [f"QID{qid_start + 1 + len(content_qs) + i}" for i in range(len(at_qs))]

    # Build advanced randomization: timing fixed first, ATs interspersed, content randomized
    # Pattern: [timing, content×2, AT1, content×2, AT2?, content×(remainder)]
    n_content = len(content_qs)
    n_at = len(at_qs)

    fixed_order = [timing_qid]  # position 0: timing
    randomize_all = list(content_qids)

    # Place ATs at evenly spaced fixed positions among the content
    # e.g. n_content=5, n_at=1: AT after position 3 → [T, r, r, AT, r, r, r]
    # e.g. n_content=5, n_at=2: AT1 after 2, AT2 after 4 → [T, r, r, AT1, r, r, AT2, r]
    if n_at == 0:
        fixed_order += ["{~Randomized~}"] * n_content
    elif n_at == 1:
        # AT after first ~half of content
        split = n_content // 2
        fixed_order += ["{~Randomized~}"] * split
        fixed_order += [at_qids[0]]
        fixed_order += ["{~Randomized~}"] * (n_content - split)
    elif n_at == 2:
        # AT1 after first 2, AT2 after next 2
        fixed_order += ["{~Randomized~}", "{~Randomized~}"]
        fixed_order += [at_qids[0]]
        fixed_order += ["{~Randomized~}", "{~Randomized~}"]
        fixed_order += [at_qids[1]]
        fixed_order += ["{~Randomized~}"] * (n_content - 4)
    else:
        # fallback: put all ATs at end, content randomized
        fixed_order += ["{~Randomized~}"] * n_content
        fixed_order += at_qids

    all_qids = [timing_qid] + content_qids + at_qids
    block_elements = [{"Type": "Question", "QuestionID": q} for q in all_qids]

    block = {
        "Type": "Standard",
        "SubType": "",
        "Description": f"{topic}_qs",
        "ID": bl_id,
        "BlockElements": block_elements,
        "Options": {
            "BlockLocking": "false",
            "RandomizeQuestions": "Advanced",
            "Randomization": {
                "Advanced": {
                    "FixedOrder": fixed_order,
                    "RandomizeAll": randomize_all,
                    "RandomSubSet": [],
                    "Undisplayed": [],
                    "TotalRandSubset": 0,
                    "QuestionsPerPage": 0,
                },
                "EvenPresentation": False,
            },
            "BlockVisibility": "Collapsed",
        },
    }

    sqs = [make_timing_question(timing_qid, f"{topic}_timing_qs")]
    for i, q_entry in enumerate(content_qs):
        sqs.append(make_mc_question(content_qids[i], q_entry, f"{topic}_{q_entry['q_id']}"))
    for i, q_entry in enumerate(at_qs):
        sqs.append(make_mc_question(at_qids[i], q_entry, f"{topic}_{q_entry['q_id']}"))

    return block, sqs


def make_feedback_block(bl_id, topic, qid_start):
    passage_qid = f"QID{qid_start}"
    qs_qid = f"QID{qid_start + 1}"
    text_qid = f"QID{qid_start + 2}"

    block = {
        "Type": "Standard",
        "SubType": "",
        "Description": f"{topic}_feedback",
        "ID": bl_id,
        "BlockElements": [
            {"Type": "Question", "QuestionID": passage_qid},
            {"Type": "Question", "QuestionID": qs_qid},
            {"Type": "Question", "QuestionID": text_qid},
        ],
        "Options": {
            "BlockLocking": "false",
            "RandomizeQuestions": "false",
            "BlockVisibility": "Collapsed",
        },
    }
    sqs = [
        make_slider_question(
            passage_qid,
            "On a scale of 0 (easy) to 10 (difficult), how difficult did you find the passage?",
            f"{topic}_passage_difficulty",
        ),
        make_slider_question(
            qs_qid,
            "On a scale of 0 (easy) to 10 (difficult), how difficult did you find the questions?",
            f"{topic}_qs_difficulty",
        ),
        make_text_question(text_qid, f"{topic}_comments"),
    ]
    return block, sqs


# --------------------------------------------------------------------------
# QSF assembly
# --------------------------------------------------------------------------

def generate_topic_qsf(topic, base_url, output_path):
    qyaml_path = MULTI_DIR / topic / "questions.yaml"
    if not qyaml_path.exists():
        print(f"  [skip] {topic}: questions.yaml not found", file=sys.stderr)
        return

    with open(qyaml_path) as f:
        qdata = yaml.safe_load(f)

    questions = qdata.get("questions", [])
    if not questions:
        print(f"  [skip] {topic}: no questions in yaml", file=sys.stderr)
        return

    # QID offset base
    if topic not in TOPIC_QID_OFFSET:
        idx = abs(hash(topic)) % 9000 + 500
        TOPIC_QID_OFFSET[topic] = idx
    qoff = TOPIC_QID_OFFSET[topic]

    if topic not in TOPIC_BL_OFFSET:
        TOPIC_BL_OFFSET[topic] = qoff // 10
    bloff = TOPIC_BL_OFFSET[topic]

    # IDs
    passage_bl_id  = blid(bloff, 1)
    qs_bl_id       = blid(bloff, 2)
    feedback_bl_id = blid(bloff, 3)

    passage_timing_qid = f"QID{qoff + 1}"
    passage_audio_qid  = f"QID{qoff + 2}"
    # questions block uses QID{qoff+10} onwards (set inside make_questions_block)
    feedback_qid_start = qoff + 50

    passage_block, passage_sqs = make_passage_block(
        passage_bl_id, topic, passage_timing_qid, passage_audio_qid, base_url
    )
    qs_block, qs_sqs = make_questions_block(qs_bl_id, topic, questions, qoff)
    feedback_block, feedback_sqs = make_feedback_block(feedback_bl_id, topic, feedback_qid_start)

    all_sqs = passage_sqs + qs_sqs + feedback_sqs

    # Build BL payload
    bl_payload = {
        "1": {
            "Type": "Trash",
            "Description": "Trash / Unused Questions",
            "ID": f"BL_MULTI{bloff:04d}TRASH",
            "BlockElements": [],
        },
        "2": passage_block,
        "3": qs_block,
        "4": feedback_block,
    }

    # Build minimal FL (just the 3 blocks in order)
    fl_payload = {
        "Type": "Root",
        "FlowID": "FL_1",
        "Flow": [
            {"Type": "Block", "ID": passage_bl_id,  "FlowID": f"FL_M{bloff:04d}01", "Autofill": []},
            {"Type": "Block", "ID": qs_bl_id,        "FlowID": f"FL_M{bloff:04d}02", "Autofill": []},
            {"Type": "Block", "ID": feedback_bl_id,  "FlowID": f"FL_M{bloff:04d}03", "Autofill": []},
        ],
        "Properties": {"Count": 3},
        "EmbeddedData": [],
    }

    survey_id = f"SV_{topic}_generated"
    qsf = {
        "SurveyEntry": {
            "SurveyID": survey_id,
            "SurveyName": f"{topic.replace('_', ' ').title()} — Listening Study",
            "SurveyDescription": None,
            "SurveyOwnerID": "",
            "SurveyBrandID": "",
            "DivisionID": None,
            "SurveyLanguage": "EN",
            "SurveyActiveResponseSet": "",
            "SurveyStatus": "Pending",
            "SurveyStartDate": "0000-00-00 00:00:00",
            "SurveyExpirationDate": "0000-00-00 00:00:00",
            "SurveyCreationDate": "2026-06-01 00:00:00",
            "CreatorID": "",
            "LastModified": "2026-06-01 00:00:00",
            "LastAccessed": "0000-00-00 00:00:00",
            "LastActivated": "0000-00-00 00:00:00",
            "Deleted": None,
        },
        "SurveyElements": (
            [{"Element": "BL", "Payload": bl_payload}]
            + [{"Element": "FL", "Payload": fl_payload}]
            + all_sqs
        ),
    }

    output_path.write_text(json.dumps(qsf, indent=2, ensure_ascii=False))
    print(f"  {topic}: {len(all_sqs)} questions → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("topic", help='Topic name or "all"')
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL for audio files")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: topic dir)")
    args = parser.parse_args()

    topics_to_run = (
        [d.name for d in MULTI_DIR.iterdir()
         if d.is_dir() and (d / "questions.yaml").exists()]
        if args.topic == "all"
        else [args.topic]
    )

    for topic in sorted(topics_to_run):
        out_dir = Path(args.output_dir) if args.output_dir else MULTI_DIR / topic
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{topic}_blocks.qsf"
        generate_topic_qsf(topic, args.base_url, out_path)


if __name__ == "__main__":
    main()
