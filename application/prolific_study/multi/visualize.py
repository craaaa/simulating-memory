#!/usr/bin/env python3
"""
Visualize study texts with sentence-overlap highlighting and questions.

Usage:
    python visualize.py <topic_dir>
    python visualize.py birds
    python visualize.py board_games --output review.html

Reads:
    <topic_dir>/texts/{control,repeat_short,repeat_long,distractor}.md
    <topic_dir>/audio/{control,repeat_short,repeat_long,distractor}.mp3  (optional)
    <topic_dir>/questions.yaml  (optional)

Outputs:
    <topic_dir>/visualize.html  (or --output path)
"""

import sys
import re
import argparse
from pathlib import Path
from difflib import SequenceMatcher

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

CONDITIONS = ["control", "repeat_short", "repeat_long", "distractor"]
COND_LABEL = {
    "control": "Control",
    "repeat_short": "Repeat Short",
    "repeat_long": "Repeat Long",
    "distractor": "Distractor",
}
COND_SHORT = {
    "control": "C",
    "repeat_short": "RS",
    "repeat_long": "RL",
    "distractor": "D",
}
# background color per frozenset of conditions sharing a sentence
SUBSET_BG = {
    frozenset(["control", "repeat_short", "repeat_long", "distractor"]): ("#fde68a", "All 4 conditions"),
    frozenset(["control", "repeat_long", "distractor"]): ("#bfdbfe", "C + RL + D"),
    frozenset(["control", "repeat_short", "repeat_long"]): ("#fcd34d", "C + RS + RL"),
    frozenset(["repeat_short", "repeat_long"]): ("#fdba74", "RS + RL (elaboration)"),
    frozenset(["control", "repeat_long"]): ("#a5f3fc", "C + RL only"),
    frozenset(["control", "distractor"]): ("#c7d2fe", "C + D only"),
    frozenset(["repeat_long"]): ("#fed7aa", "RL only"),
    frozenset(["distractor"]): ("#e9d5ff", "D only"),
    frozenset(["control"]): ("#f1f5f9", "C only"),
    frozenset(["repeat_short"]): ("#fef9c3", "RS only"),
}
DEFAULT_BG = ("#f8fafc", "unique")


def split_sentences(text):
    """Split text into (paragraph_idx, sentence) pairs."""
    results = []
    paragraphs = re.split(r'\n\n+', text.strip())
    for p_idx, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            continue
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"])', para)
        for part in parts:
            s = part.strip()
            if s:
                results.append((p_idx, s))
    return results


def render_plain_text(text):
    """Render plain text as paragraphs (no overlap highlighting)."""
    paragraphs = re.split(r'\n\n+', text.strip())
    return "".join(
        f'<p class="plain-para">{p.strip()}</p>'
        for p in paragraphs if p.strip()
    )


def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def cluster_sentences(all_sents, threshold=0.82):
    """
    all_sents: list of (condition, para_idx, sentence_text)
    Returns: list of clusters, each cluster = list of (condition, para_idx, sentence_text)
    Also returns: for each (condition, para_idx, sent_text) -> cluster_id
    """
    clusters = []
    mapping = {}  # (cond, para, text) -> cluster_id

    for item in all_sents:
        cond, para, text = item
        best_cluster = None
        best_sim = threshold
        for ci, cluster in enumerate(clusters):
            rep_text = cluster[0][2]
            s = similarity(text, rep_text)
            if s > best_sim:
                best_sim = s
                best_cluster = ci
        if best_cluster is None:
            clusters.append([item])
            mapping[item] = len(clusters) - 1
        else:
            clusters[best_cluster].append(item)
            mapping[item] = best_cluster

    return clusters, mapping


def load_questions(topic_dir):
    qfile = topic_dir / "questions.yaml"
    if not qfile.exists() or not HAS_YAML:
        return None
    with open(qfile) as f:
        return yaml.safe_load(f)


def render_questions_html(qdata):
    """Render questions HTML. Correct answers always marked; visibility toggled by CSS class."""
    if not qdata or "questions" not in qdata:
        return "<p><em>No questions file found.</em></p>"

    parts = []
    for q in qdata["questions"]:
        qid = q.get("q_id", "")
        qtype = q.get("metadata", {}).get("type", "content")
        answers = set(q.get("answer", []))
        label_class = "attn-check" if qtype == "attention_check" else "content-q"

        parts.append(f'<div class="question {label_class}">')
        parts.append(f'<p class="q-id">{qid} <span class="q-type">({qtype})</span></p>')
        parts.append(f'<p class="q-text">{q["question"]}</p>')
        parts.append('<ol class="options">')
        for opt_num, opt_text in q["options"].items():
            correct = opt_num in answers
            if correct:
                parts.append(
                    f'<li class="correct">{opt_text}'
                    f'<span class="answer-mark"> ✓</span></li>'
                )
            else:
                parts.append(f'<li>{opt_text}</li>')
        parts.append('</ol>')
        if "notes" in q.get("metadata", {}):
            parts.append(f'<p class="notes">{q["metadata"]["notes"]}</p>')
        parts.append('</div>')
    return "\n".join(parts)


def build_html(topic_name, texts, clusters, mapping, qdata, topic_dir, first_cond="control"):
    # Render highlighted text per condition (experimenter view)
    cond_highlighted = {}
    for cond in CONDITIONS:
        if cond not in texts:
            continue
        rendered = []
        prev_para = None
        for (c, para, sent_text) in [(cond, p, s) for p, s in split_sentences(texts[cond])]:
            key = (c, para, sent_text)
            cid = mapping.get(key)
            if cid is not None:
                cluster = clusters[cid]
                cond_set = frozenset(item[0] for item in cluster)
            else:
                cond_set = frozenset([cond])
            bg, label = SUBSET_BG.get(cond_set, DEFAULT_BG)
            badges = "".join(
                f'<span class="badge badge-{cc}">{COND_SHORT[cc]}</span>'
                for cc in CONDITIONS if cc in cond_set and cc != cond
            )
            if prev_para is not None and para != prev_para:
                rendered.append('<div class="para-break"></div>')
            rendered.append(
                f'<span class="sent" style="background:{bg}" title="shared with: {label}">'
                f'{sent_text} {badges}</span> '
            )
            prev_para = para
        cond_highlighted[cond] = "".join(rendered)

    # Render plain text per condition (participant view)
    cond_plain = {cond: render_plain_text(text) for cond, text in texts.items()}

    # Check for audio files
    audio_dir = topic_dir / "audio"
    cond_audio = {}
    for cond in CONDITIONS:
        mp3 = audio_dir / f"{cond}.mp3"
        if mp3.exists():
            cond_audio[cond] = f"audio/{cond}.mp3"

    # Parallel view
    anchor_cond = "repeat_long" if "repeat_long" in texts else "control"
    seen_clusters = set()
    ordered_clusters = []
    for p, s in split_sentences(texts.get(anchor_cond, "")):
        key = (anchor_cond, p, s)
        cid = mapping.get(key)
        if cid is not None and cid not in seen_clusters:
            seen_clusters.add(cid)
            ordered_clusters.append(cid)
    for ci in range(len(clusters)):
        if ci not in seen_clusters:
            ordered_clusters.append(ci)

    parallel_rows = []
    for ci in ordered_clusters:
        cluster = clusters[ci]
        cond_set = frozenset(item[0] for item in cluster)
        bg, label = SUBSET_BG.get(cond_set, DEFAULT_BG)
        row_cells = []
        cond_texts = {}
        for cond in CONDITIONS:
            cond_items = [s for c, p, s in cluster if c == cond]
            if cond_items:
                cell_text = max(cond_items, key=len)
                cond_texts[cond] = cell_text
                row_cells.append(f'<td style="background:{bg}">{cell_text}</td>')
            else:
                row_cells.append('<td class="absent">—</td>')
        # Verbatim column: compare all present sentences
        present_texts = list(cond_texts.values())
        if len(present_texts) <= 1:
            verbatim_cell = '<td class="verbatim-na">—</td>'
        elif len(set(present_texts)) == 1:
            verbatim_cell = '<td class="verbatim-yes" title="Identical across all conditions">✓</td>'
        else:
            verbatim_cell = '<td class="verbatim-no" title="Paraphrased across conditions">≈</td>'
        parallel_rows.append(f'<tr>{"".join(row_cells)}{verbatim_cell}</tr>')

    q_html = render_questions_html(qdata)

    # Legend
    legend_items = []
    for cond_set, (bg, label) in SUBSET_BG.items():
        if len(cond_set) > 1:
            badges = "".join(
                '<span class="badge badge-' + cc + '">' + COND_SHORT[cc] + '</span>'
                for cc in CONDITIONS if cc in cond_set
            )
            legend_items.append(
                '<div class="legend-item">'
                '<span class="legend-swatch" style="background:' + bg + '"></span>'
                + badges +
                ' <span class="legend-label">' + label + '</span></div>'
            )
    legend_html = "\n".join(legend_items)

    # Build per-condition tab panels
    tab_panels_parts = []
    for c in CONDITIONS:
        if c not in texts:
            continue

        audio_tag = ""
        if c in cond_audio:
            audio_tag = f'<audio controls class="audio-player" src="{cond_audio[c]}">Your browser does not support audio.</audio>'

        plain_text = cond_plain.get(c, "<em>No text file found.</em>")
        highlighted_text = cond_highlighted.get(c, "<em>No text file found.</em>")

        panel = (
            f'<div id="tab-{c}" class="tab-panel">\n'
            f'<h2>{COND_LABEL[c]}</h2>\n'
            # Participant view
            f'<div class="pview">\n'
            f'{audio_tag}\n'
            f'<div class="plain-text">{plain_text}</div>\n'
            f'<details class="questions-block">\n'
            f'  <summary>Questions <button class="toggle-answers-btn" onclick="toggleAnswers(event, this)">Show Answers</button></summary>\n'
            f'  <div class="questions-wrap answers-hidden">\n'
            f'    {q_html}\n'
            f'  </div>\n'
            f'</details>\n'
            f'</div>\n'
            # Experimenter view
            f'<div class="eview" style="display:none">\n'
            f'<div class="text-body">{highlighted_text}</div>\n'
            f'<details class="questions-block" open>\n'
            f'  <summary>Questions</summary>\n'
            f'  <div class="questions-wrap">\n'
            f'    {q_html}\n'
            f'  </div>\n'
            f'</details>\n'
            f'</div>\n'
            f'</div>'
        )
        tab_panels_parts.append(panel)

    tab_panels = "\n".join(tab_panels_parts)

    tab_buttons = "\n".join(
        '<button class="tab-btn" onclick=\'showTab("' + c + '")\'>' + COND_LABEL[c] + "</button>"
        for c in CONDITIONS if c in texts
    )
    table_headers = "".join("<th>" + COND_LABEL[c] + "</th>" for c in CONDITIONS) + "<th>Verbatim?</th>"
    parallel_body = "\n".join(parallel_rows)
    topic_title = topic_name.replace("_", " ").title()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Visualizer — {topic_title}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: Georgia, serif; margin: 0; padding: 16px; background: #f9fafb; color: #111; }}
  h1 {{ font-size: 1.5rem; margin-bottom: 8px; }}
  h2 {{ font-size: 1.2rem; margin: 12px 0 6px; }}
  h3 {{ font-size: 1.1rem; margin: 10px 0 4px; }}

  /* View toggle */
  .view-toggle {{ display: flex; align-items: center; gap: 8px; margin-bottom: 12px; }}
  .view-btn {{
    padding: 5px 14px; border: 1px solid #94a3b8; border-radius: 20px;
    background: #f1f5f9; cursor: pointer; font-size: 0.88rem;
  }}
  .view-btn.active {{ background: #1e40af; color: #fff; border-color: #1e40af; font-weight: bold; }}
  .view-label {{ font-size: 0.85rem; color: #64748b; margin-left: 4px; }}

  /* Legend */
  .legend-wrap {{ margin-bottom: 10px; }}
  .legend-toggle-btn {{
    padding: 4px 10px; font-size: 0.8rem; border: 1px solid #cbd5e1;
    border-radius: 4px; background: #f8fafc; cursor: pointer; margin-bottom: 6px;
  }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 10px; padding: 10px 14px;
             background: #fff; border: 1px solid #e2e8f0; border-radius: 6px; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 0.82rem; }}
  .legend-swatch {{ width: 18px; height: 18px; border-radius: 3px; border: 1px solid #ccc; flex-shrink: 0; }}
  .legend-label {{ color: #555; }}

  /* Tabs */
  .tabs {{ display: flex; gap: 4px; margin-bottom: 0; flex-wrap: wrap; }}
  .tab-btn {{
    padding: 6px 14px; border: 1px solid #cbd5e1; border-bottom: none;
    background: #e2e8f0; cursor: pointer; border-radius: 4px 4px 0 0; font-size: 0.9rem;
  }}
  .tab-btn.active {{ background: #fff; font-weight: bold; }}
  .tab-panel {{ display: none; border: 1px solid #cbd5e1; border-radius: 0 4px 4px 4px; background: #fff; padding: 16px; }}
  .tab-panel.active {{ display: block; }}

  /* Highlighted text (experimenter) */
  .sent {{ border-radius: 3px; padding: 1px 2px; line-height: 2; }}
  .para-break {{ display: block; height: 12px; }}
  .badge {{ font-size: 0.65rem; font-family: monospace; font-weight: bold;
            padding: 1px 4px; border-radius: 3px; margin-left: 2px; vertical-align: middle; }}
  .badge-control {{ background: #dbeafe; color: #1e40af; }}
  .badge-repeat_short {{ background: #fef3c7; color: #92400e; }}
  .badge-repeat_long {{ background: #ffedd5; color: #9a3412; }}
  .badge-distractor {{ background: #ede9fe; color: #5b21b6; }}

  /* Plain text (participant) */
  .plain-text .plain-para {{ margin: 0 0 1em; line-height: 1.7; }}

  /* Audio */
  .audio-player {{ display: block; width: 100%; margin-bottom: 14px; }}

  /* Questions block */
  .questions-block {{ margin-top: 16px; border: 1px solid #e2e8f0; border-radius: 6px; padding: 0; }}
  .questions-block summary {{
    padding: 10px 14px; font-weight: bold; cursor: pointer; font-size: 0.95rem;
    list-style: none; display: flex; align-items: center; gap: 10px; user-select: none;
  }}
  .questions-block summary::-webkit-details-marker {{ display: none; }}
  .questions-block summary::before {{ content: "▶"; font-size: 0.7rem; color: #64748b; }}
  .questions-block[open] summary::before {{ content: "▼"; }}
  .questions-wrap {{ padding: 8px 14px 14px; }}
  .toggle-answers-btn {{
    padding: 2px 10px; font-size: 0.8rem; border: 1px solid #94a3b8;
    border-radius: 4px; background: #f8fafc; cursor: pointer; font-weight: normal;
  }}
  .question {{ border: 1px solid #e2e8f0; border-radius: 6px; padding: 12px 16px; margin-bottom: 12px; }}
  .question.attn-check {{ border-color: #fca5a5; background: #fff7f7; }}
  .q-id {{ font-family: monospace; font-weight: bold; font-size: 0.85rem; margin: 0 0 4px; color: #6b7280; }}
  .q-type {{ font-weight: normal; font-style: italic; }}
  .q-text {{ margin: 4px 0 8px; font-weight: bold; }}
  ol.options {{ margin: 0; padding-left: 1.4em; }}
  ol.options li {{ margin-bottom: 3px; font-size: 0.92rem; }}
  ol.options li.correct {{ color: #166534; font-weight: bold; }}
  .answer-mark {{ color: #166534; }}
  .notes {{ font-size: 0.8rem; color: #6b7280; margin-top: 6px; font-style: italic; }}

  /* Hide answers via CSS class */
  .answers-hidden ol.options li.correct {{ color: inherit; font-weight: inherit; }}
  .answers-hidden .answer-mark {{ display: none; }}

  /* Parallel view */
  .parallel-wrap {{ overflow-x: auto; }}
  table.parallel {{ border-collapse: collapse; width: 100%; font-size: 0.88rem; }}
  table.parallel th {{ background: #f1f5f9; padding: 6px 8px; text-align: left;
                       border: 1px solid #e2e8f0; font-family: monospace; }}
  table.parallel td {{ padding: 6px 8px; border: 1px solid #e2e8f0; vertical-align: top; line-height: 1.5; }}
  table.parallel td.absent {{ color: #94a3b8; text-align: center; background: #f8fafc; }}
  table.parallel td.verbatim-yes {{ text-align: center; color: #166534; font-weight: bold; background: #dcfce7; }}
  table.parallel td.verbatim-no {{ text-align: center; color: #92400e; background: #fef3c7; }}
  table.parallel td.verbatim-na {{ text-align: center; color: #94a3b8; }}
</style>
</head>
<body>
<h1>Study Text Visualizer — {topic_title}</h1>

<div class="view-toggle">
  <button class="view-btn active" id="vbtn-participant" onclick="setView('participant')">Participant View</button>
  <button class="view-btn" id="vbtn-experimenter" onclick="setView('experimenter')">Experimenter View</button>
</div>

<div id="legend-wrap" class="legend-wrap" style="display:none">
  <button class="legend-toggle-btn" onclick="toggleLegend()">Show/Hide Overlap Key</button>
  <div id="legend-box" class="legend">
    <strong style="align-self:center">Overlap key:</strong>
    {legend_html}
  </div>
</div>

<div class="tabs">
  {tab_buttons}
  <button class="tab-btn" onclick='showTab("parallel")'>Parallel View</button>
</div>

{tab_panels}

<div id="tab-parallel" class="tab-panel">
  <div class="parallel-wrap">
  <table class="parallel">
    <thead><tr>{table_headers}</tr></thead>
    <tbody>{parallel_body}</tbody>
  </table>
  </div>
</div>

<script>
var currentView = 'participant';

function setView(v) {{
  currentView = v;
  document.querySelectorAll('.pview').forEach(function(el) {{
    el.style.display = v === 'participant' ? '' : 'none';
  }});
  document.querySelectorAll('.eview').forEach(function(el) {{
    el.style.display = v === 'experimenter' ? '' : 'none';
  }});
  var legendWrap = document.getElementById('legend-wrap');
  var activePanel = document.querySelector('.tab-panel.active');
  var isParallel = activePanel && activePanel.id === 'tab-parallel';
  legendWrap.style.display = (v === 'experimenter' || isParallel) ? '' : 'none';
  document.querySelectorAll('.view-btn').forEach(function(el) {{
    el.classList.remove('active');
  }});
  document.getElementById('vbtn-' + v).classList.add('active');
}}

function showTab(name) {{
  document.querySelectorAll('.tab-panel').forEach(function(el) {{ el.classList.remove('active'); }});
  document.querySelectorAll('.tab-btn').forEach(function(el) {{ el.classList.remove('active'); }});
  var panel = document.getElementById('tab-' + name);
  if (panel) panel.classList.add('active');
  document.querySelectorAll('.tab-btn').forEach(function(b) {{
    if (b.getAttribute('onclick') && b.getAttribute('onclick').indexOf(name) !== -1)
      b.classList.add('active');
  }});
  // Show legend when parallel tab is active
  var legendWrap = document.getElementById('legend-wrap');
  if (name === 'parallel') {{
    legendWrap.style.display = '';
  }} else if (currentView !== 'experimenter') {{
    legendWrap.style.display = 'none';
  }}
}}

function toggleLegend() {{
  var box = document.getElementById('legend-box');
  box.style.display = box.style.display === 'none' ? '' : 'none';
}}

function toggleAnswers(evt, btn) {{
  evt.preventDefault();
  var wrap = btn.closest('.questions-block').querySelector('.questions-wrap');
  var hidden = wrap.classList.toggle('answers-hidden');
  btn.textContent = hidden ? 'Show Answers' : 'Hide Answers';
}}

showTab("{first_cond}");
</script>
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("topic_dir", help="Path to topic directory (e.g. birds or board_games)")
    parser.add_argument("--output", help="Output HTML path (default: <topic_dir>/visualize.html)")
    parser.add_argument("--threshold", type=float, default=0.82, help="Sentence similarity threshold (default 0.82)")
    args = parser.parse_args()

    topic_dir = Path(args.topic_dir)
    if not topic_dir.is_absolute():
        topic_dir = Path(__file__).parent / topic_dir
    if not topic_dir.exists():
        print(f"Error: directory not found: {topic_dir}", file=sys.stderr)
        sys.exit(1)

    topic_name = topic_dir.name

    # Load texts
    texts = {}
    for cond in CONDITIONS:
        p = topic_dir / "texts" / f"{cond}.md"
        if p.exists():
            texts[cond] = p.read_text()
        else:
            print(f"  [warn] missing: {p}")

    if not texts:
        print("No text files found. Generate texts first.", file=sys.stderr)
        sys.exit(1)

    # Sentence splitting and clustering
    all_sents = []
    for cond, text in texts.items():
        for para, sent in split_sentences(text):
            all_sents.append((cond, para, sent))

    clusters, mapping = cluster_sentences(all_sents, threshold=args.threshold)
    print(f"Found {len(clusters)} unique sentence clusters across {len(texts)} conditions.")

    # Load questions
    qdata = load_questions(topic_dir)

    # First available condition for default tab
    first_cond = next((c for c in CONDITIONS if c in texts), "parallel")

    html = build_html(topic_name, texts, clusters, mapping, qdata, topic_dir, first_cond)

    out_path = Path(args.output) if args.output else topic_dir / "visualize.html"
    out_path.write_text(html)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
