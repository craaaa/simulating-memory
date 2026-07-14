#!/usr/bin/env python3
"""
Inject 24-group Latin-square condition assignment into a Qualtrics QSF.

Each group is one permutation of (control, repeat_short, repeat_long, distractor)
across the 4 topics. Group assignment comes from a `group` URL parameter (1-24),
which Prolific passes per participant.

Usage:
    python generate_condition_flow.py survey.qsf --topics birds fruits astronomy musical_instruments
    python generate_condition_flow.py survey.qsf --topics birds fruits astronomy musical_instruments --output modified.qsf

The script prepends to the existing Survey Flow:
  1. Embedded Data block to capture `group` from URL
  2. 24 Branch elements — each sets {topic}_cond for each topic

Also prints a group-assignment table and a participant URL parameter CSV.
"""

import json
import itertools
import argparse
import sys
from pathlib import Path

CONDITIONS = ["control", "repeat_short", "repeat_long", "distractor"]


def find_max_flow_id(flow_element):
    """Recursively find the highest FL_N integer in existing flow."""
    max_id = 0
    if isinstance(flow_element, dict):
        fid = flow_element.get("FlowID", "")
        if isinstance(fid, str) and fid.startswith("FL_"):
            try:
                max_id = max(max_id, int(fid[3:]))
            except ValueError:
                pass
        for v in flow_element.values():
            max_id = max(max_id, find_max_flow_id(v))
    elif isinstance(flow_element, list):
        for item in flow_element:
            max_id = max(max_id, find_max_flow_id(item))
    return max_id


def make_embedded_data_element(fl_id, fields):
    """
    fields: list of (field_name, value_or_None)
    value=None means read from URL / prior context (leave Value absent).
    """
    ed_list = []
    for name, value in fields:
        # URL parameters use "Recipient"; branch-set variables use "Custom"
        entry = {
            "Description": name,
            "Type": "Recipient" if value is None else "Custom",
            "Field": name,
            "VariableType": "String",
            "DataVisibility": [],
            "AnalyzeText": False,
        }
        if value is not None:
            entry["Value"] = value
        ed_list.append(entry)
    return {
        "Type": "EmbeddedData",
        "FlowID": fl_id,
        "EmbeddedData": ed_list,
    }


def make_branch_element(fl_id, inner_fl_id, group_num, topics, cond_perm):
    """Branch: if group == group_num, set per-topic condition fields."""
    fields = [(f"{topic}_cond", cond) for topic, cond in zip(topics, cond_perm)]
    label = ", ".join(f"{t[:3]}={c[:2]}" for t, c in zip(topics, cond_perm))
    return {
        "Type": "Branch",
        "FlowID": fl_id,
        "Description": f"Group {group_num}: {label}",
        "BranchLogic": {
            "0": {
                "0": {
                    "LogicType": "EmbeddedField",
                    "LeftOperand": "group",
                    "Operator": "EqualTo",
                    "RightOperand": str(group_num),
                    "Type": "Expression",
                    "Description": f"group Is Equal to {group_num}",
                },
                "Type": "If",
            },
            "Type": "BooleanExpression",
        },
        "Flow": [make_embedded_data_element(inner_fl_id, fields)],
    }


def generate_condition_elements(topics, start_fl):
    """Return (list_of_flow_elements, next_available_fl_int)."""
    fl = start_fl
    elements = []

    # Element 1: capture `group` from URL
    elements.append(make_embedded_data_element(f"FL_{fl}", [("group", None)]))
    fl += 1

    # Elements 2-25: 24 branch elements
    perms = list(itertools.permutations(CONDITIONS))
    for i, perm in enumerate(perms, 1):
        branch_fl = f"FL_{fl}"; fl += 1
        inner_fl  = f"FL_{fl}"; fl += 1
        elements.append(make_branch_element(branch_fl, inner_fl, i, topics, perm))

    return elements, fl


def print_assignment_table(topics, perms):
    header = f"{'Group':>6}  " + "  ".join(f"{t:<20}" for t in topics)
    print(header)
    print("-" * len(header))
    for i, perm in enumerate(perms, 1):
        row = f"{i:>6}  " + "  ".join(f"{c:<20}" for c in perm)
        print(row)


def write_participant_csv(output_path, n_participants, perms):
    """Write a CSV of group assignments cycling through 1-24."""
    lines = ["participant_id,group"]
    for idx in range(n_participants):
        group = (idx % len(perms)) + 1
        lines.append(f"P{idx+1:04d},{group}")
    csv_path = output_path.with_name(output_path.stem + "_groups.csv")
    csv_path.write_text("\n".join(lines) + "\n")
    print(f"Group assignment CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("qsf", help="Existing Qualtrics QSF file")
    parser.add_argument("--topics", nargs=4, required=True, metavar="TOPIC",
                        help="4 topic names in survey order (e.g. birds fruits astronomy musical_instruments)")
    parser.add_argument("--output", help="Output QSF path (default: <input>_conditions.qsf)")
    parser.add_argument("--participants", type=int, default=96,
                        help="Number of participants for group CSV (default: 96)")
    args = parser.parse_args()

    qsf_path = Path(args.qsf)
    if not qsf_path.exists():
        print(f"Error: {qsf_path} not found", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output) if args.output else qsf_path.with_name(qsf_path.stem + "_conditions.qsf")

    with open(qsf_path) as f:
        qsf = json.load(f)

    # Find the SurveyFlow element
    survey_flow_elem = None
    for elem in qsf.get("SurveyElements", []):
        if elem.get("Element") == "FL":
            survey_flow_elem = elem
            break

    if not survey_flow_elem:
        print("Error: no SurveyFlow (FL element) found in QSF", file=sys.stderr)
        sys.exit(1)

    payload = survey_flow_elem["Payload"]
    existing_flow = payload.get("Flow", [])

    # Find safe starting FlowID
    max_id = find_max_flow_id(qsf)
    start_fl = max_id + 1

    new_elements, _ = generate_condition_elements(args.topics, start_fl)

    payload["Flow"] = new_elements + existing_flow

    with open(out_path, "w") as f:
        json.dump(qsf, f, indent=2)

    perms = list(itertools.permutations(CONDITIONS))

    print(f"\nInjected {len(new_elements)} flow elements (1 EmbeddedData + 24 Branches).")
    print(f"Saved: {out_path}\n")
    print("Group assignments:")
    print_assignment_table(args.topics, perms)

    write_participant_csv(out_path, args.participants, perms)

    print(f"\nProlific URL tip: append ?group=<1-24> to your survey link.")
    print(f"Or use the _groups.csv to assign groups when distributing via participant list.")


if __name__ == "__main__":
    main()
