from __future__ import annotations

import json
from typing import Any, Dict, List, Union

from .llm import LLM
from .working_memory import MAX_KEYS, WorkingMemory

# ---------------------------------------------------------------------------
# OpenAI tool schemas
# ---------------------------------------------------------------------------

TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "write_memory",
            "description": (
                "Store a key-value memory entry. "
                "The key should be a short word or phrase that labels the concept or chunk. "
                "The value should be an abstractive summary consisting of one clause, no more than 15 words. "
                "Separate atomic facts should be placed in separate memory entries. "
                f"Maximum {MAX_KEYS} keys total — overwriting an existing key is allowed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Short word or phrase identifying the concept (e.g. 'beginning', 'characters', 'theme').",
                    },
                    "value": {
                        "type": "string",
                        "description": "Abstractive summary of the information to retain for this key.",
                    },
                },
                "required": ["key", "value"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_key",
            "description": "Remove a key and its value from working memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The key to delete.",
                    },
                },
                "required": ["key"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
]


def _dispatch_tool(wm: WorkingMemory, name: str, arguments_json: str) -> str:
    try:
        args = json.loads(arguments_json)
    except json.JSONDecodeError as e:
        return f"Error: could not parse arguments JSON: {e}"

    if name == "write_memory":
        key = args.get("key", "").strip()
        value = args.get("value", "")
        if not key:
            return "Error: key must be a non-empty string."
        return wm.write_key(key, str(value))
    elif name == "delete_key":
        key = args.get("key", "").strip()
        if not key:
            return "Error: key must be a non-empty string."
        return wm.clear_key(key)
    else:
        return f"Error: unknown tool '{name}'."


# ---------------------------------------------------------------------------
# System prompt templates (one per condition)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_HUMAN = """\
You are simulating a human participant in a psychology experiment on working memory.
You have a key-value memory store with at most {max_keys} slots — reflecting the ~4-chunk
limit of human short-term memory (Cowan, 2001).

You will receive all the material at once, as humans do when reading or listening.

Your task:
  1. Read through the entire material.
  2. Identify up to {max_keys} meaningful chunks — the chunks a person would naturally
     organize the material into (e.g. gist, key characters, main event, outcome).
  3. For each chunk, call write_memory with a short key label and an abstractive summary.
     Compress realistically — humans retain gist, not verbatim detail.
  4. Fewer than {max_keys} keys is fine if the material is simple.

Behave as a real human would: strategic, imperfect, and sensitive to what seems most important.
You will later recall ONLY from these key-value pairs."""

SYSTEM_PROMPT_HUMAN_STREAMING = """\
You are simulating a human participant in a psychology experiment on working memory.
You have a key-value memory store with at most {max_keys} slots — reflecting the ~4-chunk
limit of human short-term memory (Cowan, 2001).

The material will NOT be shown to you all at once. Instead it arrives as a sequence of
ordered segments, one at a time, as in listening to something once — you will NOT see a
segment again after you move on from it.

On each turn you will see:
  - The current contents of your {max_keys}-slot memory store.
  - The new segment to attend to.

Your task on each turn:
  1. Read the new segment.
  2. Decide, given what is already in memory and the {max_keys}-slot limit, whether to
     write a new chunk (write_memory), overwrite/update an existing one, or delete a
     less useful one to make room (delete_key). Decide now — you cannot revisit this
     segment later.
  3. Compress realistically: keep gist, not verbatim detail. Fewer than {max_keys}
     keys is fine if the material so far is simple.

Behave as a real human would: strategic, imperfect, and sensitive to what seems most
important at the time you encounter it. You will later recall ONLY from these
key-value pairs."""

CONDITION_PROMPTS = {
    "C2": SYSTEM_PROMPT_HUMAN,
    "C2-stream": SYSTEM_PROMPT_HUMAN_STREAMING,
}

RECALL_PROMPT = """\
The study phase is now over.
Your working memory currently contains:
{wm_contents}

Based ONLY on the above contents, recall as many words as you can.
Output ONLY recalled words, one per line or comma-separated.
Do NOT add explanations, numbering, or extra text."""


class WorkingMemoryAgent:
    """Unified WM agent supporting both batch (encode/recall) and turn-by-turn (step) modes."""

    def __init__(
        self,
        llm: LLM,
        condition_id: str = "C2",
        temperature: float = 0.0,
        debug: bool = False,
        system_prompt_override: str | None = None,
    ) -> None:
        self.llm = llm
        self.condition_id = condition_id
        self.temperature = temperature
        self.debug = debug
        self._system_override = system_prompt_override
        self.wm = WorkingMemory()
        self._encoding_log: Dict[str, Any] = {}
        self._messages: List[Dict[str, Any]] = []
        self._step_log: List[Dict[str, Any]] = []
        self._tool_interactions = 0
        self._tool_calls_used = 0

    def _tool_call_cap(self) -> int:
        return max(6, int(self._tool_interactions * 1.5))

    def _remaining_tool_calls(self) -> int:
        return max(0, self._tool_call_cap() - self._tool_calls_used)

    def _build_system(self) -> str:
        if self._system_override:
            return self._system_override
        template = CONDITION_PROMPTS[self.condition_id]
        return template.format(max_keys=MAX_KEYS)

    def _ensure_messages(self) -> None:
        """Lazily initialize the message list with the system prompt."""
        if not self._messages:
            self._messages = [{"role": "system", "content": self._build_system()}]

    def reset_messages(self) -> None:
        """Clear conversation history (keeps WM state)."""
        self._messages = []

    # ------------------------------------------------------------------
    # step() — one turn of agent interaction
    # ------------------------------------------------------------------

    def step(
        self, user_message: str, *, allow_tools: bool = True, max_tokens: int = 1024
    ) -> str:
        """One turn of the agent loop.

        Appends *user_message*, gets the LLM response (processing any tool
        calls in a loop until the agent stops calling tools), and returns the
        final text content.  Conversation history is maintained across calls.

        Parameters
        ----------
        user_message : str
            The next user-side message to present to the agent.
        allow_tools : bool
            If *True* the agent may call write_memory / delete_key.
        max_tokens : int
            Max completion tokens per LLM call within this step.

        Returns
        -------
        str
            The agent's text reply (may be empty if agent only made tool calls).
        """
        self._ensure_messages()
        if allow_tools:
            self._tool_interactions += 1
        self._messages.append({"role": "user", "content": user_message})

        tool_calls_log: List[Dict[str, Any]] = []
        collected_text = ""
        cap_before = self._tool_call_cap()
        budget_before = self._remaining_tool_calls()
        cap_hit = False

        while True:
            tools_available = allow_tools and self._remaining_tool_calls() > 0
            if tools_available:
                resp = self.llm.generate_with_tools(
                    messages=self._messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                )
            elif allow_tools:
                cap_hit = True
                resp = self.llm.generate_with_tools(
                    messages=self._messages,
                    tools=TOOLS,
                    tool_choice="none",
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                )
            else:
                resp = self.llm.generate_with_tools(
                    messages=self._messages,
                    tools=TOOLS,
                    tool_choice="none",
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                )

            # Build assistant message
            assistant_msg: Dict[str, Any] = {"role": "assistant"}
            tool_calls = resp.tool_calls or []
            if tool_calls:
                remaining = self._remaining_tool_calls()
                if len(tool_calls) > remaining:
                    cap_hit = True
                    tool_calls = tool_calls[:remaining]
            if resp.content:
                assistant_msg["content"] = resp.content
                collected_text += resp.content
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            else:
                assistant_msg.setdefault("content", "")
            self._messages.append(assistant_msg)

            # If no tool calls, we're done with this step
            if not tool_calls:
                break

            # Dispatch tool calls
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                fn_args = tc["function"]["arguments"]
                result = _dispatch_tool(self.wm, fn_name, fn_args)
                self._tool_calls_used += 1

                tool_calls_log.append(
                    {
                        "tool_call_id": tc["id"],
                        "name": fn_name,
                        "arguments": fn_args,
                        "result": result,
                    }
                )

                self._messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    }
                )

        step_entry = {
            "user_message": user_message,
            "tool_calls": tool_calls_log,
            "text": collected_text,
            "kv_snapshot": self.wm.store,
            "tool_call_cap": cap_before,
            "tool_calls_used_total": self._tool_calls_used,
            "tool_call_budget_before": budget_before,
            "tool_call_budget_after": self._remaining_tool_calls(),
            "tool_call_cap_hit": cap_hit,
        }
        self._step_log.append(step_entry)

        if self.debug and (tool_calls_log or cap_hit):
            for tc in tool_calls_log:
                try:
                    args = (
                        json.loads(tc["arguments"])
                        if isinstance(tc["arguments"], str)
                        else tc["arguments"]
                    )
                except json.JSONDecodeError:
                    print(f"    {tc['name']}(<malformed args>)")
                    continue
                if tc["name"] == "write_memory":
                    print(f"    write_memory({args['key']!r}, {args['value']!r})")
                elif tc["name"] == "delete_key":
                    print(f"    delete_key({args['key']!r})")
            if cap_hit:
                print(
                    f"    tool_call_cap_hit "
                    f"({self._tool_calls_used}/{self._tool_call_cap()} used)"
                )
            print(f"  Memory:\n{self.wm.snapshot()}")

        return collected_text

    # ------------------------------------------------------------------
    # encode() — batch encode (backward-compatible convenience wrapper)
    # ------------------------------------------------------------------

    def encode(self, content: Union[str, List[str]]) -> Dict[str, Any]:
        """Present all material at once; agent picks up to MAX_KEYS KV entries.

        Parameters
        ----------
        content : str or list of str
            The full material to encode. If a list is provided (e.g. a word list),
            items are presented as a numbered list.

        Returns
        -------
        dict
            Encoding log with keys: ``content``, ``tool_calls``, ``final_kv``.
        """
        if isinstance(content, list):
            formatted = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(content))
        else:
            formatted = content

        # Use step() internally
        self.step(f"Here is the material to remember:\n\n{formatted}")

        # Build encoding_log from the last step entry
        last_step = self._step_log[-1] if self._step_log else {}
        encoding_log: Dict[str, Any] = {
            "content": content,
            "tool_calls": last_step.get("tool_calls", []),
            "final_kv": self.wm.store,
        }
        self._encoding_log = encoding_log

        if self.debug:
            print("  === ENCODE ===")
            for tc in encoding_log["tool_calls"]:
                try:
                    args = (
                        json.loads(tc["arguments"])
                        if isinstance(tc["arguments"], str)
                        else tc["arguments"]
                    )
                except json.JSONDecodeError:
                    print(f"    {tc['name']}(<malformed args>)")
                    continue
                if tc["name"] == "write_memory":
                    print(f"    write_memory({args['key']!r}, {args['value']!r})")
                elif tc["name"] == "delete_key":
                    print(f"    delete_key({args['key']!r})")
            print(f"  Memory:\n{self.wm.snapshot()}")
            print()

        return encoding_log

    # ------------------------------------------------------------------
    # encode_streaming() — segment-by-segment encode, no lookback
    # ------------------------------------------------------------------

    def _run_segment_tool_loop(
        self, messages: List[Dict[str, Any]], max_tokens: int, budget: int
    ) -> Dict[str, Any]:
        """Tool-dispatch loop for one encode_streaming() segment.

        Not shared with step(): step() also has an allow_tools=False mode where
        tools stay off regardless of budget, which doesn't apply here (every
        segment turn allows tools), so a separate small loop is clearer than a
        parameterized common one.
        """
        tool_calls_log: List[Dict[str, Any]] = []
        collected_text = ""
        used = 0
        cap_hit = False

        while True:
            remaining = budget - used
            tools_available = remaining > 0
            resp = self.llm.generate_with_tools(
                messages=messages,
                tools=TOOLS,
                tool_choice="auto" if tools_available else "none",
                temperature=self.temperature,
                max_tokens=max_tokens,
                parallel_tool_calls=False,
            )
            if not tools_available and resp.tool_calls:
                cap_hit = True

            assistant_msg: Dict[str, Any] = {"role": "assistant"}
            tool_calls = resp.tool_calls or []
            if tool_calls and len(tool_calls) > remaining:
                cap_hit = True
                tool_calls = tool_calls[:remaining]
            if resp.content:
                assistant_msg["content"] = resp.content
                collected_text += resp.content
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            else:
                assistant_msg.setdefault("content", "")
            messages.append(assistant_msg)

            if not tool_calls:
                break

            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                fn_args = tc["function"]["arguments"]
                result = _dispatch_tool(self.wm, fn_name, fn_args)
                used += 1
                tool_calls_log.append(
                    {
                        "tool_call_id": tc["id"],
                        "name": fn_name,
                        "arguments": fn_args,
                        "result": result,
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    }
                )

        return {
            "tool_calls": tool_calls_log,
            "text": collected_text,
            "used": used,
            "cap_hit": cap_hit,
        }

    def encode_streaming(
        self,
        segments: List[str],
        *,
        max_tokens: int = 1024,
        per_segment_tool_call_cap: int = 4,
        trial_tool_call_cap: int = 12,
    ) -> Dict[str, Any]:
        """Present material as ordered segments, one turn per segment, no lookback.

        Each segment's turn builds a FRESH messages list from ``self.wm.snapshot()``
        plus the new segment only — raw text of earlier segments is never resent,
        so the agent must commit to what's worth keeping (via write_memory /
        delete_key) before moving on, rather than planning all writes with
        hindsight of the whole passage (that's what encode() does, and it's the
        gap this method is meant to close).

        Parameters
        ----------
        segments : list of str
            Ordered segments (e.g. paragraphs) presented one at a time.
        max_tokens : int
            Max completion tokens per LLM call within a segment turn.
        per_segment_tool_call_cap : int
            Tool-call budget allotted to a single segment turn.
        trial_tool_call_cap : int
            Tool-call budget shared across the whole trial (all segments
            combined) — MAX_KEYS * 2 by default, enough for every slot to be
            filled once and overwritten once. The effective per-segment
            budget is ``min(per_segment_tool_call_cap, remaining trial
            budget)``, so a segment can't exceed its own cap even when trial
            budget remains, and the trial can't exceed its cap even when
            per-segment budget remains — this also stops the agent hoarding
            writes for the final segment by only running out at the end.

        Returns
        -------
        dict
            Encoding log with keys: ``content``, ``tool_calls``, ``final_kv``,
            plus ``segments_log`` — a list of per-segment dicts with
            ``segment_index``, ``segment_text``, ``tool_calls``, ``kv_snapshot``.
        """
        n_segments = len(segments)
        segments_log: List[Dict[str, Any]] = []
        all_tool_calls: List[Dict[str, Any]] = []
        trial_calls_used = 0

        for idx, segment_text in enumerate(segments, start=1):
            turn_prompt = (
                f"Segment {idx} of {n_segments}.\n\n"
                f"Current memory contents:\n{self.wm.to_recall_text()}\n\n"
                f"New segment:\n{segment_text}"
            )
            messages = [
                {"role": "system", "content": self._build_system()},
                {"role": "user", "content": turn_prompt},
            ]

            segment_budget = min(
                per_segment_tool_call_cap, trial_tool_call_cap - trial_calls_used
            )
            result = self._run_segment_tool_loop(
                messages, max_tokens=max_tokens, budget=segment_budget
            )
            trial_calls_used += result["used"]

            segment_entry = {
                "segment_index": idx,
                "segment_text": segment_text,
                "tool_calls": result["tool_calls"],
                "kv_snapshot": self.wm.store,
            }
            segments_log.append(segment_entry)
            all_tool_calls.extend(result["tool_calls"])

            if self.debug:
                print(f"  === ENCODE_STREAMING segment {idx}/{n_segments} ===")
                for tc in result["tool_calls"]:
                    try:
                        args = (
                            json.loads(tc["arguments"])
                            if isinstance(tc["arguments"], str)
                            else tc["arguments"]
                        )
                    except json.JSONDecodeError:
                        print(f"    {tc['name']}(<malformed args>)")
                        continue
                    if tc["name"] == "write_memory":
                        print(f"    write_memory({args['key']!r}, {args['value']!r})")
                    elif tc["name"] == "delete_key":
                        print(f"    delete_key({args['key']!r})")
                if result["cap_hit"]:
                    print(
                        f"    tool_call_cap_hit (segment {result['used']}/{segment_budget}, "
                        f"trial {trial_calls_used}/{trial_tool_call_cap})"
                    )
                print(f"  Memory:\n{self.wm.snapshot()}")

        encoding_log: Dict[str, Any] = {
            "content": segments,
            "tool_calls": all_tool_calls,
            "final_kv": self.wm.store,
            "segments_log": segments_log,
        }
        self._encoding_log = encoding_log

        if self.debug:
            print()

        return encoding_log

    # ------------------------------------------------------------------
    # recall() — answer from KV contents only
    # ------------------------------------------------------------------

    def recall(self, recall_prompt: str | None = None, max_tokens: int = 512) -> str:
        """Ask agent to recall from KV contents only (no tools, no study history).

        Parameters
        ----------
        recall_prompt : str, optional
            Custom recall prompt. Must contain ``{wm_contents}`` placeholder.
            When *None* the default word-list :data:`RECALL_PROMPT` is used.
        max_tokens : int
            Max completion tokens for the recall response.
        """
        wm_contents = self.wm.to_recall_text()
        recall_prompt = (recall_prompt or RECALL_PROMPT).format(wm_contents=wm_contents)

        resp = self.llm.generate(
            recall_prompt,
            system=self._build_system(),
            temperature=self.temperature,
            max_tokens=max_tokens,
        )

        if self.debug:
            print("  === RECALL ===")
            print(f"  Memory at recall:\n{self.wm.snapshot()}")
            print(f"  Recalled: {resp.text!r}")
            print()

        return resp.text

    def get_log(self) -> List[Dict[str, Any]]:
        return [self._encoding_log] if self._encoding_log else []

    def get_step_log(self) -> List[Dict[str, Any]]:
        return list(self._step_log)


# ---------------------------------------------------------------------------
# SummarizerAgent — ablation baseline: one-shot abstractive summary + recall
# ---------------------------------------------------------------------------

SUMMARIZER_ENCODE_USER_TEMPLATE = "Here is the material to summarize:\n\n{content}"

SUMMARIZER_STEP_USER_TEMPLATE = """\
Current summary:
{summary}

New input:
{new_input}

Update your summary to incorporate the new input. Keep it short. Then, if the \
task requires a response for this turn, output the response after a line \
containing exactly "---ANSWER---". Format:

Updated summary:
<new summary text>
---ANSWER---
<response for this turn, if any>"""


def _split_summary_and_answer(text: str) -> tuple[str, str]:
    """Parse a step() response into (updated_summary, answer).

    Accepts either the exact "---ANSWER---" delimiter or a best-effort fallback
    (strip the "Updated summary:" header; if no delimiter, treat everything as
    summary and return empty answer).
    """
    if not text:
        return "", ""
    delim = "---ANSWER---"
    if delim in text:
        head, _, tail = text.partition(delim)
        summary = head.strip()
        answer = tail.strip()
    else:
        summary = text.strip()
        answer = ""
    # Strip leading "Updated summary:" or "Summary:" header if present.
    for prefix in ("Updated summary:", "updated summary:", "Summary:", "summary:"):
        if summary.startswith(prefix):
            summary = summary[len(prefix) :].lstrip("\n").lstrip()
            break
    return summary, answer


class SummarizerAgent:
    """Ablation baseline for WorkingMemoryAgent.

    Reads material once, produces an abstractive prose summary, and answers from
    that summary only. For turn-based tasks, ``step`` rewrites the running
    summary each turn and optionally emits a per-turn response.
    """

    def __init__(
        self,
        llm: LLM,
        condition_id: str = "C1",
        temperature: float = 0.0,
        debug: bool = False,
        system_prompt_override: str | None = None,
    ) -> None:
        self.llm = llm
        self.condition_id = condition_id
        self.temperature = temperature
        self.debug = debug
        self._system_override = system_prompt_override
        self.summary: str = ""
        self._encoding_log: Dict[str, Any] = {}
        self._step_log: List[Dict[str, Any]] = []

    def _build_system(self) -> str:
        if self._system_override:
            return self._system_override
        raise RuntimeError(
            "SummarizerAgent requires a system_prompt_override (built via "
            "summarizer_system_prompt in wm_prompt_parts)."
        )

    # ------------------------------------------------------------------
    # encode() — one-shot summarization
    # ------------------------------------------------------------------

    def encode(
        self, content: Union[str, List[str]], max_tokens: int = 4096
    ) -> Dict[str, Any]:
        if isinstance(content, list):
            formatted = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(content))
        else:
            formatted = content

        user_msg = SUMMARIZER_ENCODE_USER_TEMPLATE.format(content=formatted)
        resp = self.llm.generate(
            user_msg,
            system=self._build_system(),
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        self.summary = (resp.text or "").strip()

        encoding_log: Dict[str, Any] = {
            "content": content,
            "summary": self.summary,
            "raw": resp.text or "",
        }
        self._encoding_log = encoding_log

        if self.debug:
            print("  === SUMMARIZE ===")
            print(f"  Summary:\n{self.summary}")
            print()

        return encoding_log

    # ------------------------------------------------------------------
    # recall() — answer from summary only
    # ------------------------------------------------------------------

    def recall(self, recall_prompt: str, max_tokens: int = 512) -> str:
        filled = recall_prompt.format(summary=self.summary or "(summary is empty)")
        resp = self.llm.generate(
            filled,
            system=self._build_system(),
            temperature=self.temperature,
            max_tokens=max_tokens,
        )

        if self.debug:
            print("  === RECALL ===")
            print(f"  Summary at recall:\n{self.summary}")
            print(f"  Recalled: {(resp.text or '')!r}")
            print()

        return resp.text or ""

    # ------------------------------------------------------------------
    # step() — turn-based running-summary update + optional per-turn answer
    # ------------------------------------------------------------------

    def step(self, user_message: str, max_tokens: int = 1024) -> str:
        prompt = SUMMARIZER_STEP_USER_TEMPLATE.format(
            summary=self.summary or "(empty)",
            new_input=user_message,
        )
        resp = self.llm.generate(
            prompt,
            system=self._build_system(),
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        raw = resp.text or ""
        new_summary, answer = _split_summary_and_answer(raw)
        if new_summary:
            self.summary = new_summary

        step_entry = {
            "user_message": user_message,
            "raw": raw,
            "summary_after": self.summary,
            "answer": answer,
        }
        self._step_log.append(step_entry)

        if self.debug:
            print(f"  step() input: {user_message!r}")
            print(f"    summary -> {self.summary!r}")
            print(f"    answer  -> {answer!r}")

        return answer

    def get_log(self) -> List[Dict[str, Any]]:
        return [self._encoding_log] if self._encoding_log else []

    def get_step_log(self) -> List[Dict[str, Any]]:
        return list(self._step_log)

    def summary_length_words(self) -> int:
        return len(self.summary.split()) if self.summary else 0
