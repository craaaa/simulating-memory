from __future__ import annotations

import json
import re
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
                "The value should be an abstractive summary of the relevant information. "
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


_SPOKEN_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _strip_spoken_tool_calls(text: str) -> tuple[str, List[str]]:
    """Remove complete spoken tool-call blocks from a reply.

    A model that is shown tool schemas but forbidden from calling them will sometimes type
    the call out instead of answering. That text is not an answer: every task parser looks
    for a classification or a choice and finds none, so the trial is recorded as unanswered
    -- which lowers the model's score and, because the model is MORE accurate than humans
    on most of these tasks, silently raises its measured humanlikeness.

    Only complete `<tool_call>...</tool_call>` blocks are removed, so a reply that merely
    mentions a tool by name is untouched. Returns the cleaned text and the blocks removed.

    >>> _strip_spoken_tool_calls("different")
    ('different', [])
    >>> _strip_spoken_tool_calls('same\\n<tool_call>\\n{"name": "write_memory"}\\n</tool_call>')
    ('same', ['{"name": "write_memory"}'])
    >>> _strip_spoken_tool_calls("I will use write_memory next turn")
    ('I will use write_memory next turn', [])
    """
    found = _SPOKEN_CALL_RE.findall(text)
    if not found:
        return text, []
    return _SPOKEN_CALL_RE.sub("", text).strip(), found


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

CONDITION_PROMPTS = {
    "C2": SYSTEM_PROMPT_HUMAN,
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
        participant_id: Any = None,
    ) -> None:
        self.llm = llm
        self.condition_id = condition_id
        self.temperature = temperature
        self.debug = debug
        self._system_override = system_prompt_override
        # Identity of the SCORED UNIT this agent instance produces -- the same key
        # `src/score.py` turns into one observation in the model's score distribution.
        # Nothing in this class reads it. It exists so a candidate harness can seed a
        # per-participant mechanism from the participant rather than from the stimulus
        # content: content varies per participant on only some tasks (measured:
        # 4 distinct contents across 200 semantic_story_recall rows, 3 across 150
        # craft_task rows), so a content hash silently collapses the population.
        self.participant_id = participant_id
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

        Starts from a FRESH context (system prompt only), appends *user_message*, gets the
        LLM response (processing any tool calls in a loop until the agent stops calling
        tools), and returns the final text content.

        Conversation history is NOT maintained across calls. Each turn sees the system
        prompt, its own user message, and its own tool-call loop -- nothing else. The
        key-value store is the only thing that crosses a turn boundary, which is what this
        benchmark claims to measure. Before this, `_messages` accumulated every prior turn,
        so a task presented turn-by-turn could be answered from the transcript without
        consulting the store at all; `wm_nback` and `wm_variable_mapping` were both
        answerable that way, and they were the only two of the eight working-memory tasks
        whose scores any harness change could move.

        The tool loop still needs multi-message state WITHIN a turn (the assistant's
        tool_calls message and the tool result messages), which is why this method exists
        rather than a single `generate()` call.

        The batch tasks are unaffected: each calls `encode()` exactly once, and `recall()`
        never reads `_messages`.

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
        # Turn boundary: the store carries over, the transcript does not.
        self.reset_messages()
        self._ensure_messages()
        if allow_tools:
            self._tool_interactions += 1
        self._messages.append({"role": "user", "content": user_message})

        tool_calls_log: List[Dict[str, Any]] = []
        spoken_calls: List[str] = []
        refused_calls: List[Dict[str, Any]] = []
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
            else:
                # No tools this turn -- either the budget is gone or the caller passed
                # allow_tools=False. Send NO schemas, rather than schemas plus
                # tool_choice="none": the schemas are rendered into the chat template
                # either way, and a model that sees them emits a <tool_call> block as
                # ordinary content. Nothing parses that back out when calls are
                # disallowed, so it was returned as the agent's reply and reached the
                # task's answer parser, which recorded the trial as unanswered. Measured
                # on runs already collected: 554 of 1500 variable_mapping replies under
                # evicting_reset, and 2417 of 4950 n-back replies under respond_first.
                if allow_tools:
                    cap_hit = True
                resp = self.llm.generate_with_tools(
                    messages=self._messages,
                    tools=[],
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                )

            # Build assistant message
            assistant_msg: Dict[str, Any] = {"role": "assistant"}
            tool_calls = resp.tool_calls or []
            dropped_calls: List[Dict[str, Any]] = []
            if tool_calls:
                remaining = self._remaining_tool_calls()
                if len(tool_calls) > remaining:
                    cap_hit = True
                    dropped_calls = tool_calls[remaining:]
                    tool_calls = tool_calls[:remaining]
            if resp.content:
                assistant_msg["content"] = resp.content
                # Keep the full content in the transcript the model sees, but do not let a
                # spoken tool call become the agent's ANSWER: a reply that is only
                # `<tool_call>{...}</tool_call>` carries no classification, and the task
                # parsers score it as no answer. Strip complete blocks and record them.
                clean, spoken = _strip_spoken_tool_calls(resp.content)
                if spoken:
                    spoken_calls.extend(spoken)
                collected_text += clean
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            else:
                assistant_msg.setdefault("content", "")
            # Calls beyond the budget are dropped from the assistant message, so the
            # transcript the model is resent stays self-consistent -- it sees the calls it
            # kept and a result for each. What was missing was any RECORD that an intended
            # write never happened, which made a cap-truncated turn indistinguishable from
            # a turn where the model chose to write once.
            for tc in dropped_calls:
                refused_calls.append({
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                    "reason": "tool_call_budget_exhausted",
                })
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
            "spoken_tool_calls": spoken_calls,
            "refused_tool_calls": refused_calls,
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
        participant_id: Any = None,
    ) -> None:
        self.llm = llm
        self.condition_id = condition_id
        self.temperature = temperature
        self.debug = debug
        self._system_override = system_prompt_override
        # Accepted for symmetry with WorkingMemoryAgent only; nothing reads it here.
        self.participant_id = participant_id
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
