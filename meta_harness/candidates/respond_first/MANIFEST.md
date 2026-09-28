# `respond_first` — the response is produced before the store update

**Parent:** `evicting_reset` (iteration 5; Qwen mean 0.8460, held-out FAIL on `nback` and on the A4 guard).
**One change, confined to `step()`.** A turn that the harness permits tools on, after the first turn of an
episode, becomes two acts in a fixed order:

| | act | tools | what it produces |
|---|---|---|---|
| ACT 1 | respond | withheld (`tool_choice="none"`) | the turn's reply — what `step()` returns and the **only** thing `step_log["text"]` carries |
| ACT 2 | maintain | offered, on the **same cumulative budget** | tool calls; its text goes to `step_log["maintenance_text"]` and never into `text` |

Turn 1 of every episode and every `allow_tools=False` probe turn are the parent's paths, unchanged.

**Inherited verbatim:** `EpisodicDisplacementMemory` (the store, character for character), `MAX_KEYS` = 4 read
live, `_tool_call_cap()` (not overridden), `encode()`, `recall()`, `snapshot()`, `to_recall_text()`,
`reset_messages()`, `TOOLS`, `CONDITION_PROMPTS`, and the control-state block's strings except the one
obligation sentence.

---

## 1. What the run data shows route 3 to be

Verified myself from `step_log` in `runs/iter5/evicting_reset/Qwen_…` and
`runs/heldout/{baseline,episodic_reset_v3,evicting_reset}/NousResearch_Hermes-4-70B`. Four corrections, in
descending order of how much each changes the design.

### 1.1 `budget=0` is `tool_call_budget_AFTER`, and it is not the failing cell

Conditioning on the budget **entering** the turn, `tool_call_budget_before == 0` occurs on **0.000** of Hermes
turns at every level. No turn that loses a trial on Hermes enters with zero budget. The failing cell is
`b_in == 1`:

| run | lvl | P(unparsed \| b_in=1) | P(unparsed \| b_in≥2) | n at b_in=1 |
|---|---|---|---|---|
| herm evict | 2 | **0.812** | 0.000 | 112 |
| herm evict | 3 | **0.766** | 0.000 | 47 |
| qwen evict | 2 | **0.449** | ≤0.010 (1/99 at b2, 1/1 at b0) | 69 |

The mechanism is one step off the brief's. The agent enters with **one** call left, spends it, the tool loop
re-enters with `_remaining_tool_calls() == 0`, `step()` switches to `tool_choice="none"`, and the model — which
wanted a **second** call — emits it as plain text instead of the answer. Every one of the 24 unparsed replies I
sampled is a bare `<tool_call>{"name": "write_memory", …}</tool_call>` on a turn with `b_in=1`, `calls=1`,
`cap_hit=True`.

### 1.2 It is not a transient, and that is what kills cap-reshaping

Mean budget-entering and mean calls, by turn index, Hermes `evicting_reset` n=2, 50 blocks:

```
b_in  6.00 5.00 4.00 2.78 2.20 2.46 1.62 2.32 1.82 2.48 1.92 2.64 2.18 2.82 2.08 2.66
calls 1.00 1.00 1.22 1.58 1.74 1.84 1.30 1.50 1.34 1.56 1.28 1.46 1.36 1.74 1.42 1.58
```

A persistent period-2 oscillation around ~2 for the whole block; it never recovers. The unparsed turns sit at
trial indices 3, 5, 7, 9, 11, 13 with **caps of 7, 10, 13, 16, 19, 22**, and consumption tracks the cap (turn
15: cap 22, used 21). The brief's account — "spends the floor of 6 by turn 4, loses that turn, then recovers
permanently as the 1.5/turn allowance outpaces its ~1/turn demand" — is contradicted in the two runs it cites.
On `episodic_reset_v3` the same oscillation is tighter and worse (`b_in` alternating 1.3 / 2.1 for sixteen
turns).

Route 3's size is set by the **margin between demand and the 1.5/turn supply**, not by a start-up transient:

| | per-block demand | blocks | unparsed/block | corr(demand, unparsed) |
|---|---|---|---|---|
| herm n=2 | [1.3, 1.6) | 47 | 1.91 | 0.497 |
| herm n=3 | [1.05, 1.3) | 27 | 0.04 | 0.714 |
| herm n=3 | [1.3, 1.6) | 23 | 1.52 | |
| qwen n=2 | [0, 1.05) | 27 | 0.00 | 0.604 |
| qwen n=2 | [1.3, 1.6) | 19 | 1.68 | |

### 1.3 Route 3 is not the largest n-back channel on Hermes, and fixing it alone cannot clear the floor

`answered` counts trial-period replies parsing to Same/Different. A turn is lost two ways, scored identically,
caused differently:

* **UNPARSED** — `_parse_classification` returns `None`. Route 3.
* **NORESP** — it returns `"No response"` during the **trial** period. Call it **route 4**: a position /
  task-set failure, not a tool denial.

| run | lvl | answered | lost | UNPARSED | NORESP |
|---|---|---|---|---|---|
| herm base | 2 | 12.78 | 1.22 | 0.00 | 1.22 |
| herm base | 3 | 11.42 | 2.58 | 0.00 | 2.58 |
| herm evict | 2 | 8.88 | 5.12 | 1.82 | **3.30** |
| herm evict | 3 | 9.72 | 4.28 | 0.72 | **3.56** |
| qwen evict | 2 | 13.34 | 0.66 | 0.66 | 0.00 |
| qwen evict | 3 | 14.00 | 0.00 | 0.00 | 0.00 |

Crediting recovered turns at each block's own `acc_over_answered` — an upper bound, since the recovered turns
are the high-maintenance ones — and recomputing nback humanlikeness:

| scenario | Hermes (baseline 0.8724, floor −0.060) | Qwen (baseline 0.7909) |
|---|---|---|
| as run | 0.7366 (−0.1358) | 0.9587 (+0.1678) |
| + route 3 | 0.7875 (−0.0849) **FAILS** | 0.9634 (+0.1725) |
| + route 3 + route 4 | 0.8895 (+0.0171) **passes** | 0.9634 (route 4 is zero on Qwen) |

Sensitivity, discounting the accuracy credited to recovered turns: route 3 + 4 gives +0.0171 at 1.00×,
−0.0181 at 0.75×, −0.0535 at 0.50× (all inside the −0.060 floor) and −0.0904 at 0.25× (fails). Route 3 alone
fails at **every** discount. So the brief's "route 3 is the only thing standing between `evicting_reset` and a
candidate that passes on both substrates" is false by about 0.10 of humanlikeness, and a candidate that fixes
route 3 only is a predictable held-out failure.

**The two channels are one pathology**, which is what makes a single mechanism available. Both rise with
per-block maintenance demand: corr(demand, noresp) = 0.334 (herm n=2) and 0.550 (herm n=3), beside
corr(demand, unparsed) = 0.497 and 0.714. Five of six sampled mid-block NORESP turns made exactly two calls
(a delete and a write) and then said "no response" — one adding *"because it is the first two letters of the
sequence"* on trial 11. Route 3 is the turn where maintenance is cut off and the repair is spoken; route 4 is
the turn where maintenance completes and the response is an afterthought.

Route 4 is not an inherited per-participant quirk. Hermes' **baseline** NORESP histogram is bimodal —
`{0: 41, 1: 5, 14: 4}` at n=2: four blocks refuse the whole way, forty-one are clean — while under
`evicting_reset` **no** block is clean and every block loses 1 to 9 turns. The reset harness converted a rare
catastrophic per-participant failure into a pervasive per-turn one.

### 1.4 A defect: every episodic run's prompt delta was twice what it was thought to be

`run_candidate.py` calls `verify_interface.check()`, which itself does `load_candidate` + `apply`, and then
calls `load_candidate` + `apply` again. The second `load_candidate` re-executes the candidate module, whose
`from bench.core.wm_agent import WorkingMemoryAgent as _BaseAgent` now resolves to the **already-injected**
first copy. Reproduced directly:

```
after verify:            mh_candidate.WorkingMemoryAgent                 (MRO length 3)
after the second apply:  mh_candidate.WorkingMemoryAgent
                         mh_candidate.WorkingMemoryAgent
                         bench.core.wm_agent.WorkingMemoryAgent
```

So `step()` ran twice per turn and the control-state block was prepended twice. Counted over the recorded runs:
**2400 of 2400** n-back turns carry `[ongoing episode]` twice in `episodic_reset_v3` **and** `evicting_reset`,
on both substrates; the baseline carries it zero times. The outer copy's
`render_store = WM_STATE_HEADER not in user_message` test then suppressed the store in one copy and not the
other, so what the agent actually read was the block once *without* the store and once *with*, and two
counters — "Turn 5 of this episode" and "Stimulus presentations so far: 4" — stated twice each.

This candidate is immune **by construction, not by a guard**: no code path re-enters its own `step()`. Turn 1
goes to the genuine baseline `step()`, located by walking `type(self).__mro__` for the first class whose
`__module__` is `bench.core.wm_agent`, which skips every stacked copy; later turns run `_ordered_step` /
`_probe_step`, which build the block once and never delegate upward. Verified in the stub test: applying the
candidate twice produces byte-identical per-turn results and one `[ongoing episode]` per prompt.

**Attribution consequence, stated rather than buried.** This candidate's prompt differs from the parent's
*realised* prompt in two ways: the ordering (the mechanism) and the de-duplication (a defect repair any correct
implementation performs). Halving the prompt could itself change maintenance verbosity, hence demand, hence
`b_in` incidence. `logs/pending_respond_first.json` therefore names a **required** 14-minute companion arm —
`evicting_reset` with the injection applied once, on Qwen — without which the two are not separable. Until it
exists, the falsification weight sits on the mechanism rows (P4), not on the score.

---

## 2. The mechanism, and why ordering rather than cap-reshaping

### 2.1 The rejected alternative: reshaping the early-turn allowance

The brief grants a special exception for reshaping `_tool_call_cap()` without raising the total, and this is
the option I declined. Three reasons, the first decisive and empirical.

1. **There is no early-turn deficit to reshape.** §1.2. The failing turns are at trial indices 3–13 with caps
   of 7 through 22 and the residual budget oscillating persistently around 2. Front-loading the allowance
   cannot reach a failure that occurs at cap 22. Worse, consumption *tracks* whatever the cap supplies (turn
   15: cap 22, used 21), so extra early budget is simply spent and the margin returns. To reach the `b_in == 1`
   cell one would have to lift the per-turn **rate** above the ~1.43/turn demand Hermes exhibits, i.e. from
   1.5 to ≥2.0 — which is precisely the capacity change the prohibition protects `variable_mapping` from, and
   which would merely relocate the margin: the same delete-then-write blocks would consume 2.0/turn against a
   2.0/turn supply and sit back on the barrier.
2. **It cannot touch route 4 at all**, and route 4 is the larger half of the Hermes deficit (§1.3). Fixing one
   of two channels is a held-out failure on arithmetic already in hand (−0.0849 against a −0.060 floor).
3. **A fixed per-turn action budget has no counterpart in human memory.** Nothing stops a person rehearsing
   twice between two stimuli; what limits them is capacity and time, both of which this harness models
   elsewhere (`MAX_KEYS` 4; one presentation per turn). `_tool_call_cap()` exists to bound API spend. That is
   my position, and it points to a different remedy than re-tuning: if the cap is an artifact, stop routing
   the **response** through it. Which is what this candidate does, leaving the cap untouched.

For the record, the parent's own argument against touching the cap — "on the 471 v3 turns where the budget was
exhausted and nothing was refused, the agent answered 99.1% of the time" — is correct but conditioned on
`budget_after`, and does not survive the `b_in` conditioning: on `evicting_reset`, with refusals at exactly
zero, P(answer) at `b_in = 1` is 0.188 (Hermes n=2) and 0.551 (Qwen n=2). The cell that fails is real; it is
just not reachable by reshaping.

### 2.2 What ordering does

`step()` accumulates text across the tool loop, but the parent's turn produces its classification in the
**last** assistant message, after maintenance is done. The classification is therefore downstream of every way
maintenance can go wrong. Ordered, it is produced in its own model call, before any tool exists to be denied,
and it is the only text the turn returns. Route 3 then cannot cost a trial *even when it still happens*: a
spoken tool call in ACT 2 lands in `maintenance_text` and the trial is already scored. Confirmed in the stub
test — under a stub that speaks the denied call, the parent answers 1/14 at every level while the candidate
answers 14/14, with **identical cap sequences and identical total tool calls**.

Route 4 is addressed by the same move *if* the unified account in §1.3 is right — the response stops being an
afterthought to a delete/write narrative. That is the candidate's bet and it is registered as such (P4b). It
is the half that cannot be proven offline.

### 2.3 What is deliberately not done

Probe turns (`allow_tools=False`) keep the parent's structure and wording: one withheld-tools act, no
maintenance slot. Granting them a slot would also fix route 2 (§6), and the measurements say it would help,
but it is a **second** behavioural change — on tool-permitted turns an existing act moves; on probe turns an
act appears that did not exist — and with both in one arm neither would be attributable. It is handed to
iteration 7 fully measured, and in the meantime Hermes' vm probe turn is this run's **internal control** (§6c).

---

## 3. The psychological argument

The claim is about **order**, not about capacity or about how much may be stored.

A participant in an n-back block does not withhold the response until the rehearsal is finished. The response
is a speeded decision on the current stimulus, made against the contents of the focus of attention; updating —
removing the item that has fallen out of the window, encoding the new one — happens next, in the
inter-stimulus interval.

* Response selection and memory updating are **separable operations with separable costs**. Oberauer (2002,
  2009) and Ecker, Lewandowsky & Oberauer (2014) isolate *removal* as a distinct, time-consuming component of
  updating that follows retrieval-and-response rather than preceding it; Kessler & Meiran (2008) show the
  content-updating cost is paid after the current comparison, not before.
* The **focus of attention already holds the item the response is about** (Cowan 2001; Oberauer 2002), so the
  comparison is available before any store operation. Requiring the store to be reorganised first inverts the
  dependency.
* **Rehearsal occupies the inter-stimulus interval, not the response window** — Baddeley (1986) on the
  articulatory loop; Barrouillet, Bernardin & Camos (2004), whose time-based resource sharing defines
  cognitive load as the proportion of the *free time between stimuli* that processing consumes, free time
  being by construction after the response.
* Running-memory and n-back tasks show **recency without primacy** (Pollack, Johnson & Knaff 1959; Bunting,
  Cowan & Saults 2006), which is why the parent's least-recently-refreshed displacement rule is left untouched:
  this candidate changes when the response is emitted, not what is forgotten.

The converse artifact is worth naming. A harness that requires maintenance first and takes the answer
afterwards models a participant who will not speak until the filing is done. That is not a memory limitation
but a turn-taking one, and it is the single largest source of lost trials in every episodic run this project
has produced.

No swept constants: the candidate contains no numeric parameter of any kind — no rate, threshold, decay or RNG.
The parameterization is the act boundary itself, and there is exactly one place to put it.

---

## 4. Prompt delta, and why each piece is forced

Against the parent's **intended** prompt (one control-state block), the delta is two strings.

1. **`ORDERED_OBLIGATIONS`** replaces the parent's obligation sentence, **on tool-permitted turns only**. The
   parent's wording is *"Update the store to track what you will need later, and give the response this turn's
   instructions call for"* — which names the store update **first**. A structural reordering that left the
   instruction saying the opposite would test the instruction against the structure. The new text states the
   order and states that the maintenance slot is coming, because ACT 1 is a `tool_choice="none"` call and the
   whole bet is that a withheld-tools call which **promises** a slot behaves differently from one that denies
   tools outright. Without the promise the candidate is asking the model to do the thing it demonstrably does
   wrong.
2. **`MAINTENANCE_CUE`** opens ACT 2. Required: ACT 2 is a new user turn and something must occupy it. It
   deliberately contains none of "same", "different" or "no response", so no echo of it could reach a
   classification parser even by accident.

On probe turns the wording is the parent's, character for character — verified in the stub test, which reports
`PARENT` on all five vm probe turns and `ORDERED` on the four later encode turns. The first turn of every
episode is not prepended at all, so the six `encode()` → `recall()` tasks see the baseline's prompt exactly,
as under the parent.

Against the parent's **realised** prompt there is a third difference, the de-duplication (§1.4), which is a
defect repair and is why the companion arm is required.

---

## 5. Why the leak stays closed and `variable_mapping` is not traded away

**The leak.** `reset_messages()` is called at the top of every path — `_ordered_step`, the probe path, and the
baseline `step()` on turn 1 (which has no history to clear). It is upstream of everything this candidate
changes: ACT 1 and ACT 2 share the message list that was just cleared, and neither can see an earlier turn.
Nothing here adds a channel from stimulus to answer other than the store. The n=3 store carries letter identity
at 1.0000 against the baseline's 0.0202 because the store is the only channel, and that is untouched.
Pre-registered as P5.

**`variable_mapping` is untouched on the side that produces its score.** Its probe turns — the only turns whose
text is parsed into an answer — take the parent's path with the parent's wording. Its encode turns become
two-act, so the agent emits an acknowledgement before writing; the task **discards** the return value of those
steps (`bench/tasks/wm_variable_mapping.py:148`), so the acknowledgement cannot enter any score. The store it
builds can differ, which is why vm raw, matched, A4 `n_errors` and `rc_ratio_normalized` are all registered as
HOLD rows (P6) rather than assumed.

**Capacity and budget are not increased anywhere.** `MAX_KEYS` = 4 read live; the store class is the parent's;
`_tool_call_cap()` is not overridden; `_tool_interactions` is incremented **exactly once per tool-permitted
turn**, as the baseline does, so the cap sequence over a block is identical to the parent's — measured in the
stub test as identical lists `[6,6,6,6,7,9,10,12,13,15,16,18,19,21,22,…]` and identical total tool calls
(22 / 24 / 25 at n=1/2/3). Probe turns neither increment the cap nor consume from it. What changes is only
**when within a turn** the budget is spent relative to the response.

**Accepted cost:** one extra LLM call per tool-permitted turn after the first (measured: `tool_choice="none"`
calls 14 → 29 at n=1 over a block; `tool_choice="auto"` calls unchanged at 16). Roughly +45% LLM calls on
`nback` and +50% on `variable_mapping`'s encode turns; the six batch tasks are unaffected. Expect a search arm
nearer 20 minutes than 14.

---

## 6. What I predict differs between Qwen and Hermes

**(a) Route 3's size.** Qwen's demand is 1.03–1.19 calls/turn, Hermes' 1.28–1.47, against the same 1.5 supply,
so Qwen touches `b_in == 1` on 69 turns of 800 and Hermes on 112 of 800 at n=2. The fix therefore recovers
0.66 turns/block on Qwen and 1.82 on Hermes at n=2.

The demand difference is a **strategy** difference, and my first hypothesis about it was wrong in direction, so
the corrected version: blocks using **few** distinct keys carry **more** deletes, not fewer (9.7 vs 7.5 per
block at Hermes n=2) and higher demand (1.47 vs 1.41), because they delete a rolling key and rewrite it. Qwen
issues 0.0–0.1 deletes per block and lets the displacing store free the slot; Hermes issues 5–10. Under a
displacing store a delete-then-write is **redundant** — the write alone would have evicted — so Hermes pays two
calls for what the store gives for one, and that is what puts it on the margin. Corollary worth a future
candidate: `DISPLACED_TEMPLATE` tells the agent *"the least recently used entry X was displaced and is now
lost"*, which may be teaching it to delete first in order to **choose** the victim. That is a one-string
mechanism and a better next move than any cap change.

**(b) Route 4 exists only on Hermes** — zero NORESP turns on Qwen at either level, 3.30 and 3.56 per block on
Hermes. So the part of this candidate that is a bet rather than a construction is testable **only** on the
held-out substrate. On Qwen the candidate can do no better than about +0.005 on `nback` over the parent, and
that is the honest ceiling there.

**(c) The A4 guard will fire on Hermes and not on Qwen, and I can say why.** The brief asks whether A4 is
measuring something different on Hermes. **It is not.** It is measuring a failure the brief records as fixed.
**Route 2 — `allow_tools=False` on a question turn — is wide open on Hermes:** 549 of 1500 `variable_mapping`
answers do not parse, and the raw replies are `<tool_call>{"name": "write_memory", …}</tool_call>`. On Qwen the
same count is **1**. Those 549 sit at `relation_count` 6.22 against 8.43 on correct answers — they are
concentrated on **early** questions — so they drag `rc_mean_error` below `rc_mean_correct` and the normalized
ratio below zero:

| Hermes `evicting_reset` | n_errors | rc_ratio | ceiling | normalized |
|---|---|---|---|---|
| as run | 1136 | 0.9320 | 2.6742 | **−0.0406** |
| unparsed trials dropped | 587 | 1.1142 | 1.3402 | **0.3356** |
| humans | 152 | 1.3867 | 2.0372 | 0.3728 |

Hermes' **parsed** error structure is human-shaped to within 0.04 of the human value. The axis is fine; the
parse failure is not. And the vm score does not protect route 2: simulating recovery of those 549 at recovery
probability *p*, raw vm humanlikeness goes 0.6968 (p=0) → 0.7862 (p=0.383, the observed parsed accuracy) →
0.9362 (p=0.7) → 0.7280 (p=1.0) — **up at every rate** — while Qwen is unmoved at 0.6850 throughout. So fixing
route 2 is predicted to *improve* `variable_mapping` on Hermes and leave it alone on Qwen.

This candidate does not fix it (§2.3), so its Hermes A4 row is pre-registered as an **expected and explained**
violation (P6c): `vm_delta` will be ≈ +0.34, far past the 0.017 trigger, `n_errors` will clear `trustworthy`,
and `rc_ratio_normalized` will stay near −0.04. A predicted, explained guard failure is a better artifact than
a surprise one.

**The internal control this run buys.** Hermes' nback ACT 1 is a withheld-tools call that **promises** a
maintenance slot; Hermes' vm probe turn is a withheld-tools call that does not. If the promise is what matters,
nback ACT-1 unparsed collapses while vm unparsed stays near 549. **If vm unparsed falls without a probe slot,
my account of route 2 is wrong** and should be retracted.

---

## 7. Pre-registered predictions

Registered before any run. Full machine-readable form, with every baseline and reference figure, in
`meta_harness/logs/pending_respond_first.json`. Thresholds below are the voiding/verdict-bearing ones.

| id | row | quantity | threshold | voids the rest? |
|---|---|---|---|---|
| **P1** | **PRECONDITION — the mechanism engages** | `unparsed` replies per block at n=1/2/3, and ACT-1 unparsed share per level | unparsed/block **≤ 0.2** at all three levels on **both** substrates; ACT-1 unparsed share **≤ 0.02** per level | **YES** |
| P2 | n-back answered, all three levels | `answered` at n=1/2/3 | Qwen ≥ 13.5 / 13.5 / 13.5; Hermes ≥ 13.5 / 10.5 / 10.2 | no |
| P3 | nback humanlikeness clears its 0.060 floor | `humanlikeness_by_task["nback"]` | Qwen ≥ 0.7309 (point est. 0.963); **Hermes** ≥ 0.8124, banded **[−0.055, +0.020]** vs its 0.8724 baseline | no |
| P4a | MECHANISM — route 3 | `unparsed`/block, `b_in==1` incidence, `cap_hit` share | unparsed/block falls 0.66 → ≤0.2 (Qwen n=2), 1.82 → ≤0.2 and 0.72 → ≤0.2 (Hermes n=2/3); `b_in==1` incidence **unchanged**, ±0.05 | no |
| P4b | MECHANISM — route 4, **the bet** | NORESP/block at n=2/3 | Hermes 3.30 → **≤ 1.5** and 3.56 → **≤ 1.8**; Qwen stays 0.00 | no |
| P5 | the leak stays closed | n=3 store letter-identity share | ≥ 0.90 (baseline 0.0202, parent 1.0000) | no |
| P6a | vm holds, raw and matched | `variable_mapping` raw; `variable_mapping_matched` | Qwen raw ≥ 0.66 and matched ≥ 0.62 (parent 0.6850 / ~0.71); Hermes raw ≥ 0.66 | no |
| P6b | vm error structure, Qwen | A4 `n_errors`, `rc_ratio_normalized` | Qwen `n_errors` ≥ 150 and `rc_ratio_normalized` ≥ 0.15 (parent 475, 0.7145) | no |
| P6c | vm error structure, Hermes — **expected violation** | A4 `rc_ratio_normalized` | predicted **−0.10 … +0.10**, i.e. the guard fires; cause given in §6c; a value ≥ 0.15 would mean route 2 closed by itself and §6c is wrong | no |
| P7 | parse integrity | vm unparsed answers; tool-call-in-text on vm | Qwen ≤ 5 (parent 1); **Hermes predicted 450–650**, i.e. unchanged at ~549 — this is the internal control, not a hope | no |
| P8 | batch-task control, differenced against **`evicting_reset`** | the six `encode()`→`recall()` tasks | \|Δ\| ≤ **0.025** on `semantic_story_recall`, `narrative_qa`, `digit_span_reverse`, `word_recognition`; ≤ **0.030** on `craft_task`; ≤ **0.140** on `digit_span_forward` | no |
| P9 | ANTI-`full_context` — the gain must not come from an empty store | `keys_held`, `acc_over_answered` | n=3 `keys_held` ≥ 3.5 and n=2 ≥ 2.0; n=3 `acc_over_answered` ≤ 0.85 | no |
| P10 | buffer-period compliance — reported | `buffer_no_response_frac` | reported, not thresholded | no |
| H1 | NAMED NON-VOIDING HAZARD | `craft_task` | effective floor 0.030; the parent already sits at −0.0280 vs baseline on a provably identical path | no |

**Why P1 is pitched at the mechanism and not at the score.** Route 3 alone provably cannot restore Hermes'
`answered` baseline (§1.3: +0.051 of a needed +0.076). A precondition pitched at baseline restoration on Hermes
would void the route-3 mechanism row whenever route 4 fails to move — which is exactly the finding the run
exists to establish. So the precondition asserts only what the construction guarantees: the classification no
longer disappears when a tool call does. The nback floor and humanlikeness are separate, non-voiding ambition
rows, and P4b is registered as an explicit bet whose failure is informative rather than voiding.

**`capable_of_varying` evidence for every row**: `unparsed` ranges 0.00–6.52 per block across recorded arms;
`answered` 2.12–14.0; NORESP 0.00–3.56; nback humanlikeness 0.6978–0.9587 on Qwen and 0.7366–0.8724 on Hermes;
vm raw 0.3524–0.6968; A4 `rc_ratio_normalized` −0.0444 to 0.8705; letter-identity share 0.0202–1.0000;
`keys_held` 1.00–4.00. Every quantity has moved between recorded arms, so none is a constant.

### What `respond_first_checks` should assert (do **not** edit `check_predictions.py` for me)

* **P1 / P4a** — from `wm_nback.jsonl` `step_log`, over turns with index > `n_level`: `unparsed` =
  `_parse_classification(st["text"]) is None`, per block, per level. **Read `st["text"]` only.** ACT-1 unparsed
  share = the same count restricted to rows where `st.get("ordered_turn")` is true. `b_in==1` incidence from
  `st["tool_call_budget_before"] == 1`.
* **P4b** — NORESP = `_parse_classification(st["text"]) == "No response"` on turns with index > `n_level`.
* **Both must assert that `st["maintenance_text"]` was NOT concatenated into `st["text"]`** — assert
  `st["text"]` contains no `"<tool_call>"`-style substring for any row where `maintenance_text` does. This is
  load bearing: `_parse_classification` tests `"no response"` **before** `"different"`, so a leaked ACT-2
  string could invert a correct ACT-1 answer.
* **P2 / P9 / P10** — `meta_harness.nback_levels.report(run_dir)["diagnostics"][n]`: `answered`,
  `acc_over_answered`, `keys_held`, `n_no_answers`, `buffer_no_response_frac`.
* **P3** — `humanlikeness_by_task["nback"]` and `delta_vs_baseline["nback"]` from `score_candidate.evaluate`.
* **P5** — the n=3 store letter-identity share, the same computation `episodic_reset_v3_checks` uses.
* **P6** — `humanlikeness_by_task["variable_mapping"]`, `axes()["variable_mapping_matched"]`, and
  `axes()["A4"]["n_errors"]` / `["rc_ratio_normalized"]`.
* **P7** — from `wm_variable_mapping.jsonl` `step_logs`: count `q_idx` values absent from `parsed_answers`, and
  count `answer_raw` strings containing `"tool_call"` or `"write_memory"`.
* **P8** — per-task deltas against `runs/iter5/evicting_reset/<model>`, **not** the baseline, with the bands in
  the table. The band is the parent's identical-path spread (craft moved 0.0248 between baseline and v3 on a
  provably identical code path), not the recorded noise floor, which understates it ~8×.

---

## 8. Offline verification, and what I could not test

**Verified** (`verify_interface.py`: PASS — overrides `WorkingMemoryAgent`, `SummarizerAgent`, `TOOLS`,
`CONDITION_PROMPTS`, `MAX_KEYS`; round trip fine; capacity enforced at 4). Then, driving the **real**
`run_nback_block` at n=1/2/3 and the **real** `variable_mapping` encode/probe alternation against one and the
same stub — a stub reproducing the measured Hermes regime, a delete plus a write on every maintenance
opportunity (2.0 calls/turn against a 1.5/turn allowance) and the spoken tool call when denied mid-turn:

| | `evicting_reset` | `respond_first` | `respond_first` applied **twice** |
|---|---|---|---|
| answered, n=1/2/3 | 1 / 1 / 1 of 14 | **14 / 14 / 14** | 14 / 14 / 14 |
| unparsed | 13 / 13 / 13 | **0 / 0 / 0** | 0 / 0 / 0 |
| spoken tool call in `text` | 12 / 13 / 14 | **0 / 0 / 0** | 0 / 0 / 0 |
| turns with `maintenance_text` | — | 15 / 16 / 17 of 15 / 16 / 17 | same |
| cap sequence | `[6,6,6,6,7,9,10,12,…]` | **identical** | identical |
| total tool calls | 22 / 24 / 25 | **22 / 24 / 25** | 22 / 24 / 25 |
| `b_in` sequence | `[6,4,2,0,1,2,1,2,…]` | **identical** | identical |
| `[ongoing episode]` per prompt | 1 | 1 | **1** |
| refusals | 0 | 0 | 0 |
| LLM calls (none / auto) | 14 / 16 | 29 / 16 | 29 / 16 |

Batch path (`encode()` → `recall()`, the six bottlenecked tasks), all three configurations identical: one user
message, first message `"Here is the material to remember:\n\n1. alpha…"`, **no** `[ongoing episode]`, same
keys stored, `len(step_log) == 1`, no `maintenance_text` field. The parent already short-circuits on
`_episode_turn_index() == 1`, so this is provable rather than merely observed: the MRO-walk base path yields
the identical message.

`variable_mapping` alternation: probe answers parse 5/5 under both; obligation wording is `PARENT` on all five
probe turns and `ORDERED` on the four later encode turns; cap sequence `[6,6,6,6,6,6,6,6,7,7]` and 7 total tool
calls under **both**.

**What I could not test.** The one thing the candidate rests on: whether a real model, on a `tool_choice="none"`
call that **promises** a later maintenance slot, answers instead of speaking a tool call. My stub answers there
by construction — it cannot be otherwise offline, because the question is about the model's disposition, not
the harness's control flow. This is why ACT-1 unparsed share is a first-class pre-registered row (P1): if it is
nonzero the mechanism is dead and that number says so on sight. I also could not test route 4 at all (it does
not exist on Qwen and a stub cannot reproduce a position error), nor whether the acknowledgement ACT 1 produces
on `variable_mapping`'s encode turns changes the store the agent then builds. The stub test is at
`scratchpad/rf_stubtest.py` in this session's scratch directory and can be re-run against both candidates.
