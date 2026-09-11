"""Shared fixtures: synthetic trials and a fake Tinker SDK.

The fake SDK is installed into sys.modules so tinker_client / train can be exercised
(datum construction, loss masking, which model each round samples from and trains
from) without the real SDK, an API key, or any spend.
"""
from __future__ import annotations

import importlib.machinery
import sys
import types as pytypes
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest

from rationales.config import StarConfig
from rationales.data import HumanTrial


# ---------------------------------------------------------------------------
# Trials
# ---------------------------------------------------------------------------
def make_trial(
    *,
    direction: str = "forward",
    digits: List[int],
    user: List[int],
    length: Optional[int] = None,
    participant: str = "participant-0001",
    trial: int = 1,
) -> HumanTrial:
    expected = digits if direction == "forward" else list(reversed(digits))
    return HumanTrial(
        direction=direction,
        participant_id=participant,
        run_id="run-test",
        trial=trial,
        length=length or len(digits),
        sequence_index=1,
        digits=list(digits),
        expected_digits=list(expected),
        user_digits=list(user),
        expected="".join(map(str, expected)),
        user_response="".join(map(str, user)),
        correct=list(user) == list(expected),
    )


@pytest.fixture
def fwd_success() -> HumanTrial:
    return make_trial(direction="forward", digits=[8, 4, 1], user=[8, 4, 1])


@pytest.fixture
def fwd_fail() -> HumanTrial:
    # Transposition: human typed 3 2 9 for 3 9 2.
    # Distinct digits from fwd_success on purpose -- tests route stub responses by the
    # digit sequence in the prompt, so shared stimuli would collide.
    return make_trial(direction="forward", digits=[3, 9, 2], user=[3, 2, 9], trial=2)


@pytest.fixture
def rev_success() -> HumanTrial:
    # Presented 9 9 5 -> correct response 5 9 9.
    return make_trial(direction="reverse", digits=[9, 9, 5], user=[5, 9, 9])


@pytest.fixture
def rev_fail() -> HumanTrial:
    # Presented 2 7 4 5 -> correct 5 4 7 2; human truncated to 5 4 7.
    return make_trial(direction="reverse", digits=[2, 7, 4, 5], user=[5, 4, 7], trial=2)


@pytest.fixture
def trials(fwd_success, fwd_fail, rev_success, rev_fail) -> List[HumanTrial]:
    return [fwd_success, fwd_fail, rev_success, rev_fail]


@pytest.fixture(autouse=True)
def _isolate_global_ledger(tmp_path, monkeypatch):
    """Keep test spend out of the real lifetime ledger.

    CostTracker appends every call to config.GLOBAL_LEDGER by default. Tests record
    synthetic million-token calls, so without this they write fake dollars into the
    file used to report actual spend -- which is exactly what happened: 366 test calls
    inflated the lifetime tally to $34.88 against ~$0.51 of real usage.
    """
    monkeypatch.setattr("rationales.config.GLOBAL_LEDGER", tmp_path / "spend_ledger.jsonl")


@pytest.fixture
def cfg() -> StarConfig:
    return StarConfig(
        base_model="Qwen/Qwen3-8B",
        k=1,
        temperature=0.0,
        rounds=1,
        use_fewshot=False,   # real few-shot files still carry PLACEHOLDER text
        steps_1=2,
        batch_size=2,
        warmup_steps=2,
        max_workers=1,
    )


# ---------------------------------------------------------------------------
# Fake Tinker SDK
# ---------------------------------------------------------------------------
@dataclass
class FakeDatum:
    tokens: List[int]
    target_tokens: List[int]
    weights: List[float]


class _FakeModelInput:
    def __init__(self, tokens: List[int]) -> None:
        self.tokens = list(tokens)

    @classmethod
    def from_ints(cls, tokens=None, **kw):
        if tokens is None:
            tokens = kw.get("tokens")
        return cls(tokens)


class _FakeTensorData:
    def __init__(self, values) -> None:
        self.values = list(values)

    @classmethod
    def from_torch(cls, t):
        return cls(t.tolist())


class _FakeFuture:
    def __init__(self, value: Any) -> None:
        self._value = value

    def result(self):
        return self._value


class _FakeLossResult:
    def __init__(self, loss: float) -> None:
        self.loss = loss


class _FakeSaveResult:
    def __init__(self, path: str) -> None:
        self.path = path


class FakeTrainingClient:
    def __init__(self, base_model: str, rank: int, recorder: "FakeRecorder") -> None:
        self.base_model = base_model
        self.rank = rank
        self.rec = recorder
        self.steps = 0

    def get_tokenizer(self):
        return self.rec.tokenizer

    def forward_backward(self, datums, loss_fn="cross_entropy"):
        self.rec.loss_fns.append(loss_fn)
        self.rec.batches.append(list(datums))
        self.steps += 1
        return _FakeFuture(_FakeLossResult(1.0 / self.steps))

    def optim_step(self, params):
        self.rec.learning_rates.append(params.learning_rate)
        return _FakeFuture(None)

    def save_weights_for_sampler(self, name: str):
        path = f"tinker://fake/{name}"
        self.rec.saved.append(path)
        return _FakeFuture(_FakeSaveResult(path))

    def save_state(self, name: str):
        """Full state (weights + optimizer), used to resume an interrupted round."""
        path = f"tinker://fake/state/{name}"
        self.rec.states.append(path)
        return _FakeFuture(_FakeSaveResult(path))


class FakeSamplingClient:
    def __init__(self, model_path: Optional[str], base_model: str, recorder: "FakeRecorder") -> None:
        self.model_path = model_path
        self.base_model = base_model
        self.rec = recorder

    def sample(self, prompt, num_samples, sampling_params, **kw):
        seqs = []
        for _ in range(num_samples):
            text = self.rec.responder(self.rec.decode(prompt.tokens), self.model_path)
            self.rec.calls.append({"model_path": self.model_path, "text": text})
            seqs.append(pytypes.SimpleNamespace(tokens=self.rec.encode(text)))
        return pytypes.SimpleNamespace(sequences=seqs)


class FakeTokenizer:
    """Char-level tokenizer: ord(c)+1, so decode(encode(x)) == x exactly."""

    eos_token_id = 0

    def encode(self, text: str) -> List[int]:
        return [ord(c) + 1 for c in text]

    def decode(self, ids) -> str:
        return "".join(chr(i - 1) for i in ids if i != 0)


class FakeRecorder:
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.calls: List[Dict[str, Any]] = []
        self.batches: List[List[Any]] = []
        self.learning_rates: List[float] = []
        self.loss_fns: List[str] = []
        self.saved: List[str] = []
        self.states: List[str] = []
        self.resumed_from: List[str] = []
        self.training_clients: List[FakeTrainingClient] = []
        self.sampling_clients: List[FakeSamplingClient] = []
        self.responder = lambda prompt, model_path: ""

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids) -> str:
        return self.tokenizer.decode(ids)


class FakeServiceClient:
    def __init__(self, recorder: FakeRecorder) -> None:
        self.rec = recorder

    def create_lora_training_client(self, base_model: str, rank: int):
        c = FakeTrainingClient(base_model, rank, self.rec)
        self.rec.training_clients.append(c)
        return c

    def create_training_client_from_state_with_optimizer(self, path: str, **kw):
        self.rec.resumed_from.append(path)
        c = FakeTrainingClient(kw.get("base_model", "resumed"), 0, self.rec)
        self.rec.training_clients.append(c)
        return c

    def create_sampling_client(self, model_path: Optional[str] = None, base_model: str = ""):
        c = FakeSamplingClient(model_path, base_model, self.rec)
        self.rec.sampling_clients.append(c)
        return c


@pytest.fixture
def fake_tinker(monkeypatch):
    """Install a fake `tinker` module and hand back the recorder."""
    rec = FakeRecorder()

    mod = pytypes.ModuleType("tinker")
    mod.ModelInput = _FakeModelInput
    mod.TensorData = _FakeTensorData
    mod.SamplingParams = lambda **kw: pytypes.SimpleNamespace(**kw)
    mod.AdamParams = lambda **kw: pytypes.SimpleNamespace(**kw)
    mod.ServiceClient = lambda: FakeServiceClient(rec)

    types_mod = pytypes.ModuleType("tinker.types")
    types_mod.ModelInput = _FakeModelInput

    def _datum(model_input=None, loss_fn_inputs=None, **kw):
        return FakeDatum(
            tokens=model_input.tokens,
            target_tokens=loss_fn_inputs["target_tokens"].values,
            weights=loss_fn_inputs["weights"].values,
        )

    types_mod.Datum = _datum
    mod.types = types_mod

    monkeypatch.setitem(sys.modules, "tinker", mod)
    monkeypatch.setitem(sys.modules, "tinker.types", types_mod)

    # build_datum wraps its tensors with torch. Real torch is used when installed; the
    # stub only stands in when it is absent, so the suite runs either way. Shadowing a
    # real torch breaks importers that inspect torch.__spec__.
    try:
        import torch  # noqa: F401
    except ImportError:
        torch_mod = pytypes.ModuleType("torch")
        torch_mod.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
        torch_mod.tensor = lambda xs: pytypes.SimpleNamespace(tolist=lambda: list(xs))
        monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setenv("TINKER_API_KEY", "fake-key-for-tests")
    return rec


@pytest.fixture
def session(cfg, fake_tinker):
    """A TinkerSession wired to the fake SDK, with the char tokenizer forced in."""
    from rationales.tinker_client import TinkerSession

    s = TinkerSession(cfg)
    s._tokenizer = fake_tinker.tokenizer
    return s
