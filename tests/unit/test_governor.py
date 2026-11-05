"""U-GOV-props (never two generators, in-use never evicted, FIFO), U-GOV-nofit, queue cancel."""

from __future__ import annotations

import random
import threading
import time

import pytest

from core.errors import CancelledError, DoesNotFitError, QueueTimeoutError
from core.obs.log import Logger, LogRing
from core.runtime.governor import Governor
from core.schemas import ModelPreset, ModelPresets

SIZES = {"emb-text": 350, "emb-mm": 950, "gen-a": 1400, "gen-b": 1300, "gen-huge": 4000}


class FakeRouter:
    def __init__(self, base_mb: float = 300) -> None:
        self.loaded_set: set[str] = set()
        self.base = base_mb
        self.calls: list[tuple[str, str]] = []

    def load(self, model: str, timeout: float = 0) -> float:
        self.calls.append(("load", model))
        self.loaded_set.add(model)
        return 1.0

    def unload(self, model: str, timeout: float = 0) -> float:
        self.calls.append(("unload", model))
        self.loaded_set.discard(model)
        return 1.0

    def loaded(self) -> list[str]:
        return sorted(self.loaded_set)

    def used(self) -> float:
        return self.base + sum(SIZES[m] for m in self.loaded_set)


def presets() -> ModelPresets:
    def p(pid: str, role: str) -> ModelPreset:
        return ModelPreset(id=pid, label=pid, role=role, family="x", files={"model": f"{pid}.gguf"}, n_ctx=2048, est_peak_mb=SIZES[pid])  # type: ignore[arg-type]

    return ModelPresets(
        schema=1, presets=[p("emb-text", "embedder"), p("emb-mm", "embedder"), p("gen-a", "generator"), p("gen-b", "generator"), p("gen-huge", "generator")]
    )


def gov(cap: float = 2700) -> tuple[Governor, FakeRouter]:
    r = FakeRouter()
    g = Governor(r, presets(), r.used, hard_budget_mb=cap * 0.95, logger=Logger(LogRing(), stdout=False), queue_timeout_s=5)
    return g, r


def generators(r: FakeRouter) -> list[str]:
    return [m for m in r.loaded_set if m.startswith("gen")]


def test_random_sequences_keep_invariants() -> None:
    g, r = gov()
    rng = random.Random(0)
    for _ in range(400):
        m = rng.choice(["emb-text", "emb-mm", "gen-a", "gen-b"])
        try:
            with g.using(m):
                assert m in r.loaded_set
                assert len(generators(r)) <= 1, "never two generators"
                assert not ("emb-mm" in r.loaded_set and generators(r)), "emb-mm never co-resides with a generator"
                assert r.used() <= g.hard_budget_mb
        except DoesNotFitError:
            pass
        assert g.resident() == r.loaded()


def test_in_use_model_is_never_evicted() -> None:
    g, r = gov()
    g.ensure("gen-a")  # in use
    with pytest.raises(DoesNotFitError):
        g.ensure("gen-b")  # would need to evict gen-a by rule
    assert "gen-a" in r.loaded_set
    g.release("gen-a")
    g.ensure("gen-b")
    assert generators(r) == ["gen-b"]


def test_does_not_fit_leaves_system_healthy() -> None:
    g, r = gov()
    g.ensure("emb-text")
    g.release("emb-text")
    with pytest.raises(DoesNotFitError) as ei:
        g.ensure("gen-huge")
    assert "does not fit" in ei.value.user_message
    assert "gen-huge" not in r.loaded_set
    with g.using("gen-a"):  # still usable afterwards
        assert "gen-a" in r.loaded_set


def test_memory_eviction_order_prefers_generators_then_emb_mm() -> None:
    g, r = gov(cap=2000)  # budget 1900
    g.ensure("emb-text")
    g.release("emb-text")
    g.ensure("gen-b")
    g.release("gen-b")
    r.calls.clear()
    g.ensure("emb-mm")  # rule unloads the generator first
    assert r.calls[0] == ("unload", "gen-b") and ("load", "emb-mm") in r.calls


def test_fifo_queue_and_positions() -> None:
    g, _ = gov()
    order: list[str] = []
    first = g.enqueue("s1", "chat")
    g.wait_turn(first)
    tickets = [g.enqueue(f"s{i}", "chat") for i in range(2, 5)]
    assert [g.position(t) for t in tickets] == [1, 2, 3]

    def worker(t, name):  # type: ignore[no-untyped-def]
        g.wait_turn(t)
        order.append(name)
        time.sleep(0.02)
        g.finish(t)

    threads = [threading.Thread(target=worker, args=(t, f"s{i}")) for i, t in zip(range(2, 5), tickets, strict=True)]
    for th in threads:
        th.start()
    time.sleep(0.05)
    g.finish(first)
    for th in threads:
        th.join(5)
    assert order == ["s2", "s3", "s4"]


def test_cancel_queued_ticket_does_not_affect_running() -> None:
    g, _ = gov()
    running = g.enqueue("a", "chat")
    g.wait_turn(running)
    queued = g.enqueue("b", "chat")
    g.cancel(queued)
    with pytest.raises(CancelledError):
        g.wait_turn(queued, timeout=1)
    assert g.position(running) == 0
    g.finish(running)


def test_queue_timeout() -> None:
    g, _ = gov()
    holder = g.enqueue("a", "chat")
    g.wait_turn(holder)
    t = g.enqueue("b", "chat")
    with pytest.raises(QueueTimeoutError):
        g.wait_turn(t, timeout=0.2)
    assert g.position(t) == -1
    g.finish(holder)


def test_load_over_hard_cap_is_rolled_back() -> None:
    g, r = gov()
    g.presets["gen-a"].est_peak_mb = 10  # estimate far too optimistic
    SIZES_BACKUP = SIZES["gen-a"]
    try:
        SIZES["gen-a"] = 3000
        with pytest.raises(DoesNotFitError):
            g.ensure("gen-a")
        assert "gen-a" not in r.loaded_set
    finally:
        SIZES["gen-a"] = SIZES_BACKUP
