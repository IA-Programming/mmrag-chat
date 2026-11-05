"""History policy (FR-025): full | window(N) | none; contextualization: off | concat_last_user | llm_rewrite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

POLICIES = ("full", "window", "none")
CONTEXT_MODES = ("off", "concat_last_user", "llm_rewrite")

REWRITE_PROMPT = (
    "Rewrite the user's last question so it can be understood without the conversation. "
    "Resolve pronouns and ellipsis using the previous turns. Output only the rewritten question."
)


@dataclass(frozen=True)
class HistoryConfig:
    policy: str = "window"
    window: int = 4
    contextualize: str = "llm_rewrite"  # EXP-HIST-01


def select_history(turns: list[dict[str, Any]], cfg: HistoryConfig) -> list[dict[str, Any]]:
    """Previous turns (role/text dicts, oldest first) to include, before the token budget trims them."""
    prior = [t for t in turns if t.get("role") in ("user", "assistant") and t.get("text")]
    if cfg.policy == "none":
        return []
    if cfg.policy == "window":
        n = max(0, cfg.window) * 2  # N exchanges = N user + N assistant messages
        return prior[-n:] if n else []
    return prior


def retrieval_query(query: str, turns: list[dict[str, Any]], cfg: HistoryConfig) -> str:
    """Query text used for retrieval (`llm_rewrite` is done by the pipeline: one extra call)."""
    if cfg.contextualize == "concat_last_user":
        last_user = next((t["text"] for t in reversed(turns) if t.get("role") == "user" and t.get("text")), "")
        return f"{last_user} {query}".strip() if last_user else query
    return query


def rewrite_messages(query: str, turns: list[dict[str, Any]], cfg: HistoryConfig) -> list[dict[str, Any]]:
    hist = select_history(turns, HistoryConfig(cfg.policy if cfg.policy != "none" else "window", cfg.window or 2, "off"))
    convo = "\n".join(f"{t['role']}: {t['text'][:500]}" for t in hist)
    return [
        {"role": "system", "content": REWRITE_PROMPT},
        {"role": "user", "content": f"Conversation:\n{convo}\n\nLast question: {query}\n\nRewritten question:"},
    ]
