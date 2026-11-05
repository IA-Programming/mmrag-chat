"""DEV ONLY (DR-007): verify bundle/sampling_presets.json against the live model cards.

    python -m tools.verify_sampling_presets            # report drift (exit 1 on missing provenance)
    python -m tools.verify_sampling_presets --update   # write revision + checked_at for matching presets

For every preset: fetch the card (README.md) at the repo's current revision with HF_TOKEN from the
environment, extract recommended sampling values (temperature, top_p, top_k, min_p,
repetition/repeat penalty, presence penalty), compare with the stored default mode, and report:
  ok      every value found on the card matches the stored preset
  drift   the card states a different value -> needs human review (never auto-applied)
  silent  the card states no values -> status stays partial / neutral values
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PRESETS = ROOT / "bundle" / "sampling_presets.json"
CARD_REPO = {  # card repos (the base model card holds the recommendations for most families)
    "lfm3-vl-2b": "LiquidAI/LFM3-VL-2B",
    "lfm3-1.7b": "LiquidAI/LFM3-1.7B",
    "lfm3-omni-2b": "LiquidAI/LFM3-Omni-2B",
    "qwen3.5-0.8b": "Qwen/Qwen3.5-0.8B",
    "qwen3.5-2b": "Qwen/Qwen3.5-2B",
    "minicpm-v-4.7-1b": "openbmb/MiniCPM-V-4.7-1B",
    "ministral-3-3b": "mistralai/Ministral-3-3B-Instruct-2512",
    "lfm2.5-vl-1.6b": "LiquidAI/LFM2.5-VL-1.6B",
    "gemma-4-e2b": "google/gemma-4-E2B-it",
}
FIELDS = {
    "temperature": r"temp(?:erature)?",
    "top_p": r"top[_\s-]?p",
    "top_k": r"top[_\s-]?k",
    "min_p": r"min[_\s-]?p",
    "repeat_penalty": r"(?:repetition|repeat)[_\s-]?penalty",
    "presence_penalty": r"presence[_\s-]?penalty",
}
NUM = r"[\s`*:=\"']{0,6}(-?\d+(?:\.\d+)?)"


def clean(card: str) -> str:
    """Drop HTML tags/styles (CSS numbers caused false drift) and keep prose + code."""
    card = re.sub(r"<style.*?</style>", " ", card, flags=re.S | re.I)
    card = re.sub(r"<[^>]+>", " ", card)
    return card


def extract(card: str) -> dict[str, list[float]]:
    found: dict[str, list[float]] = {}
    card = clean(card)
    for f, pat in FIELDS.items():
        # (?<![A-Za-z_]) skips e.g. audio_temperature / audio_repetition_penalty (speech output only)
        for m in re.finditer(r"(?<![A-Za-z_])" + pat + NUM, card, re.I):
            found.setdefault(f, []).append(float(m.group(1)))
    return found


def fetch(repo: str) -> tuple[str, str]:
    from huggingface_hub import HfApi, hf_hub_download

    sha = HfApi().model_info(repo).sha or ""
    path = hf_hub_download(repo, "README.md", revision=sha)
    return Path(path).read_text(errors="replace"), sha


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true")
    args = ap.parse_args(argv)
    data: dict[str, Any] = json.loads(PRESETS.read_text())
    report = []
    missing = []
    for pid, p in data["presets"].items():
        if not p.get("source_url"):
            missing.append(pid)
        repo = CARD_REPO.get(pid)
        if not repo:
            continue
        try:
            card, sha = fetch(repo)
        except Exception as exc:
            report.append({"preset": pid, "status": "error", "detail": str(exc)[:200]})
            continue
        found = extract(card)
        stored = p["modes"][p["default_mode"]]
        all_modes = [m for m in p["modes"].values()]
        drift = {}
        for f, vals in found.items():
            known = {m.get(f) for m in all_modes if m.get(f) is not None}
            if known and not any(abs(v - k) < 1e-6 for v in vals for k in known):
                drift[f] = {"card": sorted(set(vals))[:6], "stored": sorted(known)}
        status = "silent" if not found else ("drift" if drift else "ok")
        report.append({"preset": pid, "card": repo, "revision": sha[:12], "status": status, "card_values": {k: sorted(set(v))[:6] for k, v in found.items()},
                       "stored_default": stored, "drift": drift})  # fmt: skip
        if args.update and status in ("ok", "silent"):
            p["revision"] = sha
            p["checked_at"] = time.strftime("%Y-%m-%d")
    for r in report:
        print(json.dumps(r))
    if args.update:
        PRESETS.write_text(json.dumps(data, indent=2) + "\n")
    if missing:
        print("presets without source_url:", missing, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
