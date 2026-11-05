"""DEV ONLY (DR-010): replay a field-report bundle from the deployed app in the capped emulator.

    python -m tools.replay field-report.zip            # print the reported config + versions + probe diff
    python -m tools.replay field-report.zip --env      # emit the env vars to start the emulator identically

The report holds the session settings and the effective parameters of the last turns (no media,
no secrets), so the same configuration can be reproduced against the same bundle id.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.obs.field_report import read_field_report
from core.runtime.probe import run_probe


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--env", action="store_true")
    args = ap.parse_args(argv)
    fr = read_field_report(args.bundle.read_bytes())
    rep = fr["report"]
    cfg = rep["config"]
    if args.env:
        print(f"export MMRAG_MEMORY_CAP_MB={cfg.get('memory_cap_mb', 2700)}")
        print(f"export MMRAG_LOG_LEVEL={cfg.get('log_level', 'INFO')}")
        print(f"# bundle: {rep.get('bundle_id')} (manifest sha256 {rep.get('manifest_sha256')})")
        return 0
    print(json.dumps({k: rep[k] for k in ("created_at", "bundle_id", "manifest_sha256", "event_count")}, indent=2))
    print("session settings:", json.dumps(rep.get("session", {}), indent=2)[:2000])
    here = json.loads(run_probe().model_dump_json(by_alias=True))["values"]
    there = fr["probe"]["values"]
    print("\nprobe differences (cloud vs this machine):")
    for k in there:
        a, b = there[k].get("value"), here.get(k, {}).get("value")
        if a != b:
            print(f"  {k}: cloud={str(a)[:80]!s}  here={str(b)[:80]!s}")
    errors = [e for e in fr["events"] if e["level"] in ("ERROR", "WARNING")]
    print(f"\n{len(errors)} warning/error events; last ones:")
    for e in errors[-10:]:
        print(f"  {e['component']}.{e['event']} {json.dumps(e['fields'])[:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
