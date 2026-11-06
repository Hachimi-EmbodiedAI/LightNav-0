#!/usr/bin/env python3
"""Verify that a transformers-4.x tokenizer load is byte-identical to 5.x.

The Orin deployment runs transformers 4.57 (forced: the only vLLM wheel with
native sm_87 kernels pins <5), while the checkpoint's ``tokenizer_config.json``
was written in the 5.x format. ``lightnav.inference.tokenizer_compat`` bridges
the two by renaming ``extra_special_tokens`` -> ``additional_special_tokens``.

Renaming a field is exactly the kind of change that can silently shift token ids,
and a shifted id in ``<act_l0_*>`` / ``<apos_*>` decodes to a different waypoint
chunk. So the claim "semantically equivalent" has to be MEASURED, not asserted.

This script dumps a fingerprint of the tokenizer + a real prompt encoding. Run it
in each environment and diff the two JSON files:

    # on the 3.11 venv (transformers 5.8)
    .venv311/bin/python scripts/verify_tokenizer_parity.py --out /tmp/tok58.json

    # on the 3.10 venv (transformers 4.57)
    .venv310/bin/python scripts/verify_tokenizer_parity.py --out /tmp/tok457.json

    diff <(python -m json.tool /tmp/tok58.json) <(python -m json.tool /tmp/tok457.json)

An empty diff (apart from the reported transformers version) means the 4.x path
reproduces the trained tokenization exactly.

Exits non-zero if an in-process self-check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Tokens whose id MUST match: they are the model's action/pointing vocabulary.
CRITICAL_TOKENS = [
    "<navigation_task>",
    "</navigation_task>",
    "<move_forward>",
    "<turn_left>",
    "<turn_right>",
    "<stop>",
    "<act_l0_0>",
    "<act_l1_0>",
    "<act_l2_0>",
    "<apos_0>",
    "<opos_0>",
    "<tpos_0>",
    "<traj_0>",
]

# A prompt shaped like the real VLN template, so timestamp/task-token handling and
# special-token boundaries are exercised the same way inference will hit them.
PROBE_TEXT = (
    "Imagine you are a robot programmed for navigation tasks. "
    "You have been assigned the task: <navigation_task>walk to the red chair and stop"
    "</navigation_task>. Analyze these observations to determine your next 10 actions."
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="checkpoints/LightNav-0")
    ap.add_argument("--out", default="-", help="output JSON path ('-' for stdout)")
    ap.add_argument("--no-shim", action="store_true", help="do not install the 4.x tokenizer shim")
    args = ap.parse_args()

    if args.no_shim:
        import os

        os.environ["LIGHTNAV_NO_TOKENIZER_SHIM"] = "1"

    import transformers

    from lightnav.inference.tokenizer_compat import (
        describe_tokenizer_compat,
        patch_autotokenizer,
    )

    shim_active = patch_autotokenizer()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    def tid(tok_str: str):
        """Token string -> id, or None if the token is not in this vocabulary."""
        try:
            ids = tok.convert_tokens_to_ids(tok_str)
        except Exception:
            return None
        if ids is None:
            return None
        unk = getattr(tok, "unk_token_id", None)
        return None if (unk is not None and ids == unk) else ids

    enc = tok(PROBE_TEXT, add_special_tokens=False)
    enc_special = tok(PROBE_TEXT, add_special_tokens=True)

    report = {
        "transformers": transformers.__version__,
        "shim_active": shim_active,
        "compat": describe_tokenizer_compat(),
        "tokenizer_class": type(tok).__name__,
        "vocab_size": tok.vocab_size,
        "len": len(tok),
        "added_tokens": len(getattr(tok, "added_tokens_decoder", {}) or {}),
        "bos": tid(getattr(tok, "bos_token", "") or ""),
        "eos": tid(getattr(tok, "eos_token", "") or ""),
        "pad": tid(getattr(tok, "pad_token", "") or ""),
        "unk": tid(getattr(tok, "unk_token", "") or ""),
        "critical_tokens": {t: tid(t) for t in CRITICAL_TOKENS},
        "probe_no_special": {
            "input_ids": enc["input_ids"],
            "n_tokens": len(enc["input_ids"]),
        },
        "probe_with_special": {
            "input_ids": enc_special["input_ids"],
            "n_tokens": len(enc_special["input_ids"]),
        },
    }

    # Self-check: the probe must keep the action tokens atomic. If a special token
    # were split into pieces, its id would not appear as a single element.
    stop_id = report["critical_tokens"]["<stop>"]
    nav_id = report["critical_tokens"]["<navigation_task>"]
    atomic = []
    if stop_id is not None and stop_id in enc["input_ids"]:
        atomic.append("<stop>")
    if nav_id is not None and nav_id in enc["input_ids"]:
        atomic.append("<navigation_task>")

    blob = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out == "-":
        print(blob)
    else:
        Path(args.out).write_text(blob, encoding="utf-8")
        crit = report["critical_tokens"]
        missing = [k for k, v in crit.items() if v is None]
        print(f"wrote {args.out}")
        print(f"  transformers {report['transformers']}  shim={shim_active}")
        print(f"  vocab {report['vocab_size']} / len {report['len']} / added {report['added_tokens']}")
        print(f"  probe {report['probe_no_special']['n_tokens']} tokens (no special)")
        print(f"  atomic special tokens seen in probe: {atomic or 'NONE'}")
        if missing:
            print(f"  WARN: tokens absent from vocab: {missing}")

    if not atomic:
        print("SELF-CHECK FAILED: probe text did not keep special tokens atomic", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
