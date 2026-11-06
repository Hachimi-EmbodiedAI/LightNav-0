#!/usr/bin/env python3
"""End-to-end WebSocket smoke test against a running lightnav-serve.

Measures per-step latency over the wire (the number that decides whether the
onboard loop is fast enough) and prints a verdict.

    .venv310/bin/python scripts/smoke_orin.py --steps 8
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import statistics
import time

import numpy as np
from PIL import Image


def jpeg_b64(rgb: np.ndarray, quality: int = 90) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--server", default="ws://127.0.0.1:8050")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--instruction", default="walk to the red chair and stop")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    args = ap.parse_args()

    from websockets.sync.client import connect

    rng = np.random.default_rng(0)
    frame = np.full((args.height, args.width, 3), 110, np.uint8)
    frame[100:200, 250:400] = (200, 60, 60)
    frame = np.clip(frame.astype(np.int16) + rng.integers(-6, 6, frame.shape), 0, 255).astype(np.uint8)
    img = jpeg_b64(frame)

    with connect(args.server, max_size=64 * 1024 * 1024) as ws:

        def rpc(action, data):
            ws.send(json.dumps({"action": action, "data": data}))
            return json.loads(ws.recv())["data"]

        r = rpc("login", {"clientId": "orin-smoke"})
        print(f"login  rc={r.get('rc')} {r.get('msg')}")
        r = rpc("reset", {})
        print(f"reset  rc={r.get('rc')} {r.get('msg')}")

        # Warm up the history window first (buffer-only frames, no inference).
        warm = min(63, 8)
        for i in range(warm):
            rpc("next", {"seq": i, "image": img, "instruction": ""})

        lat, vit, llm = [], [], []
        seq = warm
        for i in range(args.steps):
            t = time.time()
            d = rpc("next", {"seq": seq, "image": img, "instruction": args.instruction})
            wall = (time.time() - t) * 1000
            seq += 1
            if d.get("rc") != 0:
                print(f"  step {i}: rc={d.get('rc')} {d.get('msg')}")
                continue
            lat.append(float(d.get("latency_ms") or wall))
            tm = d.get("timings_ms") or {}
            if "vit_ms" in tm:
                vit.append(float(tm["vit_ms"]))
            if "llm_ms" in tm:
                llm.append(float(tm["llm_ms"]))
            acts = (d.get("actions") or {}).get("actions") or []
            a0 = acts[0] if acts else None
            print(
                f"  step {i}: {lat[-1]:6.0f} ms  stop={d.get('stop')}  "
                f"rows={len(acts)}  a0={a0}  raw={str(d.get('raw_text'))[:34]}"
            )

    if not lat:
        print("FAIL: no successful steps")
        return 1

    med = statistics.median(lat)
    print()
    print(f"latency min={min(lat):.0f} med={med:.0f} max={max(lat):.0f} ms  (n={len(lat)})")
    if vit:
        print(f"  vit_ms  med={statistics.median(vit):.0f}")
    if llm:
        print(f"  llm_ms  med={statistics.median(llm):.0f}")
    hz = 1000.0 / med
    print(f"throughput = {hz:.2f} Hz  (end-to-end, single session)")
    print()
    if hz >= 4.0:
        print("VERDICT: excellent -- matches the training frame rate (4 Hz).")
        print("         v_max can stay at the default.")
    elif hz >= 2.0:
        print("VERDICT: usable -- expect visible stop-and-go at control rate.")
        print("         Recommend v_max ~0.35 m/s.")
    elif hz >= 1.0:
        print("VERDICT: marginal -- slow but drivable for object-goal navigation.")
        print("         Recommend v_max ~0.25 m/s, objnav only.")
    else:
        print("VERDICT: too slow for closed-loop control on this board.")
        print("         Use the remote-GPU deployment instead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
