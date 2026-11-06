#!/usr/bin/env python3
"""Build the LightNav vLLM engine and report per-step latency -- the Orin gate.

    .venv310/bin/python scripts/load_engine.py --bench 5

Run it the way the docs prescribe: `MODE_30W`, and with `LD_LIBRARY_PATH` set the
way `scripts/serve_orin.sh` sets it (that script also has the preflight checks).
On MAXN this machine hard-powers-off during weight load -- see docs/JETSON_ORIN.md.

Useful when you want a latency number without the WebSocket layer:
`scripts/smoke_orin.py` measures the same thing end-to-end through the server.

Prints each stage with a timestamp so a hard power-off is distinguishable from a
Python exception: on a crash the last printed line names the stage that died.
Nothing here touches the network -- the checkpoint must already be local.
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import signal
import sys
import time
import traceback

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="checkpoints/LightNav-0")
    ap.add_argument("--task", default="vlnce",
                    help="eval_config.json task key: vlnce | trackvla")
    ap.add_argument("--prompt", default="vlnce_traj",
                    help="prompt family for generate: vlnce_traj | tracking")
    ap.add_argument("--gpu-mem", type=float, default=0.45)
    ap.add_argument("--bench", type=int, default=0)
    args = ap.parse_args()

    faulthandler.enable()

    def _sig(signum, _f):
        log(f"signal {signum} ({signal.Signals(signum).name})")
        sys.exit(3)

    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, _sig)

    log(f"start pid={os.getpid()}")

    try:
        import torch

        log(
            f"torch {torch.__version__} cuda={torch.version.cuda} "
            f"avail={torch.cuda.is_available()} "
            f"cap={torch.cuda.get_device_capability(0)}"
        )
        # Prove the kernel image is present before asking vLLM to allocate 10 GB.
        probe = torch.zeros(8, 8, device="cuda", dtype=torch.bfloat16)
        probe = probe @ probe
        torch.cuda.synchronize()
        log(f"cuda kernel probe OK (sum={float(probe.sum()):.1f})")

        from lightnav.inference.config import InferenceConfig
        from lightnav.inference.engine import build_engine

        log("lightnav imports OK")

        cfg = InferenceConfig(
            model_path=args.model,
            backend="vllm_local",
            device="cuda",
            gpu_memory_utilization=args.gpu_mem,
            max_num_seqs=1,
        )
        log("building engine (loads ~9.7 GB of weights)...")
        engine, bundle = build_engine(cfg, task_type=args.task, max_new_tokens=8)
        log("ENGINE BUILD OK")
        log(
            f"video_size={bundle.video_size} hist={bundle.num_history_frames} "
            f"fps={bundle.video_fps} horizon={bundle.predict_horizon} "
            f"slowfast={bool(bundle.slowfast_tiers)} action={bundle.action_method}"
        )

        if args.bench:
            import numpy as np

            h, w = bundle.video_size
            # SlowFast tiers need a full episode worth of frames: the mid/long
            # tiers sample by age, and an 8-frame buffer leaves them with zero
            # samples (grid ending in 0). Feed num_history_frames instead.
            n = int(bundle.num_history_frames)
            frames = torch.stack(
                [torch.from_numpy(np.full((h, w, 3), 110, np.uint8)) for _ in range(n)]
            )
            log(f"benchmarking {args.bench} steps at {w}x{h}, history={n}")
            lat = []
            for i in range(args.bench):
                t = time.time()
                text, _ = engine.generate_from_frames(
                    frames, "walk to the red chair and stop", task_type=args.prompt
                )
                dt = (time.time() - t) * 1000.0
                lat.append(dt)
                log(f"  step {i}: {dt:7.0f} ms -> {text[:50]}")
            lat.sort()
            med = lat[len(lat) // 2]
            log(f"latency min={lat[0]:.0f} med={med:.0f} max={lat[-1]:.0f} ms")
            log(f"THROUGHPUT = {1000.0 / med:.2f} Hz")

        log("SUCCESS")
        return 0
    except Exception:
        traceback.print_exc()
        log("FAILED (python exception)")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
