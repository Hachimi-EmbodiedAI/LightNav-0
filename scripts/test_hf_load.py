#!/usr/bin/env python3
"""Diagnostic: load the checkpoint with plain transformers, bypassing vLLM.

This is the control experiment that isolated the Orin power-off to vLLM's loader.
Sequential HF loading peaks far below vLLM's sharded-concurrent load + immediate
CUDA-graph capture, so on MAXN this script completes where `serve_orin.sh` powers
off. If THIS also powers off, suspect the supply rather than vLLM.

    .venv310/bin/python scripts/test_hf_load.py

It is not part of the serving path and needs no server running.
"""
import faulthandler, sys, time, traceback
faulthandler.enable()
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

log("importing torch/transformers")
try:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    log(f"torch {torch.__version__} avail={torch.cuda.is_available()}")

    # transformers 4.x needs the 5.x checkpoint fields renamed first.
    from lightnav.inference.config_compat import (
        patch_qwen3vl_rotary,
        describe_rope_compat,
    )
    from lightnav.inference.tokenizer_compat import patch_autotokenizer

    log(f"tokenizer shim: {patch_autotokenizer()}")
    log(f"rope shim     : {patch_qwen3vl_rotary()}  {describe_rope_compat()}")

    log("loading processor")
    proc = AutoProcessor.from_pretrained("checkpoints/LightNav-0", trust_remote_code=True)
    log(f"processor OK: {type(proc).__name__}")
    log("loading model on CPU (bf16) -- no CUDA yet")
    m = AutoModelForImageTextToText.from_pretrained(
        "checkpoints/LightNav-0", dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    log(f"model on CPU OK, params={sum(p.numel() for p in m.parameters())/1e9:.2f}B")
    log("moving to cuda...")
    m = m.to("cuda")
    torch.cuda.synchronize()
    log(f"model on GPU OK, allocated={torch.cuda.memory_allocated()/1e9:.2f}GB")
    log("SUCCESS")
except Exception:
    traceback.print_exc(); log("FAILED"); sys.exit(1)
