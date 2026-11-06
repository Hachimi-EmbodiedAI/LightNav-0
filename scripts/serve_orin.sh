#!/usr/bin/env bash
# One-command lightnav-serve launcher for NVIDIA Jetson AGX Orin (SM 8.7).
#
# Sibling of scripts/serve_thor.sh, adapted for Orin rather than Thor:
#   * NO fp8. Orin is Ampere: there are no fp8 tensor cores, so
#     `--quantization fp8_llm_only` is both useless and slower. bf16 only.
#   * NO VLLM_DISABLED_KERNELS. That guard exists because three fp8 kernels are
#     compiled for the SM 10.0 family only; it is a no-op (harmless) here but we
#     do not set it, to keep the bf16 path exactly stock.
#   * Unified-memory budget tuned for a robot that ALSO runs ROS 2, SLAM and the
#     web console: default gpu_memory_utilization 0.45, KV cache pinned to 2 GiB
#     so the fraction does not decide the cache size (see docs/CONFIGURATION.md).
#   * LD_LIBRARY_PATH bootstrap is REQUIRED here, not optional: the cu126 torch
#     wheel does not bundle CUDA libraries (torch/lib/ has libtorch_cuda.so but
#     no libcublas/libcudnn/libcudart), so every libcublas.so.12 etc. comes from
#     the pip `nvidia-*` packages under site-packages/nvidia/*/lib.
#
#   bash scripts/serve_orin.sh                 # start, wait for READY
#   RECORD=1 bash scripts/serve_orin.sh        # + episode recording
#   bash scripts/serve_orin.sh --stop          # stop THIS PORT's server
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Which venv to serve from.
#
# Orin needs Python 3.10: the only vLLM build with native sm_87 kernels is
# `vllm-0.19.0+cu126-cp310`, and the matching torch is the jetson-ai-lab cp310
# wheel. A `.venv` built for 3.11 with the pytorch.org aarch64 wheel looks fine
# (`cuda.is_available()` is True) but contains only sm_80/sm_90 kernels and dies
# with "no kernel image is available" on the first launch.
#
# So: prefer .venv310, and if VENV is unset fall back to .venv only when no
# Orin-capable venv exists -- the preflight will then report the real problem.
default_venv() {
    if [ -x "$REPO_ROOT/.venv310/bin/python" ]; then
        echo "$REPO_ROOT/.venv310"
    else
        echo "$REPO_ROOT/.venv"
    fi
}
VENV=${VENV:-$(default_venv)}

# ----------------------------------------------------------------- knobs ----
MODEL_PATH=${MODEL_PATH:-$REPO_ROOT/checkpoints/LightNav-0}
ACTION_TOKENIZER_BUNDLE=${ACTION_TOKENIZER_BUNDLE:-}  # only for ckpts with no decoder
TASK=${TASK:-vln}
PORT=${PORT:-8050}
HOST=${HOST:-127.0.0.1}           # onboard: loopback only. No auth of its own.
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.45}  # unified memory: leave room for ROS/SLAM/web
KV_CACHE_GIB=${VLN_KV_CACHE_GIB:-2}
VIT_CACHE=${VLN_VIT_CACHE_ENTRIES:-512}
TEMPERATURE=${TEMPERATURE:-0.0}
RECORD=${RECORD:-0}
RECORD_DIR=${RECORD_DIR:-$REPO_ROOT/vln_episodes}
LOG_DIR=${LOG_DIR:-$REPO_ROOT/logs}
LOG=${LOG:-$LOG_DIR/server_${PORT}.log}
READY_DIR="$REPO_ROOT/.servers_ready"
READY=${READY:-$READY_DIR/port${PORT}.ready}

# Only THIS port's server; the bracket keeps pkill from matching us.
PATTERN="lightnav[.]serving[.]ws_server.*--port ${PORT}([^0-9]|\$)"

# ------------------------------------------------------------------ stop ----
drain() {
    pkill -f "$PATTERN" 2>/dev/null || true
    # Jetson iGPUs report memory.used as "[N/A]": fall back to the process check
    # alone. A recording server flushes segments first, so wait rather than respawn.
    for _ in $(seq 1 60); do
        procs=$(pgrep -f "$PATTERN" 2>/dev/null | wc -l) || procs=0
        if [ "$procs" -eq 0 ]; then return 0; fi
        sleep 2
    done
    return 1
}

if [ "${1:-}" = "--stop" ]; then
    echo "[serve_orin] stopping port $PORT..."
    drain || echo "[serve_orin] WARN: still draining after timeout" >&2
    rm -f "$READY"
    echo "[serve_orin] stopped."
    exit 0
fi

# ----------------------------------------------------------------- start ----
PY="$VENV/bin/python"
[ -x "$PY" ] || { echo "[serve_orin] no venv python at $PY (set VENV=...)" >&2; exit 1; }
[ -d "$MODEL_PATH" ] || { echo "[serve_orin] MODEL_PATH not found: $MODEL_PATH" >&2; exit 1; }

# --- dependency guards: fail here, with the fix in the message, rather than deep
# --- inside the vLLM engine after a 2-minute weight load.
preflight() {
    local out
    out=$(CUDA_LIB_DIR="$CUDA_LIB_DIR" VENV="$VENV" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" "$PY" - <<'PYEOF' 2>&1
import importlib.metadata as md, os, sys, torch
bad = []
venv = os.environ.get("VENV", "?")

# Report which SM list the loaded torch wheel was actually built for, so a
# "no kernel image" failure names its own cause.
def gencode():
    try:
        p = os.path.join(os.path.dirname(torch.__file__), "lib", "libtorch_cuda.so")
        d = open(p, "rb").read()
        i = d.find(b"-gencode")
        return d[i:i + 200].split(b"\x00")[0].decode("latin1") if i >= 0 else "(none)"
    except Exception as exc:
        return f"(unreadable: {exc})"

try:
    dsl = md.version("nvidia-cutlass-dsl")
except md.PackageNotFoundError:
    dsl = None

# vLLM's own runtime deps. A `--no-deps` install of the vllm wheel (which is what
# you want on Orin, so it cannot drag in datacenter CUDA wheels) leaves all of
# these absent, and the failure surfaces one at a time, deep in an import chain:
#   import pybase64 -> ModuleNotFoundError, from vllm.multimodal.media.audio
_DEP_MODULES = {
    "pybase64": "pybase64", "cachetools": "cachetools", "sentencepiece": "sentencepiece",
    "blake3": "blake3", "cpuinfo": "py-cpuinfo", "google.protobuf": "protobuf",
    "fastapi": "fastapi", "aiohttp": "aiohttp", "openai": "openai",
    "prometheus_client": "prometheus-client", "tiktoken": "tiktoken",
    "lmformatenforcer": "lm-format-enforcer", "diskcache": "diskcache", "lark": "lark",
    "depyf": "depyf", "cloudpickle": "cloudpickle", "watchfiles": "watchfiles",
    "pythonjsonlogger": "python-json-logger", "cbor2": "cbor2", "ijson": "ijson",
    "setproctitle": "setproctitle", "msgspec": "msgspec", "gguf": "gguf",
    "mistral_common": "mistral-common", "partial_json_parser": "partial-json-parser",
    "prometheus_fastapi_instrumentator": "prometheus-fastapi-instrumentator",
}
_missing = []
for mod, dist in _DEP_MODULES.items():
    try:
        __import__(mod)
    except Exception:
        _missing.append(dist)
if _missing:
    bad.append("missing vLLM runtime deps: " + " ".join(sorted(set(_missing)))
               + "\n    -> install with --no-deps of the vllm wheel, then:\n"
                 "       uv pip install " + " ".join(sorted(set(_missing))))
if dsl is not None:
    # vLLM 0.19.x only asks for >=4.4.0.dev1, so a plain install resolves to a
    # recent build that dropped cutlass.cute.core.ThrMma; quack-kernels still
    # imports it and the engine dies while starting.
    try:
        from cutlass.cute import core
        if not hasattr(core, "ThrMma"):
            bad.append(f"nvidia-cutlass-dsl=={dsl} has no cutlass.cute.core.ThrMma"
                       " -> uv pip install --no-deps 'nvidia-cutlass-dsl==4.5.2'")
    except Exception as exc:
        bad.append(f"cutlass.cute import failed ({exc})")

if not torch.cuda.is_available():
    bad.append(f"torch {torch.__version__} reports cuda unavailable"
               " -> you are on the CPU wheel; install the jetson-ai-lab torch build")
else:
    cap = torch.cuda.get_device_capability(0)
    if cap != (8, 7):
        bad.append(f"CUDA capability {cap} is not Orin (8,7) -- wrong device?")
    # is_available()/get_device_capability() only query driver attributes: they pass
    # even on a wheel built for the wrong SM. Actually launch a kernel, and then a
    # cuBLAS one -- the two fail differently and the messages are misleading:
    #   torch kernel missing -> "no kernel image is available"
    #   cuBLAS missing sm_87 -> "CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate"
    # The second one looks like OOM but is an architecture mismatch in libcublas.
    try:
        torch.zeros(4, device="cuda")
        torch.cuda.synchronize()
    except Exception as exc:
        bad.append(
            f"torch CUDA kernel failed: {str(exc).splitlines()[0]}\n"
            f"    venv   : {venv}\n"
            f"    torch  : {torch.__version__}\n"
            f"    built  : {gencode()}\n"
            "    -> this torch wheel has no sm_87 kernels. On Orin you need the "
            "jetson-ai-lab cp310 build in a Python 3.10 venv (.venv310):\n"
            "       uv venv --python 3.10 .venv310 && source .venv310/bin/activate\n"
            "       uv pip install --index https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/ \\\n"
            "           --extra-index-url https://pypi.org/simple \\\n"
            "           --index-strategy unsafe-best-match torch==2.10.0 torchvision==0.25.0"
        )
    else:
        try:
            a = torch.randn(64, 64, device="cuda")
            (a @ a)
            torch.cuda.synchronize()
        except Exception as exc:
            first = str(exc).splitlines()[0]
            msg = f"cuBLAS failed: {first}"
            if "CUBLAS_STATUS_ALLOC_FAILED" in first or "no kernel image" in first:
                msg += (
                    " -> libcublas has no sm_87 kernels. The pip nvidia-* wheels are"
                    " datacenter builds (sm_90/sm_100 only). Do NOT put"
                    " site-packages/nvidia/*/lib on LD_LIBRARY_PATH; use the JetPack"
                    " CUDA runtime instead (see docs/JETSON_ORIN.md)"
                )
            bad.append(msg)

if bad:
    print("\n".join("  - " + b for b in bad))
    sys.exit(1)
print(f"  torch {torch.__version__} | {torch.cuda.get_device_name(0)} {torch.cuda.get_device_capability(0)}"
      f" | built {gencode()} | cutlass-dsl {dsl} | venv {os.path.basename(venv)}")
PYEOF
) || { echo "[serve_orin] preflight FAILED:" >&2; echo "$out" >&2; exit 1; }
    echo "[serve_orin] preflight OK:$out"
}

# CUDA library resolution.
#
# DEFAULT (NV_LIBS=0, "jetson"): prepend ONLY the JetPack CUDA runtime dir, never
#   the pip nvidia/*/lib dirs. Two distinct problems, both real:
#
#   1. The PyPI `nvidia-*-cu12` wheels are datacenter builds -- libcublas ships only
#      sm_90 and its embedded PTX is `.target sm_90`, which cannot JIT *down* to
#      sm_87. On LD_LIBRARY_PATH they produce
#          CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`
#      (a misleading OOM-shaped error).
#   2. `libcupti.so.12` is a symlink to `libcupti.so.2024.3.1`. ldconfig will not
#      index that real filename (its version segment is not a plain dotted number),
#      so `import torch` fails with `libcupti.so.12: cannot open shared object file`
#      unless the directory is on the search path. libtorch_cpu links it even
#      though nothing here profiles.
#
# Prepending the CUDA dir fixes (2) unconditionally and is harmless for (1).
# NV_LIBS=1 additionally prepends the pip dirs, for debugging only.
CUDA_LIB_DIR=${CUDA_LIB_DIR:-/usr/local/cuda/targets/aarch64-linux/lib}
if [ -d "$CUDA_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="$CUDA_LIB_DIR:${LD_LIBRARY_PATH:-}"
fi

# libcudss is the ONE pip CUDA library that must stay: it is not in the JetPack apt
# repo, and libtorch_cuda.so lists it in NEEDED. Guard the trap that comes with it:
#
#   nvidia-cudss-cu12 declares a dependency on nvidia-cublas-cu12, which resolves
#   to a CUDA *12.9* libcublas whenever pip/uv installs it for us. Worse,
#   libcudss.so.0 carries RUNPATH=$ORIGIN/../../cublas/lib, so merely having
#   site-packages/nvidia/cublas/lib present makes the 12.9 library load -- and a
#   12.9 cuBLAS against a 12.6 driver fails exactly like a wrong-architecture
#   build, with CUBLAS_STATUS_ALLOC_FAILED from cublasCreate. Version mismatch and
#   arch mismatch are indistinguishable from the error code alone.
#
# The fix is at install time, not here: delete site-packages/nvidia/cublas and
# leave symlinks to the JetPack libraries in its place (see docs/JETSON_ORIN.md).
# This block only verifies it and warns loudly if the 12.9 copy came back.
for d in "$VENV"/lib/python*/site-packages/nvidia/cublas/lib; do
    [ -d "$d" ] || continue
    _bl="$d/libcublas.so.12"
    if [ -f "$_bl" ] && [ ! -L "$_bl" ]; then
        _sz=$(stat -c %s "$_bl" 2>/dev/null || echo 0)
        if [ "$_sz" -gt 50000000 ]; then
            echo "[serve_orin] WARN: $d holds a real (non-symlink) libcublas.so.12 (${_sz}B)." >&2
            echo "[serve_orin]       If cuBLAS fails with CUBLAS_STATUS_ALLOC_FAILED this is why:" >&2
            echo "[serve_orin]         rm -rf $d && mkdir -p $d && ln -sf /usr/local/cuda/targets/aarch64-linux/lib/libcublas.so.12 $d/" >&2
            echo "[serve_orin]         ln -sf /usr/local/cuda/targets/aarch64-linux/lib/libcublasLt.so.12 $d/" >&2
        fi
    fi
done

# Appended, never prepended: the CUDA dir must win so only cudss itself is taken
# from site-packages.
for d in "$VENV"/lib/python*/site-packages/nvidia/cu12/lib; do
    [ -d "$d" ] && export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$d"
done

NV_LIBS=${NV_LIBS:-0}
if [ "$NV_LIBS" = "1" ]; then
    _nv=""
    for d in "$VENV"/lib/python*/site-packages/nvidia/*/lib; do
        [ -d "$d" ] && _nv="${_nv:+$_nv:}$d"
    done
    if [ -n "$_nv" ]; then
        export LD_LIBRARY_PATH="$_nv:${LD_LIBRARY_PATH:-}"
        echo "[serve_orin] WARN: NV_LIBS=1 prepends pip nvidia libs (no sm_87 kernels on Orin)" >&2
    fi
fi


# --------------------------------------------------------------- power mode --
# Observed on this board (2026-11-06): the machine hard-powers-off during the
# 9.7 GB weight load in MAXN. It is NOT a power-ceiling problem -- a monitor
# caught the last reading at 2.1 W / 306 MHz -- so the cause is still open
# (suspect the supply rail itself or a PMIC/driver event). MODE_30W loads the
# same model reliably and peaks at 12.5 W.
#
# This is a WARNING, not a gate: the operator may have a supply that handles
# MAXN, or may be chasing this very bug. Set MAXN_WARN=0 to silence it.
_nvp=$(nvpmodel -q 2>/dev/null | tr '\n' ' ')
if [ "${MAXN_WARN:-1}" = "1" ]; then
    case "$_nvp" in
        *MAXN*)
            echo "[serve_orin] WARN: power mode is MAXN." >&2
            echo "[serve_orin]       On this board MAXN has hard-powered-off during weight load" >&2
            echo "[serve_orin]       (no kernel log; last power reading was ~2 W, so it is not a" >&2
            echo "[serve_orin]       wattage ceiling). Continuing anyway -- Ctrl-C now if unwanted." >&2
            echo "[serve_orin]       Known-good alternative:  sudo nvpmodel -m 2   # MODE_30W" >&2
            echo "[serve_orin]       Silence this notice with MAXN_WARN=0." >&2
            sleep 3
            ;;
        *"MODE 15W"*)
            echo "[serve_orin] WARN: MODE_15W -- will work, at roughly half the 30W GPU clock." >&2
            ;;
    esac
fi

preflight

mkdir -p "$LOG_DIR" "$READY_DIR"
if ! drain; then
    # Never race a still-flushing server for the unified memory pool.
    echo "[serve_orin] ERROR: previous server did not exit; refusing to start a second one" >&2
    exit 1
fi
rm -f "$READY"

# src layout: `-m lightnav...` must resolve even without `pip install -e .`.
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

# Air-gapped robots: don't stall on vLLM's telemetry post.
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
# The vllm_local backend extracts the ViT from the engine: it must live in-process.
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLN_EVAL_TEMPERATURE="$TEMPERATURE"
# Never pick a quantization here (see the header): bf16 on Ampere.
unset VLLM_QUANT VLLM_DISABLED_KERNELS 2>/dev/null || true

# Small per-step CPU ops are dominated by thread-launch overhead when fanned
# across many cores -- cap at min(16, nproc) on Orin's 12-core cluster set.
CORES=$(nproc 2>/dev/null || echo 8)
THREADS=${THREADS:-$(( CORES < 16 ? CORES : 16 ))}

extra_args=()
[ -n "$ACTION_TOKENIZER_BUNDLE" ] && extra_args+=(--action_tokenizer_bundle "$ACTION_TOKENIZER_BUNDLE" --horizon 10)
[ -n "$MAX_BATCH_SIZE" ] && extra_args+=(--max_batch_size "$MAX_BATCH_SIZE")
[ "$RECORD" = "1" ] && { mkdir -p "$RECORD_DIR"; extra_args+=(--record_dir "$RECORD_DIR"); }

echo "[serve_orin] bind=$HOST:$PORT task=$TASK quant=bf16 gpu_mem_util=$GPU_MEM_UTIL kv=${KV_CACHE_GIB}GiB"
echo "[serve_orin] model=$MODEL_PATH  log=$LOG"

TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_${USER:-$(id -un)}_port${PORT}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONUNBUFFERED=1 \
VLN_KV_CACHE_GIB="$KV_CACHE_GIB" VLN_VIT_CACHE_ENTRIES="$VIT_CACHE" \
OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS \
    nohup "$PY" -u -m lightnav.serving.ws_server \
        --task "$TASK" \
        --model_path "$MODEL_PATH" \
        --backend vllm_local \
        --gpu_memory_utilization "$GPU_MEM_UTIL" \
        --host "$HOST" \
        --port "$PORT" \
        --ready_file "$READY" \
        ${extra_args[@]+"${extra_args[@]}"} \
        >"$LOG" 2>&1 &

SERVER_PID=$!
echo "[serve_orin] pid=$SERVER_PID  waiting for READY (first start: weights + warm-up + CUDA graphs, be patient)..."
for _ in $(seq 1 180); do
    [ -f "$READY" ] && break
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[serve_orin] FAILED before ready -- last log lines:" >&2
        tail -30 "$LOG" >&2
        exit 1
    fi
    sleep 5
done
[ -f "$READY" ] || { echo "[serve_orin] TIMEOUT waiting for READY" >&2; tail -30 "$LOG" >&2; exit 1; }

echo "[serve_orin] READY  ->  ws://127.0.0.1:$PORT"
