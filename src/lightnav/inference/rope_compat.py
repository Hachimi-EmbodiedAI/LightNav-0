"""``Qwen3VLModel.get_rope_index`` 的版本兼容层。

transformers 4.x 和 5.x 在这个方法上的签名不同，而调用点有两处（HF 后端和
vLLM 后端各一处）。这里集中处理，避免调用点各自判断。

    4.57:  get_rope_index(self, input_ids=None, image_grid_thw=None,
                          video_grid_thw=None, attention_mask=None)
    5.8 :  get_rope_index(self, input_ids, mm_token_type_ids,
                          image_grid_thw=None, video_grid_thw=None,
                          attention_mask=None, **kwargs)

5.x 新增的 ``mm_token_type_ids``（text=0 / image=1 / video=2）用于定位视觉 span；
4.x 则自己从 ``input_ids`` 里的 vision_start/vision_end 推断。所以只需在 4.x 上
把这个参数丢掉，语义等价。

为什么要这个兼容层：本机（Jetson Orin, SM 8.7）唯一带原生 sm_87 kernel 的 vLLM
轮子是 ``vllm-0.19.0+cu126-cp310``，它的元数据把 transformers 钉在 ``<5``。
而项目默认面向 transformers 5.8。两种都必须能跑。
"""

from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=8)
def _accepts_mm_token_type_ids(func: Any) -> bool:
    """Cached probe: does ``func`` accept ``mm_token_type_ids``?

    Caching matters: :func:`call_get_rope_index` sits on the per-step mrope path
    (``_patched_get_mrope_input_positions`` runs once per inference step), and
    ``inspect.signature`` costs tens of microseconds each call. The answer can
    never change within a process, so a small LRU is safe.
    """
    try:
        return "mm_token_type_ids" in inspect.signature(func).parameters
    except (TypeError, ValueError):  # builtins / C callables
        return False


def supports_mm_token_type_ids(func: Any) -> bool:
    """True if ``func`` (an unbound method) accepts ``mm_token_type_ids``."""
    return _accepts_mm_token_type_ids(func)


def call_get_rope_index(
    func: Any,
    self: Any,
    input_ids: Any,
    mm_token_type_ids: Any = None,
    **kwargs: Any,
) -> Any:
    """Call ``Qwen3VLModel.get_rope_index`` across transformers 4.x / 5.x.

    ``func`` is the unbound class method, ``self`` the (possibly fake) instance.
    Returns whatever the underlying method returns: ``(position_ids, rope_deltas)``.
    """
    if supports_mm_token_type_ids(func):
        return func(self, input_ids, mm_token_type_ids, **kwargs)
    return func(self, input_ids, **kwargs)


def describe_rope_index_compat() -> str:
    """One-line summary for startup logs (which signature is in play)."""
    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

        v5 = supports_mm_token_type_ids(Qwen3VLModel.get_rope_index)
        import transformers

        return (
            f"transformers {transformers.__version__} "
            f"get_rope_index({'5.x mm_token_type_ids' if v5 else '4.x legacy'})"
        )
    except Exception as exc:  # pragma: no cover - diagnostic only
        return f"transformers rope compat unavailable ({exc})"
