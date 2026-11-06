"""transformers 4.x / 5.x 的配置字段兼容层。

**为什么需要**

transformers 5.0 把 RoPE 配置从 ``rope_scaling`` 改名为 ``rope_parameters``
（与 ``additional_special_tokens`` -> ``extra_special_tokens`` 是同一批破坏性
改名的组成部分）。LightNav-0 的 checkpoint 按 5.x 写，字段内容与旧名完全等价：

    5.x  config.text_config.rope_parameters = {
           "mrope_interleaved": true,
           "mrope_section": [24, 20, 20],
           "rope_theta": 5000000,
           "rope_type": "default",
         }
    4.x  config.text_config.rope_scaling    = 上面那个 dict（或 None）

在 4.57 上 ``Qwen3VLTextRotaryEmbedding.__init__`` 直接 ``config.rope_scaling.get(...)``，
于是 ``AttributeError: 'NoneType' object has no attribute 'get'``。

**做法**

对齐到 4.x 期望的形状：把 ``rope_parameters`` 的内容搬到 ``rope_scaling`` 名下，
让 ``rope_scaling`` 成为一个 dict。只在 4.x 上动，5.x 下是 no-op。

**校验义务**

``rope_scaling`` 的值决定 mRoPE 的 section 划分和 theta，读错会让位置编码整体
偏移 —— 不报错，但输出全错。所以这里只在 ``rope_scaling`` 缺失时才搬运，
且迁移后 ``scripts/verify_config_patch.py`` 会打印三个关键字段供人工核对。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Fields the 4.x rotary-embedding code reads out of ``rope_scaling``.
_ROPE_KEYS = ("rope_type", "mrope_section", "mrope_interleaved", "rope_theta")


def transformers_major() -> int:
    try:
        import transformers

        return int(str(transformers.__version__).split(".")[0])
    except Exception:  # pragma: no cover - diagnostic only
        return 0


def normalise_rope_config(config) -> bool:
    """Make ``config.rope_scaling`` a dict on transformers 4.x.

    Accepts an HF ``PretrainedConfig`` (text_config) or any object with
    ``rope_scaling`` / ``rope_parameters`` attributes. Returns True if changed.
    """
    if transformers_major() >= 5:
        return False  # 5.x reads rope_parameters natively

    scaling = getattr(config, "rope_scaling", None)
    if isinstance(scaling, dict) and scaling:
        return False  # already in the shape 4.x wants

    params = getattr(config, "rope_parameters", None)
    if not isinstance(params, dict) or not params:
        # Nothing to migrate; leave the original value (4.x tolerates None in most
        # paths, and inventing a dict here would change semantics).
        return False

    migrated = {k: params[k] for k in _ROPE_KEYS if k in params}
    # Carry over anything else the checkpoint stored, so nothing is silently lost.
    for k, v in params.items():
        migrated.setdefault(k, v)

    config.rope_scaling = migrated
    if not hasattr(config, "rope_parameters"):
        config.rope_parameters = migrated
    logger.info(
        "[lightnav] rope_scaling synthesised from rope_parameters: type=%s section=%s theta=%s",
        migrated.get("rope_type"),
        migrated.get("mrope_section"),
        migrated.get("rope_theta"),
    )
    return True


def patch_qwen3vl_rotary() -> bool:
    """Install the rope_scaling shim for every Qwen3VL rotary-embedding construction.

    Patches ``Qwen3VLTextRotaryEmbedding.__init__`` rather than
    ``AutoConfig.from_pretrained`` because vLLM builds its own config objects
    (``Qwen3VLForConditionalGeneration`` under vLLM) and does not always route
    through ``AutoConfig``. Patching the consumer covers all producers.

    Idempotent; no-op on transformers 5.x. Returns True if the patch is active.
    """
    if transformers_major() >= 5:
        return False

    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

    original = Qwen3VLTextRotaryEmbedding.__init__
    if getattr(original, "_lightnav_patched", False):
        return True

    def __init__(self, config=None, *args, **kwargs):
        if config is not None:
            normalise_rope_config(config)
        return original(self, config, *args, **kwargs)

    __init__._lightnav_patched = True
    Qwen3VLTextRotaryEmbedding.__init__ = __init__
    return True


def describe_rope_compat() -> str:
    try:
        import transformers

        if transformers_major() >= 5:
            return f"transformers {transformers.__version__} rope (5.x native: rope_parameters)"
        return (
            f"transformers {transformers.__version__} rope "
            "(4.x shim: rope_parameters -> rope_scaling)"
        )
    except Exception as exc:  # pragma: no cover
        return f"rope compat unavailable ({exc})"
