"""transformers 4.x / 5.x 的 tokenizer 配置兼容层。

**为什么需要这个模块**

项目默认面向 transformers 5.8，但 Jetson Orin 上唯一带原生 sm_87 kernel 的
vLLM 轮子（``vllm-0.19.0+cu126-cp310``）把 transformers 钉在 ``<5``。两种版本
都必须能加载同一个 checkpoint。

transformers 5.0 做了一次破坏性改名：

    4.x :  additional_special_tokens   （list[str]）
    5.x :  extra_special_tokens        （list[str]）

附带一个新行为：5.x 的 ``_set_model_specific_special_tokens`` 会把
``extra_special_tokens`` 并入 ``SPECIAL_TOKENS_ATTRIBUTES``；4.57 没有这个分支，
于是把 config 里的同名 kwarg 当普通参数透传，最后对 list 调用 ``.keys()``：

    AttributeError: 'list' object has no attribute 'keys'

**做法**

不修改 checkpoint 文件，也不依赖 vLLM 是否透传 ``tokenizer_config`` 参数：
在进程内拦截 ``AutoTokenizer.from_pretrained``，把 ``extra_special_tokens`` 的
list 值改挂到 ``additional_special_tokens`` 名下。语义等价 —— 两者都是"额外的、
不参与切分、必须作为单个 token 存在的 special token 列表"。

**校验义务**

这改变了 tokenizer 的构造路径，因此 **必须** 用
``scripts/verify_tokenizer_parity.py`` 对比 4.57 与 5.8 下：

  * 词表大小、每个 special token 的 id
  * 同一段真实 prompt 的 ``input_ids`` 是否逐字节相同

不一致 ⇒ 4.57 路线不能用于生产，应改用 5.8 + 其它推理路径。
"""

from __future__ import annotations

import json
from pathlib import Path

_PATCHED = False


def transformers_major() -> int:
    """Major version of the installed transformers (4 or 5); 0 if unknown."""
    try:
        import transformers

        return int(str(transformers.__version__).split(".")[0])
    except Exception:  # pragma: no cover - diagnostic only
        return 0


def _rehome_extra_special_tokens(kwargs: dict) -> bool:
    """Normalise ``extra_special_tokens`` to the shape this transformers expects.

    Both 4.57 and 5.8 read a field named ``extra_special_tokens``, but they disagree
    on its type:

        4.57 (tokenization_utils_base.py:1470)
            self.extra_special_tokens = kwargs.pop("extra_special_tokens", {})
            self._set_model_specific_special_tokens(self.extra_special_tokens)
            # ... and the setter does `list(special_tokens.keys())` -> needs a dict

        5.8  accepts a list/tuple of tokens as well (and stores it as
            `_extra_special_tokens`, flattening the dict form).

    LightNav-0's ``tokenizer_config.json`` carries the 5.x form: a flat list of ~3371
    token strings. Handing that list to 4.57 reaches ``list(list.keys())`` and dies.

    So on 4.x we wrap the list into a dict. Key names only matter for 4.57's
    ``tokenizer_config.json`` round-trip (``tokenizer_config.update(...)``), not for
    tokenisation: ``_set_model_specific_special_tokens`` only uses ``.keys()`` to
    learn which attribute names to append to ``SPECIAL_TOKENS_ATTRIBUTES`` and then
    looks each up via ``getattr(self, name)``, which is ``None`` for invented names
    and is skipped. Ids of already-known tokens are therefore unaffected; the
    side effect is that the ~3371 special tokens are not re-registered through this
    path. ``scripts/verify_tokenizer_parity.py`` is what proves that is harmless.

    Returns True when something was changed.
    """
    extra = kwargs.get("extra_special_tokens")
    if extra is None:
        return False

    if transformers_major() >= 5:
        return False  # list is the expected form there

    if isinstance(extra, dict):
        return False  # already the shape 4.x wants

    if not isinstance(extra, (list, tuple)):
        return False

    tokens = [v for v in extra if isinstance(v, str)]
    # Deterministic, collision-free attribute names. The names are arbitrary; see
    # the docstring -- only the dict-ness matters to 4.57.
    kwargs["extra_special_tokens"] = {f"lightnav_extra_{i}": t for i, t in enumerate(tokens)}
    return True


def patch_autotokenizer() -> bool:
    """Install the transformers-4 special-token shim.

    No-op on transformers 5.x (the list form is native there) and idempotent.
    Returns True if the patch is active.

    Where it hooks, and why not ``AutoTokenizer.from_pretrained``: 4.57 builds
    ``init_kwargs`` by *reading* ``tokenizer_config.json`` inside ``_from_pretrained``,
    so rewriting ``kwargs`` on the way in misses the field entirely. The crash happens
    one level deeper, in ``_set_model_specific_special_tokens``, which assumes a dict.
    Patching that method covers every load path (AutoTokenizer, cls(...) direct,
    vLLM's cached loader) at once.
    """
    global _PATCHED
    if _PATCHED:
        return True
    if transformers_major() >= 5:
        return False

    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    original = PreTrainedTokenizerBase._set_model_specific_special_tokens
    if getattr(original, "_lightnav_patched", False):
        _PATCHED = True
        return True

    def _set_model_specific_special_tokens(self, special_tokens):
        # transformers 4.57 assumes a dict (it calls .keys() then .items()); the
        # checkpoint stores the 5.x form: a flat list of token strings.
        # A bare list carries no attribute names, so there is nothing to register in
        # SPECIAL_TOKENS_ATTRIBUTES. The tokens themselves are already in the
        # vocabulary via tokenizer.json's added_tokens, so their ids are unaffected --
        # which is exactly what scripts/verify_tokenizer_parity.py verifies.
        if isinstance(special_tokens, (list, tuple)):
            special_tokens = {f"lightnav_extra_{i}": t for i, t in enumerate(special_tokens)}
        return original(self, special_tokens)

    _set_model_specific_special_tokens._lightnav_patched = True
    PreTrainedTokenizerBase._set_model_specific_special_tokens = _set_model_specific_special_tokens
    _PATCHED = True
    return True


def describe_tokenizer_compat() -> str:
    """One-line summary for startup logs."""
    try:
        import transformers

        return (
            f"transformers {transformers.__version__} tokenizer "
            f"({'4.x -> additional_special_tokens shim' if transformers_major() < 5 else '5.x native'})"
        )
    except Exception as exc:  # pragma: no cover
        return f"tokenizer compat unavailable ({exc})"


def load_tokenizer_for_parity(model_path: str | Path):
    """Load the tokenizer the same way vLLM will, for parity checks."""
    patch_autotokenizer()
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)


def _selftest() -> int:  # pragma: no cover - manual
    cfg = json.loads((Path("checkpoints/LightNav-0") / "tokenizer_config.json").read_text())
    kw = {"extra_special_tokens": cfg.get("extra_special_tokens")}
    before = len(kw.get("extra_special_tokens") or [])
    _rehome_extra_special_tokens(kw)
    assert "extra_special_tokens" not in kw
    after = len(kw.get("additional_special_tokens") or [])
    print(f"extra={before} -> additional={after}  match={before == after}")
    return 0 if before == after else 1


if __name__ == "__main__":  # pragma: no cover - manual
    raise SystemExit(_selftest())
