import os

import gradio as gr


def apply_gradio_4_api_info_patch():
    """按需启用 Gradio 4 的 API schema 兼容补丁。

    训练页和推理页在当前项目里已经验证过：默认不打补丁也能正常打开。
    所以这个补丁现在退化为“显式兜底”：

    - 默认不启用
    - 只有设置 `GRADIO_ENABLE_API_INFO_PATCH=1` 时才会生效

    这样未来升级 Gradio 时，主路径会更接近官方默认行为；如果某次升级后
    又碰到首页因 `get_api_info()` 崩溃，再临时打开这个补丁即可。
    """
    blocks_cls = getattr(gr, "Blocks", None)
    if blocks_cls is None:
        return
    if os.environ.get("GRADIO_ENABLE_API_INFO_PATCH") != "1":
        return
    if getattr(blocks_cls, "_svc_api_info_patched", False):
        return

    original = blocks_cls.get_api_info
    blocks_cls._svc_original_get_api_info = original

    def safe_get_api_info(self, all_endpoints: bool = False):
        try:
            return original(self, all_endpoints)
        except Exception:
            return {"named_endpoints": {}, "unnamed_endpoints": {}}

    blocks_cls.get_api_info = safe_get_api_info
    blocks_cls._svc_api_info_patched = True
