"""
@Author: li
@Email: lijianqiao2906@live.com
@FileName: __init__.py
@DateTime: 2026-05-28
@Docs: 运维知识库 RAG 应用
"""

import os
import warnings


def _install_warning_filters() -> None:
    """注册第三方库的弃用告警过滤器（库代码不可改，只能定向屏蔽）。"""
    # jieba 内部 import pkg_resources，触发 setuptools 弃用告警
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated.*", category=UserWarning)


_install_warning_filters()
# Windows 无符号链接权限时 HF 缓存的噪声告警
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
