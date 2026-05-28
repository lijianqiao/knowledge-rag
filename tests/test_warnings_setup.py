"""启动告警治理验证。"""

import os
import warnings


def test_hf_symlink_warning_env_set():
    import app  # noqa: F401  触发包初始化

    assert os.environ.get("HF_HUB_DISABLE_SYMLINKS_WARNING") == "1"


def test_pkg_resources_warning_filtered():
    import app

    # catch_warnings(record=True) 会重置过滤器，故在块内重新应用我方过滤器再触发
    with warnings.catch_warnings(record=True) as caught:
        warnings.resetwarnings()
        app._install_warning_filters()
        warnings.warn("pkg_resources is deprecated as an API.", UserWarning)
    assert not [w for w in caught if "pkg_resources is deprecated" in str(w.message)]
