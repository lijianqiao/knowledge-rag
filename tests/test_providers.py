"""provider 切换：按 CHAT_PROVIDER/EMBED_PROVIDER 选 base_url/model/key。"""

# 模型客户端构造逻辑已拆到 app.models；此测试验证构造内部（OpenAILike/_llm/provider），
# 须对 app.models 命名空间打桩（index 只是 re-export）。
import app.models as m


def test_llm_local(monkeypatch):
    cap = {}
    monkeypatch.setattr(m, "CHAT_PROVIDER", "local")
    monkeypatch.setattr(m, "OpenAILike", lambda **k: cap.update(k) or object())
    m._llm = None
    m.get_llm()
    assert cap["api_base"] == m.CHAT_BASE_URL
    assert cap["model"] == m.CHAT_MODEL


def test_llm_cloud(monkeypatch):
    cap = {}
    monkeypatch.setattr(m, "CHAT_PROVIDER", "cloud")
    monkeypatch.setattr(m, "CLOUD_CHAT_BASE_URL", "https://api.x.com/v1")
    monkeypatch.setattr(m, "CLOUD_CHAT_MODEL", "x-chat")
    monkeypatch.setattr(m, "CLOUD_CHAT_API_KEY", "sk-x")
    monkeypatch.setattr(m, "OpenAILike", lambda **k: cap.update(k) or object())
    m._llm = None
    m.get_llm()
    assert cap["api_base"] == "https://api.x.com/v1"
    assert cap["model"] == "x-chat"
    assert cap["api_key"] == "sk-x"
    m._llm = None  # 复位单例避免污染其他测试


def test_embed_cloud(monkeypatch):
    cap = {}
    monkeypatch.setattr(m, "EMBED_PROVIDER", "cloud")
    monkeypatch.setattr(m, "CLOUD_EMBED_BASE_URL", "https://api.x.com/v1")
    monkeypatch.setattr(m, "CLOUD_EMBED_MODEL", "x-embed")
    monkeypatch.setattr(m, "CLOUD_EMBED_API_KEY", "sk-e")
    monkeypatch.setattr(m, "OpenAIEmbedding", lambda **k: cap.update(k) or object())
    m._embed_model = None
    m.get_embed_model()
    assert cap["api_base"] == "https://api.x.com/v1"
    assert cap["model_name"] == "x-embed"
    m._embed_model = None
