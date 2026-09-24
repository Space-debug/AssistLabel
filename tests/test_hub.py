import pytest

from assistlabel import hub


def _raise(*a, **k):
    raise RuntimeError("should not be called")


# -- source ordering ---------------------------------------------------------
def test_auto_prefers_modelscope(monkeypatch):
    monkeypatch.setattr(hub, "has_modelscope", lambda: True)
    monkeypatch.setattr(hub, "_ms_snapshot", lambda rid: f"/ms/{rid}")
    assert hub.resolve_model_dir("hf/x", "", hub.SOURCE_AUTO) == "/ms/hf/x"


def test_auto_falls_back_to_hf_when_modelscope_fails(monkeypatch):
    monkeypatch.setattr(hub, "has_modelscope", lambda: True)

    def boom(rid):
        raise RuntimeError("network down")

    monkeypatch.setattr(hub, "_ms_snapshot", boom)
    monkeypatch.setattr(hub, "has_huggingface_hub", lambda: True)
    monkeypatch.setattr(hub, "_hf_snapshot", lambda rid: f"/hf/{rid}")
    assert hub.resolve_model_dir("hf/x", "", hub.SOURCE_AUTO) == "/hf/hf/x"


def test_auto_final_fallback_is_hf_id(monkeypatch):
    monkeypatch.setattr(hub, "has_modelscope", lambda: True)
    monkeypatch.setattr(hub, "_ms_snapshot", _raise)
    monkeypatch.setattr(hub, "has_huggingface_hub", lambda: False)
    # no huggingface_hub -> return hf_id, transformers does its own download
    assert hub.resolve_model_dir("hf/x", "", hub.SOURCE_AUTO) == "hf/x"


def test_modelscope_only_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(hub, "has_modelscope", lambda: False)
    with pytest.raises(hub.HubError):
        hub.resolve_model_dir("hf/x", "", hub.SOURCE_MODELSCOPE)


def test_hf_source_skips_modelscope(monkeypatch):
    monkeypatch.setattr(hub, "has_modelscope", lambda: True)
    monkeypatch.setattr(hub, "_ms_snapshot", _raise)
    monkeypatch.setattr(hub, "has_huggingface_hub", lambda: True)
    monkeypatch.setattr(hub, "_hf_snapshot", lambda rid: f"/hf/{rid}")
    assert hub.resolve_model_dir("hf/x", "ms/x", hub.SOURCE_HF) == "/hf/hf/x"


def test_invalid_source_rejected():
    with pytest.raises(hub.HubError):
        hub.resolve_model_dir("hf/x", "", "gitlab")


# -- multi-file resolution (sam3 style) -----------------------------------
def test_files_prefers_modelscope(monkeypatch):
    monkeypatch.setattr(hub, "has_modelscope", lambda: True)
    paths = iter([f"/ms/cache/{f}" for f in ("config.json", "model.safetensors")])
    monkeypatch.setattr(hub, "_ms_file", lambda rid, fn: next(paths))
    d = hub.resolve_model_dir_files("facebook/sam3", "", ["config.json", "model.safetensors"])
    assert d == "/ms/cache"


def test_files_auto_returns_none_when_all_fail(monkeypatch):
    monkeypatch.setattr(hub, "has_modelscope", lambda: False)
    monkeypatch.setattr(hub, "has_huggingface_hub", lambda: False)
    assert hub.resolve_model_dir_files("facebook/sam3", "", ["model.safetensors"]) is None


def test_files_modelscope_only_raises(monkeypatch):
    monkeypatch.setattr(hub, "has_modelscope", lambda: False)
    with pytest.raises(hub.HubError):
        hub.resolve_model_dir_files("facebook/sam3", "", ["model.safetensors"],
                                    source=hub.SOURCE_MODELSCOPE)


def test_files_empty_list_rejected():
    with pytest.raises(hub.HubError):
        hub.resolve_model_dir_files("hf/sam3", "", [])


def test_ms_id_overrides_hf_id(monkeypatch):
    seen = {}

    def fake_ms_file(rid, fn):
        seen["repo"] = rid
        return f"/ms/{rid}/{fn}"

    monkeypatch.setattr(hub, "has_modelscope", lambda: True)
    monkeypatch.setattr(hub, "_ms_file", fake_ms_file)
    hub.resolve_model_dir_files("hf/sam3", "mirror/sam3", ["model.safetensors"])
    assert seen["repo"] == "mirror/sam3"
