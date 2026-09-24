"""Model weight source resolution: ModelScope first, HuggingFace fallback.

``source`` comes from run.yaml ``download.source``:

- ``auto``        (default) ModelScope -> HuggingFace -> engine's own downloader
- ``modelscope``  ModelScope only (fails loudly if unreachable)
- ``huggingface`` HuggingFace only

Heavy deps (``modelscope``, ``huggingface_hub``) are imported lazily so the
package installs and imports fine without either. ModelScope repo ids default
to the HuggingFace id (the depth-anything and facebook orgs use the same
names on both hubs — verified 2026-09).
"""

from __future__ import annotations

import os

SOURCE_AUTO = "auto"
SOURCE_MODELSCOPE = "modelscope"
SOURCE_HF = "huggingface"
VALID_SOURCES = (SOURCE_AUTO, SOURCE_MODELSCOPE, SOURCE_HF)


class HubError(RuntimeError):
    """Raised when a hard-requested source fails (source != auto)."""


def sanitize_ssl_env() -> None:
    """Defuse a stale ``SSL_CERT_FILE``.

    Some conda activation scripts point ``SSL_CERT_FILE`` at a CA bundle path
    that does not exist; httpx then raises FileNotFoundError on any request
    (observed on Windows: ``<env>/ssl/cacert.pem``). If the variable is set
    but missing, repoint it to certifi's bundle (or drop it entirely).
    """
    value = os.environ.get("SSL_CERT_FILE")
    if value and not os.path.isfile(value):
        try:
            import certifi

            os.environ["SSL_CERT_FILE"] = certifi.where()
        except ImportError:
            os.environ.pop("SSL_CERT_FILE", None)


sanitize_ssl_env()


# -- availability -----------------------------------------------------------
def has_modelscope() -> bool:
    try:
        import modelscope  # noqa: F401
        return True
    except ImportError:
        return False


def has_huggingface_hub() -> bool:
    try:
        import huggingface_hub  # noqa: F401
        return True
    except ImportError:
        return False


# -- thin download wrappers (isolated for monkeypatching) --------------------
def _ms_snapshot(repo_id: str) -> str:
    from modelscope import snapshot_download

    try:
        return snapshot_download(repo_id)
    except TypeError:  # older signature
        return snapshot_download(model_id=repo_id)


def _ms_file(repo_id: str, filename: str) -> str:
    from modelscope.hub.file_download import model_file_download

    try:
        return model_file_download(repo_id, filename)
    except TypeError:
        return model_file_download(model_id=repo_id, file_path=filename)


def _hf_snapshot(repo_id: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id)


def _hf_file(repo_id: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename)


def _order(source: str) -> list[str]:
    if source == SOURCE_MODELSCOPE:
        return [SOURCE_MODELSCOPE]
    if source == SOURCE_HF:
        return [SOURCE_HF]
    if source != SOURCE_AUTO:
        raise HubError(
            f"Unknown download source '{source}'. Valid: {', '.join(VALID_SOURCES)}"
        )
    return [SOURCE_MODELSCOPE, SOURCE_HF]


# -- public API --------------------------------------------------------------
def resolve_model_dir(hf_id: str, ms_id: str = "", source: str = SOURCE_AUTO) -> str:
    """Local directory usable by ``transformers.from_pretrained``.

    Tries sources in order; as the last resort returns ``hf_id`` itself so
    transformers performs its own (cached) HuggingFace download. With
    ``source="modelscope"`` failures raise instead of falling back.
    """
    ms_id = ms_id or hf_id
    errors: list[str] = []
    for src in _order(source):
        try:
            if src == SOURCE_MODELSCOPE:
                if not has_modelscope():
                    raise HubError("package 'modelscope' not installed (pip install modelscope)")
                return _ms_snapshot(ms_id)
            # HuggingFace
            if has_huggingface_hub():
                return _hf_snapshot(hf_id)
            return hf_id  # no huggingface_hub: transformers will try itself
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{src}: {exc}")
    if source == SOURCE_AUTO:
        return hf_id
    raise HubError(f"Model dir resolution failed -> {'; '.join(errors)}")


def resolve_model_dir_files(
    hf_id: str,
    ms_id: str = "",
    filenames: list[str] | None = None,
    source: str = SOURCE_AUTO,
) -> str | None:
    """Ensure a specific set of files exists locally and return the snapshot
    directory that contains them.

    Unlike ``resolve_model_dir`` (full snapshot download), this pulls only the
    named files — useful when the mirror hosts extra large files a loader does
    not need. Returns None when unreachable under ``source="auto"``; raises
    for hard-requested sources.
    """
    filenames = filenames or []
    if not filenames:
        raise HubError("resolve_model_dir_files needs at least one filename")
    ms_id = ms_id or hf_id
    first: str | None = None
    errors: list[str] = []
    for src in _order(source):
        try:
            if src == SOURCE_MODELSCOPE:
                if not has_modelscope():
                    raise HubError("package 'modelscope' not installed (pip install modelscope)")
                for name in filenames:
                    first = _ms_file(ms_id, name)
                return os.path.dirname(first) if first else None
            if has_huggingface_hub():
                for name in filenames:
                    first = _hf_file(hf_id, name)
                return os.path.dirname(first) if first else None
            return None
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{src}: {exc}")
            first = None
    if source == SOURCE_AUTO:
        return None
    raise HubError(f"File resolution failed -> {'; '.join(errors)}")
