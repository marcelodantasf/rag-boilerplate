from __future__ import annotations

import pytest

from embedding_api.infrastructure.settings import PINNED_MODEL_REVISION, Settings


ENVIRONMENT_KEYS = [
    "DEFAULT_MODEL_ID",
    "DEFAULT_EMBEDDING_MODEL",
    "MODEL_SOURCE",
    "MODEL_REVISION",
    "EXPECTED_DIMENSION",
    "INFERENCE_DEVICE",
    "MODEL_CACHE_DIR",
    "NORMALIZE_EMBEDDINGS",
    "ENGINE_BATCH_SIZE",
    "MAX_BATCH_ITEMS",
    "MAX_INPUT_BYTES",
    "MAX_TOTAL_INPUT_BYTES",
    "MAX_BATCH_TOKENS",
    "LOG_LEVEL",
]


@pytest.fixture(autouse=True)
def clean_settings_environment(monkeypatch) -> None:
    for key in ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_default_settings_pin_model_and_expected_capability() -> None:
    settings = Settings.from_env()

    assert settings.model_id == "all-MiniLM-L6-v2"
    assert settings.model_source == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.model_revision == PINNED_MODEL_REVISION
    assert settings.expected_dimension == 384
    assert settings.normalize_embeddings is True
    assert settings.max_batch_items > 0
    assert settings.max_total_input_bytes >= settings.max_input_bytes


def test_all_supported_environment_values_are_loaded(monkeypatch, tmp_path) -> None:
    values = {
        "DEFAULT_MODEL_ID": "public-id",
        "DEFAULT_EMBEDDING_MODEL": "ignored-alias",
        "MODEL_SOURCE": "org/model",
        "MODEL_REVISION": "immutable-revision",
        "EXPECTED_DIMENSION": "17",
        "INFERENCE_DEVICE": "mps",
        "MODEL_CACHE_DIR": str(tmp_path),
        "NORMALIZE_EMBEDDINGS": "FALSE",
        "ENGINE_BATCH_SIZE": "2",
        "MAX_BATCH_ITEMS": "3",
        "MAX_INPUT_BYTES": "10",
        "MAX_TOTAL_INPUT_BYTES": "20",
        "MAX_BATCH_TOKENS": "30",
        "LOG_LEVEL": "warning",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)

    settings = Settings.from_env()

    assert settings == Settings(
        model_id="public-id",
        model_source="org/model",
        model_revision="immutable-revision",
        expected_dimension=17,
        inference_device="mps",
        model_cache_dir=str(tmp_path),
        normalize_embeddings=False,
        engine_batch_size=2,
        max_batch_items=3,
        max_input_bytes=10,
        max_total_input_bytes=20,
        max_batch_tokens=30,
        log_level="WARNING",
    )


def test_legacy_model_id_alias_is_supported(monkeypatch) -> None:
    monkeypatch.setenv("DEFAULT_EMBEDDING_MODEL", "legacy-id")

    assert Settings.from_env().model_id == "legacy-id"


@pytest.mark.parametrize(
    "name,value",
    [
        ("EXPECTED_DIMENSION", "0"),
        ("ENGINE_BATCH_SIZE", "-1"),
        ("MAX_BATCH_ITEMS", "0"),
        ("MAX_INPUT_BYTES", "nope"),
        ("MAX_TOTAL_INPUT_BYTES", "0"),
        ("MAX_BATCH_TOKENS", "-2"),
    ],
)
def test_positive_integer_settings_fail_fast(monkeypatch, name: str, value: str) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=name):
        Settings.from_env()


def test_normalization_setting_is_strict(monkeypatch) -> None:
    monkeypatch.setenv("NORMALIZE_EMBEDDINGS", "yes")

    with pytest.raises(ValueError, match="NORMALIZE_EMBEDDINGS"):
        Settings.from_env()


@pytest.mark.parametrize("name", ["DEFAULT_MODEL_ID", "MODEL_SOURCE", "MODEL_REVISION"])
def test_blank_model_identity_values_fail_fast(monkeypatch, name: str) -> None:
    monkeypatch.setenv(name, "   ")

    with pytest.raises(ValueError):
        Settings.from_env()


def test_total_byte_limit_cannot_be_below_item_limit(monkeypatch) -> None:
    monkeypatch.setenv("MAX_INPUT_BYTES", "11")
    monkeypatch.setenv("MAX_TOTAL_INPUT_BYTES", "10")

    with pytest.raises(ValueError, match="MAX_TOTAL_INPUT_BYTES"):
        Settings.from_env()


def test_invalid_log_level_fails_fast(monkeypatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "verbose")

    with pytest.raises(ValueError, match="LOG_LEVEL"):
        Settings.from_env()
