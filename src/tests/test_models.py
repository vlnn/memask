import pytest

from memask.rag.models import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_URL,
    models_dir,
    model_path,
    model_status,
)


class TestModelsDir:
    def test_default_location(self):
        path = models_dir()
        assert str(path).endswith(".memask/models"), (
            "default models dir should be under ~/.memask/models"
        )

    def test_custom_base(self, tmp_path):
        path = models_dir(base=tmp_path)
        assert path == tmp_path / "models", (
            "should put models dir under provided base"
        )


class TestModelPath:
    def test_returns_path_under_models_dir(self, tmp_path):
        path = model_path(base=tmp_path)
        assert str(path).startswith(str(tmp_path)), (
            "model path should be under base dir"
        )
        assert path.name == DEFAULT_MODEL_NAME, (
            "should use default model filename"
        )

    def test_custom_model_name(self, tmp_path):
        path = model_path(model_name="custom.gguf", base=tmp_path)
        assert path.name == "custom.gguf", "should use custom model name"


class TestModelStatus:
    def test_not_downloaded(self, tmp_path):
        status = model_status(base=tmp_path)
        assert status["downloaded"] is False, (
            "should report not downloaded when file missing"
        )
        assert status["path"] is not None, "should always include path"

    def test_downloaded(self, tmp_path):
        mdir = tmp_path / "models"
        mdir.mkdir()
        mpath = mdir / DEFAULT_MODEL_NAME
        mpath.write_bytes(b"x" * 200_000)
        status = model_status(base=tmp_path)
        assert status["downloaded"] is True, (
            "should report downloaded when file exists"
        )
        assert status["size_mb"] > 0, "should report file size"

    def test_includes_model_name(self, tmp_path):
        status = model_status(base=tmp_path)
        assert status["model_name"] == DEFAULT_MODEL_NAME, (
            "should include model name"
        )

    def test_includes_url(self, tmp_path):
        status = model_status(base=tmp_path)
        assert status["url"] == DEFAULT_MODEL_URL, "should include download URL"


class TestDefaults:
    def test_default_model_name_is_gguf(self):
        assert DEFAULT_MODEL_NAME.endswith(".gguf"), (
            "default model should be a GGUF file"
        )

    def test_default_url_is_https(self):
        assert DEFAULT_MODEL_URL.startswith("https://"), (
            "download URL should be HTTPS"
        )
