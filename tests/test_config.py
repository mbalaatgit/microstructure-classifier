import importlib
from pathlib import Path


def test_config_uses_environment_overrides(monkeypatch, tmp_path):
    data_dir = tmp_path / "qa-data"
    monkeypatch.setenv("MICROSTRUCTURE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("MICROSTRUCTURE_EMBEDDING_MODEL", "resnet50")

    import config

    reloaded = importlib.reload(config)
    assert reloaded.DATA_DIR == data_dir
    assert reloaded.RAW_DIR == data_dir / "raw"
    assert reloaded.EMBEDDINGS_DIR == data_dir / "embeddings"
    assert reloaded.FAISS_INDEX_PATH == data_dir / "embeddings" / "microstructure.index"
    assert reloaded.METADATA_PATH == data_dir / "embeddings" / "metadata.pkl"
    assert reloaded.INDEX_MANIFEST_PATH == data_dir / "embeddings" / "index_manifest.json"
    assert reloaded.EMBEDDING_MODEL == "resnet50"
    assert reloaded.RAW_DIR.exists()
    assert reloaded.EMBEDDINGS_DIR.exists()

    monkeypatch.delenv("MICROSTRUCTURE_DATA_DIR")
    monkeypatch.delenv("MICROSTRUCTURE_EMBEDDING_MODEL")
    importlib.reload(config)
