import numpy as np

from src.index import MicrostructureIndex


def test_cosine_index_round_trip(tmp_path):
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    metadata = [
        {"path": "ferrite.png", "label": "ferrite", "source": "qa"},
        {"path": "martensite.png", "label": "martensite", "source": "qa"},
    ]

    index = MicrostructureIndex(embedding_dim=3, metric="cosine")
    index.add(embeddings, metadata)

    results = index.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=2)
    assert results[0]["label"] == "ferrite"
    assert results[0]["distance"] > results[1]["distance"]

    index_path = tmp_path / "microstructure.index"
    metadata_path = tmp_path / "metadata.pkl"
    index.save(index_path=index_path, metadata_path=metadata_path)

    loaded = MicrostructureIndex.load(index_path=index_path, metadata_path=metadata_path)
    loaded_results = loaded.search(np.array([0.0, 1.0, 0.0], dtype=np.float32), top_k=1)
    assert loaded.size == 2
    assert loaded_results[0]["label"] == "martensite"
