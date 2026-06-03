from scripts import doctor


def test_faiss_numpy_smoke_check_runs_in_subprocess():
    ok, detail = doctor._check_faiss_numpy()
    assert ok, detail
    assert "faiss=" in detail
    assert "numpy=" in detail
    assert "smoke=ok" in detail
