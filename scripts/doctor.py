#!/usr/bin/env python3
"""Preflight checks for local deployment and index readiness."""
import json
import platform
import subprocess
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.utils import collect_image_paths


def _status(flag: bool) -> str:
    return "OK" if flag else "MISSING"


def _run_python_check(code: str, timeout: int = 20) -> tuple[bool, str]:
    """Run a dependency check in a subprocess so SIGILL cannot kill doctor.py."""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        timeout=timeout,
    )

    output = (proc.stdout + proc.stderr).strip()
    if proc.returncode == 0:
        return True, output or "ok"

    if proc.returncode < 0:
        signal_number = -proc.returncode
        signal_name = "SIGILL / illegal instruction" if signal_number == 4 else f"signal {signal_number}"
        return False, f"terminated by {signal_name}. {output}".strip()

    return False, output or f"exit code {proc.returncode}"


def _check_dependency(import_name: str, version_attr: str = "__version__") -> tuple[bool, str]:
    code = f"""
import importlib
module = importlib.import_module({import_name!r})
print(getattr(module, {version_attr!r}, 'version unavailable'))
"""
    return _run_python_check(code)


def _check_faiss_numpy() -> tuple[bool, str]:
    code = """
import faiss
import numpy as np
idx = faiss.IndexFlatIP(3)
vectors = np.eye(3, dtype='float32')
idx.add(vectors)
distances, indices = idx.search(vectors[:1], 1)
assert indices[0, 0] == 0
assert distances[0, 0] > 0.99
print(f"faiss={getattr(faiss, '__version__', 'unknown')} numpy={np.__version__} smoke=ok")
"""
    return _run_python_check(code)


def _cpu_feature_summary() -> str:
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        return f"{platform.machine()} (non-x86; AVX flags not applicable)"

    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return f"{platform.machine()} (CPU flags unavailable)"

    flags_line = ""
    for line in cpuinfo.read_text(errors="ignore").splitlines():
        if line.startswith("flags"):
            flags_line = line
            break

    flags = set(flags_line.split())
    supported = [flag for flag in ["sse4_2", "avx", "avx2", "avx512f"] if flag in flags]
    return f"{platform.machine()} ({', '.join(supported) if supported else 'no AVX flags found'})"


def main() -> int:
    print("μStruct deployment doctor")
    print("=" * 72)

    print("\nConfiguration")
    print(f"  project root:       {config.PROJECT_ROOT}")
    print(f"  raw data dir:       {config.RAW_DIR}")
    print(f"  embeddings dir:     {config.EMBEDDINGS_DIR}")
    print(f"  index path:         {config.FAISS_INDEX_PATH}")
    print(f"  metadata path:      {config.METADATA_PATH}")
    print(f"  manifest path:      {config.INDEX_MANIFEST_PATH}")
    print(f"  server model:       {config.EMBEDDING_MODEL}")
    print(f"  similarity metric:  {config.SIMILARITY_METRIC}")
    print(f"  CPU features:       {_cpu_feature_summary()}")

    print("\nPython dependencies")
    dependency_checks = [
        ("torch", "torch", lambda: _check_dependency("torch")),
        ("torchvision", "torchvision", lambda: _check_dependency("torchvision")),
        ("open-clip-torch", "open_clip", lambda: _check_dependency("open_clip")),
        ("faiss-cpu + numpy", "faiss/numpy", _check_faiss_numpy),
        ("fastapi", "fastapi", lambda: _check_dependency("fastapi")),
        ("Pillow", "PIL", lambda: _check_dependency("PIL", "__version__")),
        ("numpy", "numpy", lambda: _check_dependency("numpy")),
    ]
    dependency_failures = 0
    for package_name, _import_name, checker in dependency_checks:
        ok, detail = checker()
        dependency_failures += 0 if ok else 1
        print(f"  {_status(ok):8} {package_name:18} {detail}")

    print("\nData and index")
    raw_images = collect_image_paths(config.RAW_DIR) if config.RAW_DIR.exists() else []
    has_index = config.FAISS_INDEX_PATH.exists()
    has_metadata = config.METADATA_PATH.exists()
    has_manifest = config.INDEX_MANIFEST_PATH.exists()
    print(f"  {_status(config.RAW_DIR.exists()):8} raw data directory ({len(raw_images)} images found)")
    print(f"  {_status(has_index):8} FAISS index")
    print(f"  {_status(has_metadata):8} metadata pickle")
    print(f"  {_status(has_manifest):8} index manifest")

    manifest_model = None
    if has_manifest:
        try:
            manifest = json.loads(config.INDEX_MANIFEST_PATH.read_text())
            manifest_model = manifest.get("model")
            print("\nIndex manifest")
            print(f"  created at:         {manifest.get('created_at', 'unknown')}")
            print(f"  indexed model:      {manifest_model}")
            print(f"  vectors:            {manifest.get('count', 'unknown')}")
            print(f"  embedding dim:      {manifest.get('embedding_dim', 'unknown')}")
            print(f"  sources:            {manifest.get('sources', {})}")
        except Exception as exc:
            print(f"  Manifest could not be read: {type(exc).__name__}: {exc}")

    if manifest_model and manifest_model != config.EMBEDDING_MODEL:
        print(
            "\nWARNING: index was built with model "
            f"'{manifest_model}', but the server is configured for "
            f"'{config.EMBEDDING_MODEL}'. Set MICROSTRUCTURE_EMBEDDING_MODEL={manifest_model} "
            "or rebuild the index."
        )

    print("\nNext commands")
    if not raw_images:
        print("  1. Put micrograph images under data/raw/<dataset-or-class-name>/")
        print("     or use: python scripts/build_index.py --data-dir /path/to/images --dry-run")
    if not (has_index and has_metadata):
        print("  2. Build embeddings/index:")
        print("     python scripts/build_index.py --model clip --include-excluded")
    print("  3. Start app:")
    print("     uvicorn app.server:app --host 0.0.0.0 --port 8000")
    print("  4. Check health:")
    print("     curl http://localhost:8000/api/health")

    if dependency_failures:
        return 2
    if not (has_index and has_metadata):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
