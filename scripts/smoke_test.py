"""Minimal end-to-end check against the running stack.

Usage:
    python scripts/smoke_test.py path/to/sample.wav

Hits master's /process (full pipeline) plus each service's /health. Requires
the stack to be up (`docker compose up`).
"""

import sys

import requests

SERVICES = {
    "asr": "http://localhost:8001/health",
    "bert": "http://localhost:8002/health",
    "llm": "http://localhost:8003/health",
    "master": "http://localhost:8000/health",
}


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    audio_path = sys.argv[1]

    print("== health ==")
    for name, url in SERVICES.items():
        try:
            r = requests.get(url, timeout=10)
            print(f"  {name:6} {r.status_code} {r.json()}")
        except requests.RequestException as exc:
            print(f"  {name:6} UNREACHABLE: {exc}")
            return 1

    print("\n== full pipeline (/process) ==")
    with open(audio_path, "rb") as fh:
        files = {"audio": (audio_path, fh, "application/octet-stream")}
        r = requests.post("http://localhost:8000/process", files=files, timeout=180)
    r.raise_for_status()
    data = r.json()
    print(f"  transcript        : {data['transcript']!r}")
    print(f"  restored_text     : {data['restored_text']!r}")
    print(f"  bert_labels       : {data['bert_labels']}")
    print(f"  llm_classification: {data['llm_classification']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
