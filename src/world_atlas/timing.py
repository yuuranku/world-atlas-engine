"""Persist completed and failed stage durations across checkpoint resumes."""
from contextlib import contextmanager
import json
from pathlib import Path
import time


@contextmanager
def measure_stage(output: Path, name: str):
    path = output / "timing.json"
    document = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"stages": []}
    stage = {"name": name, "status": "running", "startedAtUnix": time.time()}
    document["stages"].append(stage)
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    started = time.perf_counter()
    try:
        yield
    except BaseException:
        stage["status"] = "failed"
        raise
    else:
        stage["status"] = "complete"
    finally:
        stage["elapsedSeconds"] = round(time.perf_counter() - started, 3)
        document["elapsedSeconds"] = round(sum(item.get("elapsedSeconds", 0.) for item in document["stages"]), 3)
        path.write_text(json.dumps(document, indent=2), encoding="utf-8")


def elapsed_seconds(output: Path) -> float:
    return json.loads((output / "timing.json").read_text(encoding="utf-8"))["elapsedSeconds"]
