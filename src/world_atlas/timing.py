"""Persist stage durations and elapsed work across nested or parallel stages."""
from contextvars import ContextVar
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import uuid


_SCHEMA = "world-atlas-timing-v2"
_STAGES = ContextVar("world_atlas_timing_stages", default=())
_LOCKS = {}
_LOCKS_GUARD = threading.Lock()


@contextmanager
def _locked(path):
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(path, threading.RLock())
    with lock, path.with_suffix(".lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def _work_seconds(stages):
    """Union measured intervals; a parent and its children count only once."""
    intervals = sorted((stage["startedAtMonotonic"], stage["finishedAtMonotonic"])
                       for stage in stages if "finishedAtMonotonic" in stage)
    total, previous_end = 0., float("-inf")
    for begin, end in intervals:
        total += max(0., end - max(begin, previous_end))
        previous_end = max(previous_end, end)
    return round(total, 3)


def _update(path, change):
    with _locked(path):
        document = (json.loads(path.read_text(encoding="utf-8")) if path.exists()
                    else {"schema": _SCHEMA, "stages": []})
        if document.get("schema") != _SCHEMA:
            raise ValueError("obsolete timing document: use the matching engine to resume it")
        change(document["stages"])
        document["elapsedSeconds"] = _work_seconds(document["stages"])
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name,
                                             suffix=".tmp", delete=False,
                                             mode="w", encoding="utf-8") as stream:
                temporary = Path(stream.name)
                json.dump(document, stream, indent=2)
                stream.write("\n")
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


@contextmanager
def measure_stage(output: Path, name: str):
    """Record one stage without counting overlapping work more than once.

    Monotonic intervals share the system clock across threads and processes.
    Checkpoint resumes exclude time between stages; stages recorded on one
    machine during the same system boot may share this document.
    """
    path = output.resolve() / "timing.json"
    started = time.perf_counter()
    stack = _STAGES.get()
    parent = next((identifier for target, identifier in reversed(stack) if target == path), None)
    stage = {"id": uuid.uuid4().hex, "name": name, "parentId": parent,
             "status": "running", "startedAtUnix": time.time(),
             "startedAtMonotonic": started}
    _update(path, lambda stages: stages.append(stage))
    token = _STAGES.set((*stack, (path, stage["id"])))
    status = "failed"
    try:
        yield
        status = "complete"
    finally:
        finished = time.perf_counter()
        _STAGES.reset(token)
        def finish(stages):
            item = next(item for item in stages if item["id"] == stage["id"])
            item.update(status=status, finishedAtMonotonic=finished,
                        elapsedSeconds=round(finished-started, 3))
        _update(path, finish)


def elapsed_seconds(output: Path) -> float:
    return json.loads((output / "timing.json").read_text(encoding="utf-8"))["elapsedSeconds"]
