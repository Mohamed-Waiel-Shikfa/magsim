"""Background simulation jobs for the UI: one folder per job under res_sim/ui/.

Layout of a job folder `res_sim/ui/<YYYYmmdd-HHMMSS>_<sim>_<action>_<name>/`:
  scenario.yaml  the edited scenario (the simulator may overwrite it with its normalised copy)
  log.txt        stdout + stderr of the command
  job.json       {name, sim, action, scenario_name, cmd, cwd, pid, start, end, returncode, stopped}
  ...            everything the simulator writes (`--out <job dir>`)

The command runs in its own session (process group) so Stop also ends sweep workers. Status comes from
the returncode when known, else from the pid: `running` while the process lives, `lost` when the pid is
gone without a recorded returncode (the app was restarted while the job ran, so nobody collected it).
Processes started by this app process are kept in `_PROCS` and reaped by `poll`.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from magsim.ui import JOBS_DIR, ROOT

_PROCS: dict[str, subprocess.Popen] = {}  # job dir (str) -> Popen, for jobs started by this process

STATUSES = ("running", "done", "failed", "stopped", "lost")


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(s)).strip("-") or "scenario"


def _write_meta(job_dir: Path, meta: dict) -> None:
    tmp = job_dir / "job.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    os.replace(tmp, job_dir / "job.json")


def read_meta(job_dir: Path) -> dict:
    try:
        return json.loads((Path(job_dir) / "job.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def new_job_dir(sim: str, action: str, name: str, jobs_dir: Path = JOBS_DIR) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"{stamp}_{_safe(sim)}_{_safe(action)}_{_safe(name)}"
    d = Path(jobs_dir) / base
    k = 2
    while d.exists():
        d = Path(jobs_dir) / f"{base}-{k}"
        k += 1
    d.mkdir(parents=True)
    return d


def launch(job_dir: Path, cmd: list[str], meta: dict | None = None, env: dict | None = None,
           cwd: Path = ROOT) -> dict:
    """Start `cmd` in the background with stdout/stderr to job_dir/log.txt; write and return job.json."""
    job_dir = Path(job_dir)
    full_env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
    full_env.update(env or {})
    with open(job_dir / "log.txt", "wb") as log:
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, env=full_env, start_new_session=True)
    _PROCS[str(job_dir)] = proc
    m = dict(meta or {})
    m.update(name=job_dir.name, cmd=[str(c) for c in cmd], cwd=str(cwd), pid=proc.pid,
             start=time.time(), end=None, returncode=None, stopped=False)
    _write_meta(job_dir, m)
    return m


def start(sim: str, action: str, yaml_text: str, name: str, workers: int = 2,
          jobs_dir: Path = JOBS_DIR) -> Path:
    """Write the scenario into a new job folder and run `python -m magsim.cli <sim> <action>` on it."""
    if sim not in ("force", "switch") or action not in ("run", "sweep", "validate"):
        raise ValueError(f"unknown job {sim} {action}")
    d = new_job_dir(sim, action, name, jobs_dir)
    scen = d / "scenario.yaml"
    scen.write_text(yaml_text, encoding="utf-8")
    cmd = [sys.executable, "-m", "magsim.cli", sim, action, str(scen), "--out", str(d),
           "--workers", str(int(workers))]
    launch(d, cmd, meta=dict(sim=sim, action=action, scenario_name=name, workers=int(workers)))
    return d


def _pid_alive(pid: int | None, job_dir: Path) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    proc = Path(f"/proc/{int(pid)}")
    if proc.exists():  # Linux: reject zombies and recycled pids
        try:
            if (proc / "stat").read_text().split(")")[-1].split()[0] == "Z":
                return False
            cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            return str(job_dir.name) in cmdline
        except OSError:
            return False
    return True


def poll(job_dir: Path) -> dict:
    """Refresh job.json (reap a finished child) and return the metadata with a `status` key."""
    job_dir = Path(job_dir)
    m = read_meta(job_dir)
    if not m:
        return {"name": job_dir.name, "status": "lost"}
    proc = _PROCS.get(str(job_dir))
    if m.get("returncode") is None and proc is not None:
        rc = proc.poll()
        if rc is not None:
            m.update(returncode=rc, end=time.time())
            _write_meta(job_dir, m)
            _PROCS.pop(str(job_dir), None)
    rc = m.get("returncode")
    if rc is not None:
        status = "stopped" if m.get("stopped") else ("done" if rc == 0 else "failed")
    elif str(job_dir) in _PROCS or _pid_alive(m.get("pid"), job_dir):
        status = "running"
    else:
        status = "stopped" if m.get("stopped") else "lost"
    m["status"] = status
    return m


def stop(job_dir: Path, timeout: float = 5.0) -> dict:
    """SIGTERM the job's process group (SIGKILL after `timeout` s); returns the refreshed metadata."""
    job_dir = Path(job_dir)
    m = poll(job_dir)
    pid = m.get("pid")
    if m.get("status") == "running":
        m.pop("status")
        m["stopped"] = True
        _write_meta(job_dir, m)
        try:
            os.killpg(int(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        proc = _PROCS.get(str(job_dir))
        if proc is not None:
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(int(pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                proc.wait(timeout=timeout)
    return poll(job_dir)


def list_jobs(jobs_dir: Path = JOBS_DIR, sim: str | None = None) -> list[dict]:
    """All jobs (newest first), each with `status` and `dir`; optionally only one simulator."""
    jobs_dir = Path(jobs_dir)
    if not jobs_dir.is_dir():
        return []
    out = []
    for d in sorted((p for p in jobs_dir.iterdir() if (p / "job.json").is_file()), reverse=True):
        m = poll(d)
        if sim is None or m.get("sim") == sim:
            m["dir"] = d
            out.append(m)
    return out


def running(sim: str | None = None, action: str | None = None, jobs_dir: Path = JOBS_DIR) -> list[dict]:
    return [j for j in list_jobs(jobs_dir, sim) if j["status"] == "running"
            and (action is None or j.get("action") == action)]


def tail(job_dir: Path, n: int = 30) -> str:
    """Last n lines of log.txt (reads at most the last 256 kB)."""
    p = Path(job_dir) / "log.txt"
    try:
        with open(p, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 256_000))
            data = f.read().decode(errors="replace")
    except OSError:
        return ""
    data = data.replace("\r", "\n")
    return "\n".join(data.splitlines()[-n:])


def elapsed(m: dict) -> float | None:
    if not m.get("start"):
        return None
    return (m.get("end") or time.time()) - m["start"]
