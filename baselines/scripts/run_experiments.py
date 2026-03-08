#!/usr/bin/env python3
"""Experiment runner for MPS sweep across schedulers.

Generates unique (scheduler, R0%, R1%, R2%, R3%, R4%, R5%)
experiment tuples, modifies the YAML config, invokes
deploy_asteroid.sh, and collects JSONL files from nodes.

Usage
-----
    python3 baselines/scripts/run_experiments.py \
        --config baselines/configs/asteroid_default.yaml \
        --num-experiments 20 \
        --db experiments.db

Env vars
--------
    ASTEROID_METRICS=1     — enable JSONL logging in train.py
    ASTEROID_METRICS_DIR   — override /tmp/asteroid_metrics
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
import random
import shutil
import sqlite3
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# ── Constants ────────────────────────────────────────────

MPS_CHOICES = [20, 40]
SCHEDULERS = ["confident", "dtfm", "asteroid"]
GPU_TOTAL_MB = 45459   # L40S usable memory
NUM_RANKS = 8
METRICS_REMOTE_DIR = "/tmp/asteroid_metrics"

# Subnet pairs: nodes within a pair share fast links and
# are always grouped together by the planner.  Assigning
# the same MPS% to both nodes in a pair cuts the search
# space from 4^8=65536 to 4^4=256 per scheduler (768 total).
SUBNET_PAIRS = [
    (0, 1),   # subnet 54  (R0↔R1)
    (2, 5),   # subnet 53  (R2↔R5)
    (3, 4),   # subnet 56  (R3↔R4)
    (6, 7),   # subnet 57  (R6↔R7)
]

BASELINES_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASELINES_DIR / "scripts"
DEPLOY_SCRIPT = SCRIPTS_DIR / "deploy_asteroid.sh"


# ── Timestamp helper ─────────────────────────────────────


def _ts() -> str:
    """Return a compact timestamp for log lines."""
    return datetime.now().strftime("%H:%M:%S")


def _fmt_duration(seconds: float) -> str:
    """Format seconds as  Xh Ym Zs  or  Ym Zs  etc."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


# ── SSH helpers ──────────────────────────────────────────

_SSH_PASS: str | None = None


def _get_ssh_pass() -> str:
    """Extract SSH password from ansible-vault secrets.

    Mirrors the sshpass approach in deploy_asteroid.sh so
    that scp/ssh calls don't hang waiting for a password.
    """
    global _SSH_PASS
    if _SSH_PASS is not None:
        return _SSH_PASS

    secrets = BASELINES_DIR / "deploy_asteroid" / "secrets.yml"
    vault_pw = Path.home() / ".baselines_vault_pass"
    venv_py = BASELINES_DIR / ".venv" / "bin" / "python"
    venv_vault = BASELINES_DIR / ".venv" / "bin" / "ansible-vault"

    if secrets.exists() and vault_pw.exists() and venv_vault.exists():
        try:
            raw = subprocess.check_output(
                [str(venv_vault), "view", str(secrets),
                 "--vault-password-file", str(vault_pw)],
                stderr=subprocess.DEVNULL,
                timeout=10,
            ).decode()
            parsed = yaml.safe_load(raw)
            _SSH_PASS = parsed.get(
                "ansible_ssh_pass",
                parsed.get("ansible_password", ""),
            )
        except Exception:
            _SSH_PASS = ""
    else:
        _SSH_PASS = ""
    return _SSH_PASS


def _ssh_cmd(ip: str, remote_cmd: str,
             timeout: int = 15) -> subprocess.CompletedProcess:
    """Run a command on a remote node via ssh/sshpass."""
    pw = _get_ssh_pass()
    base = [
        "ssh", "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        "-o", "BatchMode=yes" if not pw else "BatchMode=no",
        f"ubuntu@{ip}", remote_cmd,
    ]
    if pw:
        base = ["sshpass", "-p", pw] + base
    return subprocess.run(
        base, capture_output=True, timeout=timeout,
    )


def _scp_cmd(ip: str, remote_path: str,
             local_path: str,
             timeout: int = 30) -> subprocess.CompletedProcess:
    """SCP a file from a remote node."""
    pw = _get_ssh_pass()
    base = [
        "scp", "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        "-o", "BatchMode=yes" if not pw else "BatchMode=no",
        f"ubuntu@{ip}:{remote_path}",
        local_path,
    ]
    if pw:
        base = ["sshpass", "-p", pw] + base
    return subprocess.run(
        base, capture_output=True, timeout=timeout,
        check=True,
    )


# ── YAML helpers ─────────────────────────────────────────


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.dump(
            data, f,
            default_flow_style=False,
            sort_keys=False,
            width=120,
        )


# ── Experiment ID ────────────────────────────────────────


def make_experiment_id(
    scheduler: str,
    mps_pcts: list[int],
) -> str:
    """Deterministic experiment ID from config tuple."""
    key = f"{scheduler}:" + ",".join(
        str(p) for p in mps_pcts
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# ── Sweep generation ────────────────────────────────────


def generate_sweep(
    num_experiments: int | None = None,
    seed: int = 42,
) -> list[tuple[str, list[int]]]:
    """Generate (scheduler, [R0%, ... R7%]) tuples.

    All nodes are treated as identical hardware, so only
    the *multiset* of MPS percentages matters — not which
    node gets which value.  This reduces the space from
    3×4^8 = 196,608 to 3×C(11,8) = 3×165 = 495 combos.
    If num_experiments is given, samples randomly.
    """
    all_combos: list[tuple[str, list[int]]] = []
    for sched in SCHEDULERS:
        for combo in itertools.combinations_with_replacement(
            MPS_CHOICES, NUM_RANKS
        ):
            # combo is sorted (e.g. (20,20,40,60,...))
            # — assign to ranks in order; the planner
            # handles topology-aware placement.
            all_combos.append((sched, list(combo)))

    if num_experiments is None or num_experiments >= len(
        all_combos
    ):
        return all_combos

    rng = random.Random(seed)
    return rng.sample(all_combos, num_experiments)


# ── Config mutation ──────────────────────────────────────


def apply_experiment_config(
    cfg: dict,
    scheduler: str,
    mps_pcts: list[int],
) -> dict:
    """Mutate config dict for this experiment."""
    cfg = copy.deepcopy(cfg)

    # Set scheduler / strategy
    cfg["parallelism"]["strategy"] = scheduler

    # Set per-node MPS
    nodes = cfg["cluster"]["nodes"]
    for i, node in enumerate(nodes):
        if i >= NUM_RANKS:
            break
        pct = mps_pcts[i]
        mem_mb = int(math.floor(GPU_TOTAL_MB * pct / 100))
        if "mps" not in node:
            node["mps"] = {}
        node["mps"]["active_thread_percentage"] = pct
        node["mps"]["memory_limit_mb"] = mem_mb

    return cfg


# ── SQLite setup ─────────────────────────────────────────


def init_db(db_path: str) -> sqlite3.Connection:
    """Create tables if not exists, return connection."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    schema_sql = (
        BASELINES_DIR / "docs" / "PRD_experiment_schema.md"
    )
    # We create tables inline rather than parsing markdown
    conn.executescript(_SCHEMA_SQL)
    conn.commit()
    return conn


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id       TEXT PRIMARY KEY,
    scheduler           TEXT NOT NULL
                        CHECK (scheduler IN ('dtfm', 'confident', 'asteroid')),
    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status              TEXT NOT NULL DEFAULT 'running'
                        CHECK (status IN ('running', 'completed', 'failed')),
    failure_reason      TEXT,
    git_commit          TEXT,
    docker_image        TEXT,
    model_name          TEXT NOT NULL,
    model_type          TEXT NOT NULL,
    num_layers          INTEGER NOT NULL,
    embedding_dim       INTEGER NOT NULL,
    num_heads           INTEGER NOT NULL,
    d_ff                INTEGER NOT NULL,
    vocab_size          INTEGER NOT NULL,
    max_seq_len         INTEGER NOT NULL,
    global_batch_size   INTEGER NOT NULL,
    micro_batch_size    INTEGER NOT NULL,
    num_microbatches    INTEGER NOT NULL,
    lr                  REAL NOT NULL,
    min_lr              REAL NOT NULL,
    weight_decay        REAL NOT NULL,
    max_iters           INTEGER NOT NULL,
    warmup_iters        INTEGER NOT NULL,
    grad_clip           REAL NOT NULL,
    dataset             TEXT NOT NULL,
    synthetic_samples   INTEGER NOT NULL DEFAULT 2000,
    schedule_type       TEXT NOT NULL,
    pp_size             INTEGER NOT NULL,
    dp_size             INTEGER NOT NULL,
    world_size          INTEGER NOT NULL,
    comm_backend        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rank_config (
    experiment_id       TEXT NOT NULL REFERENCES experiments(experiment_id),
    rank                INTEGER NOT NULL,
    hostname            TEXT NOT NULL,
    ip                  TEXT NOT NULL,
    gpu_id              INTEGER NOT NULL DEFAULT 0,
    pp_stage            INTEGER NOT NULL,
    dp_group            INTEGER NOT NULL,
    mps_pct             INTEGER NOT NULL
                        CHECK (mps_pct IN (20, 40, 60, 80)),
    memory_limit_mb     INTEGER NOT NULL,
    PRIMARY KEY (experiment_id, rank)
);

CREATE TABLE IF NOT EXISTS load_metrics (
    experiment_id           TEXT NOT NULL,
    rank                    INTEGER NOT NULL,
    model_load_start_ts     TIMESTAMP,
    model_load_end_ts       TIMESTAMP,
    model_load_duration_ms  REAL,
    nccl_init_start_ts      TIMESTAMP,
    nccl_init_end_ts        TIMESTAMP,
    nccl_init_duration_ms   REAL,
    memory_after_load_mb    REAL,
    PRIMARY KEY (experiment_id, rank),
    FOREIGN KEY (experiment_id, rank) REFERENCES rank_config(experiment_id, rank)
);

CREATE TABLE IF NOT EXISTS step_metrics (
    experiment_id       TEXT NOT NULL,
    rank                INTEGER NOT NULL,
    iter                INTEGER NOT NULL,
    start_ts            TIMESTAMP,
    end_ts              TIMESTAMP,
    duration_ms         REAL,
    forward_start_ts    TIMESTAMP,
    forward_end_ts      TIMESTAMP,
    forward_ms          REAL,
    backward_start_ts   TIMESTAMP,
    backward_end_ts     TIMESTAMP,
    backward_ms         REAL,
    optimizer_start_ts  TIMESTAMP,
    optimizer_end_ts    TIMESTAMP,
    optimizer_step_ms   REAL,
    pp_send_start_ts    TIMESTAMP,
    pp_send_end_ts      TIMESTAMP,
    pp_send_ms          REAL,
    pp_recv_start_ts    TIMESTAMP,
    pp_recv_end_ts      TIMESTAMP,
    pp_recv_ms          REAL,
    pp_send_count       INTEGER,
    pp_recv_count       INTEGER,
    pp_send_bytes       BIGINT,
    pp_recv_bytes       BIGINT,
    dp_allreduce_start_ts   TIMESTAMP,
    dp_allreduce_end_ts     TIMESTAMP,
    dp_allreduce_ms     REAL,
    dp_allreduce_bytes  BIGINT,
    loss                REAL,
    lr                  REAL,
    memory_allocated_mb REAL,
    memory_reserved_mb  REAL,
    memory_peak_mb      REAL,
    PRIMARY KEY (experiment_id, rank, iter),
    FOREIGN KEY (experiment_id, rank) REFERENCES rank_config(experiment_id, rank)
);

CREATE TABLE IF NOT EXISTS global_load_metrics (
    experiment_id               TEXT PRIMARY KEY REFERENCES experiments(experiment_id),
    overall_load_start_ts       TIMESTAMP,
    overall_load_end_ts         TIMESTAMP,
    overall_load_duration_ms    REAL,
    slowest_rank                INTEGER,
    slowest_rank_load_ms        REAL,
    fastest_rank                INTEGER,
    fastest_rank_load_ms        REAL,
    overall_nccl_init_ms        REAL,
    total_memory_after_load_mb  REAL
);

CREATE TABLE IF NOT EXISTS global_step_metrics (
    experiment_id       TEXT NOT NULL REFERENCES experiments(experiment_id),
    iter                INTEGER NOT NULL,
    iter_start_ts       TIMESTAMP,
    iter_end_ts         TIMESTAMP,
    loss                REAL,
    lr                  REAL,
    iter_duration_ms    REAL,
    total_forward_ms    REAL,
    total_backward_ms   REAL,
    total_optimizer_ms  REAL,
    total_pp_send_ms    REAL,
    total_pp_recv_ms    REAL,
    total_pp_bytes      BIGINT,
    max_pp_send_ms      REAL,
    max_pp_recv_ms      REAL,
    total_dp_allreduce_ms   REAL,
    max_dp_allreduce_ms     REAL,
    dp_allreduce_bytes      BIGINT,
    slowest_rank        INTEGER,
    fastest_rank        INTEGER,
    straggler_ratio     REAL,
    PRIMARY KEY (experiment_id, iter)
);

CREATE INDEX IF NOT EXISTS idx_experiments_scheduler ON experiments(scheduler);
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_step_metrics_exp_rank ON step_metrics(experiment_id, rank);
CREATE INDEX IF NOT EXISTS idx_step_metrics_duration ON step_metrics(duration_ms DESC);
CREATE INDEX IF NOT EXISTS idx_rank_config_mps ON rank_config(mps_pct);
CREATE INDEX IF NOT EXISTS idx_global_step_exp ON global_step_metrics(experiment_id, iter);
CREATE INDEX IF NOT EXISTS idx_global_step_straggler ON global_step_metrics(straggler_ratio DESC);
"""


def insert_experiment(
    conn: sqlite3.Connection,
    experiment_id: str,
    cfg: dict,
    scheduler: str,
    mps_pcts: list[int],
) -> None:
    """Insert experiment + rank_config rows."""
    m = cfg["model"]
    t = cfg["training"]
    p = cfg["parallelism"]
    pp_size = p.get("num_stages", 2)
    world_size = p.get("world_size", 6)
    dp_size = world_size // pp_size

    git_commit = None
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(BASELINES_DIR),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        pass

    conn.execute(
        """INSERT OR REPLACE INTO experiments VALUES (
            ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,
            ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
        )""",
        (
            experiment_id,
            scheduler,
            datetime.now(timezone.utc).isoformat(),
            "running",
            None,  # failure_reason
            git_commit,
            cfg.get("deploy", {}).get("image_name", ""),
            m["model_name"],
            m["model_type"],
            m["num_layers"],
            m["embedding_dim"],
            m["num_heads"],
            m.get("d_ff", 0),
            m.get("vocab_size", 50272),
            m.get("max_seq_len", 1024),
            t["global_batch_size"],
            t["micro_batch_size"],
            t["global_batch_size"] // t["micro_batch_size"],
            t["lr"],
            t.get("min_lr", 1e-5),
            t.get("weight_decay", 0.01),
            t["max_iters"],
            t.get("warmup_iters", 50),
            t.get("grad_clip", 1.0),
            t.get("dataset", "synthetic"),
            t.get("synthetic_samples", 2000),
            p.get("schedule_type", "1f1b"),
            pp_size,
            dp_size,
            world_size,
            p.get("comm_backend", "torch_dist"),
        ),
    )

    nodes = cfg["cluster"]["nodes"]
    for i in range(min(NUM_RANKS, len(nodes))):
        node = nodes[i]
        pct = mps_pcts[i]
        mem_mb = int(math.floor(GPU_TOTAL_MB * pct / 100))
        pp_stage = i % pp_size
        dp_group = i // pp_size
        conn.execute(
            """INSERT OR REPLACE INTO rank_config VALUES (
                ?,?,?,?,?,?,?,?,?
            )""",
            (
                experiment_id,
                i,
                node.get("hostname", ""),
                node["ip"],
                node.get("gpu_id", 0),
                pp_stage,
                dp_group,
                pct,
                mem_mb,
            ),
        )

    conn.commit()


# ── One-time cluster setup ───────────────────────────────

# Infrastructure phases: only needed once per cluster.
# Order matters: mps must precede profile so that layer
# timings reflect the correct GPU configuration.
_INFRA_PHASES = ["k3s", "registry", "gpu", "mps", "profile"]

# Build phase: docker build + push to registry.
# Run before EVERY sweep so code changes in train.py /
# metrics.py etc. are baked into the image used by pods.
_BUILD_PHASE = "build"


def _run_phase(
    phase: str,
    config_path: str | Path,
    env: dict,
    timeout: int = 1800,
) -> None:
    """Run a single deploy_asteroid.sh phase, exit on failure."""
    abs_config = str(Path(config_path).resolve())
    print(
        f"  [{_ts()}] [SETUP] Running phase: {phase}",
        flush=True,
    )
    cmd = [
        "bash", str(DEPLOY_SCRIPT),
        "--config", abs_config,
        "--phase", phase,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(BASELINES_DIR),
        )
        if result.returncode != 0:
            print(
                f"  [{_ts()}] [SETUP] Phase "
                f"{phase} FAILED:\n"
                f"  {result.stderr[:500]}",
                flush=True,
            )
            sys.exit(1)
        print(
            f"  [{_ts()}] [SETUP] Phase {phase} OK",
            flush=True,
        )
    except subprocess.TimeoutExpired:
        print(
            f"  [{_ts()}] [SETUP] Phase {phase} TIMEOUT",
            flush=True,
        )
        sys.exit(1)


def run_initial_setup(
    config_path: str | Path,
) -> None:
    """Run infra-only deploy phases (k3s → profile).

    These phases set up the cluster once and do *not*
    change between experiments. The Docker image build
    is handled separately by run_build_phase() so it
    can run unconditionally even when --skip-setup is
    given.
    """
    env = os.environ.copy()
    env["ASTEROID_METRICS"] = "1"
    env["ASTEROID_METRICS_DIR"] = METRICS_REMOTE_DIR

    for phase in _INFRA_PHASES:
        _run_phase(phase, config_path, env)

    print(f"  [{_ts()}] [SETUP] Infra setup complete\n",
          flush=True)


def run_build_phase(
    config_path: str | Path,
) -> None:
    """Docker build + push to the registry.

    Always called before the experiment loop so any code
    changes (train.py, metrics.py, etc.) are baked into
    the image that pods will pull.  Skip with
    --skip-build only if you are 100% sure the image is
    already up-to-date.
    """
    env = os.environ.copy()
    env["ASTEROID_METRICS"] = "1"
    env["ASTEROID_METRICS_DIR"] = METRICS_REMOTE_DIR

    print(
        f"[{_ts()}] === Building & pushing Docker "
        f"image ===",
        flush=True,
    )
    _run_phase(_BUILD_PHASE, config_path, env,
               timeout=3600)
    print(
        f"[{_ts()}] === Build complete ===\n",
        flush=True,
    )


# ── Deploy invocation ────────────────────────────────────


def run_deploy(
    config_path: str | Path,
    strategy: str,
    timeout_s: int = 1800,
) -> tuple[bool, str]:
    """Run deploy: plan + MPS phase + redeploy."""
    env = os.environ.copy()
    env["ASTEROID_METRICS"] = "1"
    env["ASTEROID_METRICS_DIR"] = METRICS_REMOTE_DIR

    output_parts: list[str] = []

    # 0) Generate HPP plan for this strategy
    #    --stop deletes hpp_plan.json, so we must
    #    regenerate it before manifests/deploy.
    print(
        f"  [{_ts()}] [DEPLOY 1/3] Generating HPP "
        f"plan (strategy={strategy})...",
        flush=True,
    )
    plan_cmd = [
        "bash", str(DEPLOY_SCRIPT),
        "--config", str(config_path),
        "--strategy", strategy,
        "--phase", "plan",
    ]
    print(f"  [{_ts()}] [CMD] {' '.join(plan_cmd)}",
          flush=True)
    try:
        plan_result = subprocess.run(
            plan_cmd,
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
            cwd=str(BASELINES_DIR),
        )
        output_parts.append(
            plan_result.stdout + plan_result.stderr
        )
        if plan_result.returncode != 0:
            return False, "Plan phase failed:\n" + (
                output_parts[-1]
            )
    except subprocess.TimeoutExpired as e:
        return False, f"Plan phase timeout: {e}"
    except Exception as e:
        return False, f"Plan phase error: {e}"

    # 1) Apply per-node MPS config
    print(
        f"  [{_ts()}] [DEPLOY 2/3] Applying "
        f"per-node MPS percentages...",
        flush=True,
    )
    mps_cmd = [
        "bash", str(DEPLOY_SCRIPT),
        "--config", str(config_path),
        "--phase", "mps",
    ]
    print(f"  [{_ts()}] [CMD] {' '.join(mps_cmd)}",
          flush=True)
    try:
        mps_result = subprocess.run(
            mps_cmd,
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
            cwd=str(BASELINES_DIR),
        )
        output_parts.append(
            mps_result.stdout + mps_result.stderr
        )
        if mps_result.returncode != 0:
            return False, "MPS phase failed:\n" + (
                output_parts[-1]
            )
    except subprocess.TimeoutExpired as e:
        return False, f"MPS phase timeout: {e}"
    except Exception as e:
        return False, f"MPS phase error: {e}"

    # 2) Regenerate manifests + deploy pods
    print(
        f"  [{_ts()}] [DEPLOY 3/3] Regenerating "
        f"manifests & deploying pods...",
        flush=True,
    )
    deploy_cmd = [
        "bash", str(DEPLOY_SCRIPT),
        "--config", str(config_path),
        "--strategy", strategy,
        "--redeploy",
    ]
    print(f"  [{_ts()}] [CMD] "
          f"{' '.join(deploy_cmd)}", flush=True)

    try:
        result = subprocess.run(
            deploy_cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
            cwd=str(BASELINES_DIR),
        )
        output = "\n".join(output_parts) + "\n" + (
            result.stdout + result.stderr
        )
        if result.returncode != 0:
            return False, output
        return True, output
    except subprocess.TimeoutExpired as e:
        return False, f"Timeout after {timeout_s}s: {e}"
    except Exception as e:
        return False, str(e)


# ── JSONL collection ─────────────────────────────────────


def collect_jsonl(
    cfg: dict,
    local_dir: str | Path,
) -> dict[int, Path]:
    """SCP JSONL files from all nodes to local_dir."""
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    collected: dict[int, Path] = {}
    nodes = cfg["cluster"]["nodes"]

    for i, node in enumerate(nodes):
        if i >= NUM_RANKS:
            break
        ip = node["ip"]
        remote_path = (
            f"{METRICS_REMOTE_DIR}/rank_{i}.jsonl"
        )
        local_path = local_dir / f"rank_{i}.jsonl"

        try:
            _scp_cmd(ip, remote_path, str(local_path))
            collected[i] = local_path
        except Exception as e:
            print(
                f"  [{_ts()}] [WARN] Failed to collect"
                f" rank {i} from {ip}: {e}",
                flush=True,
            )

    return collected


def clear_remote_metrics(cfg: dict) -> None:
    """Remove metrics dir on all nodes before a run."""
    nodes = cfg["cluster"]["nodes"]
    for i, node in enumerate(nodes):
        if i >= NUM_RANKS:
            break
        ip = node["ip"]
        try:
            _ssh_cmd(ip, f"rm -rf {METRICS_REMOTE_DIR}")
        except Exception:
            pass


# ── Progress tracking ────────────────────────────────────


class ProgressTracker:
    """Simple file-backed progress tracker."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._data: dict[str, Any] = {"completed": []}
        if self._path.exists():
            with open(self._path) as f:
                self._data = json.load(f)

    def is_done(self, experiment_id: str) -> bool:
        return experiment_id in self._data["completed"]

    def mark_done(self, experiment_id: str) -> None:
        self._data["completed"].append(experiment_id)
        self._save()

    def _save(self) -> None:
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=2)

    @property
    def num_completed(self) -> int:
        return len(self._data["completed"])


# ── Wait for training completion ─────────────────────────


def wait_for_training(
    cfg: dict,
    timeout_s: int = 3600,
    poll_interval: int = 30,
) -> tuple[bool, str]:
    """Poll kubectl for job completion."""
    namespace = cfg.get("deploy", {}).get(
        "namespace", "default"
    )
    pp_size = cfg["parallelism"].get("num_stages", 2)
    world_size = cfg["parallelism"].get("world_size", 6)

    start = time.time()
    while time.time() - start < timeout_s:
        try:
            result = subprocess.run(
                [
                    "kubectl", "get", "jobs",
                    "-n", namespace,
                    "-o", "json",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                time.sleep(poll_interval)
                continue

            jobs = json.loads(result.stdout)
            items = jobs.get("items", [])

            if not items:
                time.sleep(poll_interval)
                continue

            # Check if ALL training jobs completed
            all_done = True
            any_failed = False
            for job in items:
                status = job.get("status", {})
                if status.get("failed", 0) > 0:
                    any_failed = True
                    break
                if status.get("succeeded", 0) < 1:
                    all_done = False

            if any_failed:
                return False, "One or more jobs failed"
            if all_done:
                return True, "All jobs completed"

        except Exception as e:
            print(
                f"  [WARN] kubectl poll error: {e}",
                flush=True,
            )

        time.sleep(poll_interval)

    return False, f"Timeout after {timeout_s}s"


# ── Stop previous run ───────────────────────────────────


def stop_previous_run(config_path: str | Path) -> None:
    """Run deploy_asteroid.sh --stop to clean up."""
    try:
        subprocess.run(
            [
                "bash", str(DEPLOY_SCRIPT),
                "--config", str(config_path),
                "--stop",
            ],
            capture_output=True,
            timeout=120,
            cwd=str(BASELINES_DIR),
        )
    except Exception:
        pass


# ── Reset experiment artifacts ───────────────────────────


def _reset_experiment_artifacts(args: argparse.Namespace) -> None:
    """Delete all experiment artifacts and exit."""
    print(f"[{_ts()}] === RESET: Clearing experiment "
          f"artifacts ===", flush=True)

    db = Path(args.db)
    if db.exists():
        db.unlink()
        print(f"  [{_ts()}] Deleted DB: {db}", flush=True)

    # Also delete WAL/SHM files
    for suffix in ("-wal", "-shm"):
        wal = Path(str(args.db) + suffix)
        if wal.exists():
            wal.unlink()

    out_dir = Path(args.output_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
        print(f"  [{_ts()}] Deleted experiment data: "
              f"{out_dir}", flush=True)

    config_path = Path(args.config).resolve()
    tmp_config = config_path.parent / "_experiment.yaml"
    if tmp_config.exists():
        tmp_config.unlink()
        print(f"  [{_ts()}] Deleted tmp config: "
              f"{tmp_config}", flush=True)

    print(f"[{_ts()}] === RESET COMPLETE ===", flush=True)


# ── Main ─────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MPS sweep experiment runner",
    )
    parser.add_argument(
        "--config",
        default=str(
            BASELINES_DIR / "configs"
            / "asteroid_default.yaml"
        ),
        help="Base YAML config path",
    )
    parser.add_argument(
        "--num-experiments",
        type=int,
        default=None,
        help=(
            "Number of experiments to run "
            "(default: all 12288)"
        ),
    )
    parser.add_argument(
        "--db",
        default=str(BASELINES_DIR / "experiments.db"),
        help="SQLite database path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for experiment sampling",
    )
    parser.add_argument(
        "--train-timeout",
        type=int,
        default=3600,
        help="Timeout per training run (seconds)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(BASELINES_DIR / "experiment_data"),
        help="Local directory for collected JSONL files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print experiments without running",
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help=(
            "Skip infra cluster setup "
            "(k3s/registry/gpu/mps/profile). "
            "Use when the cluster is already configured. "
            "Docker build+push still runs unless "
            "--skip-build is also given."
        ),
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help=(
            "Skip Docker image build and push. "
            "Use only when the image in the registry "
            "is already up-to-date with the current "
            "source code."
        ),
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Delete experiment DB, progress tracker, "
            "collected data, and tmp configs, then exit."
        ),
    )
    args = parser.parse_args()

    # Handle --reset before anything else
    if args.reset:
        _reset_experiment_artifacts(args)
        return

    # Load base config
    base_cfg = load_yaml(args.config)
    config_path = Path(args.config).resolve()

    # Generate sweep
    sweep = generate_sweep(
        num_experiments=args.num_experiments,
        seed=args.seed,
    )
    total = len(sweep)
    print(
        f"[{_ts()}] === MPS Sweep: {total} "
        f"experiments ===",
        flush=True,
    )

    if args.dry_run:
        for i, (sched, pcts) in enumerate(sweep):
            eid = make_experiment_id(sched, pcts)
            pct_str = ",".join(str(p) for p in pcts)
            print(f"  {i+1:>5d}. {eid} {sched:>10s}"
                  f" [{pct_str}]")
        return

    # ── One-time cluster setup ───────────────────────
    # Infra: k3s → registry → gpu → mps → profile
    # (only needed once; skip with --skip-setup)
    if not args.skip_setup:
        print(
            f"[{_ts()}] === Running infra cluster "
            f"setup ===",
            flush=True,
        )
        run_initial_setup(config_path)
    else:
        print(
            f"[{_ts()}] === Skipping infra setup "
            f"(--skip-setup) ===",
            flush=True,
        )

    # ── Docker build + push ──────────────────────────
    # Always rebuild so code changes are in the image.
    # Skip only with --skip-build (image is current).
    if not args.skip_build:
        run_build_phase(config_path)
    else:
        print(
            f"[{_ts()}] === Skipping Docker build "
            f"(--skip-build) ===",
            flush=True,
        )

    # Init DB + progress tracker
    conn = init_db(args.db)
    progress = ProgressTracker(
        Path(args.output_dir) / "progress.json"
    )
    Path(args.output_dir).mkdir(
        parents=True, exist_ok=True
    )

    # Temp config for mutations
    tmp_config = config_path.parent / "_experiment.yaml"

    failed_count = 0
    skipped_count = 0
    sweep_start = time.time()
    experiment_durations: list[float] = []

    for i, (scheduler, mps_pcts) in enumerate(sweep):
        experiment_id = make_experiment_id(
            scheduler, mps_pcts
        )
        pct_str = ",".join(str(p) for p in mps_pcts)

        # Progress bar
        done = progress.num_completed + skipped_count
        print(
            f"\n{'='*60}\n"
            f"[{_ts()}] [{done+1}/{total}] Experiment "
            f"{experiment_id}\n"
            f"  scheduler={scheduler}"
            f"  MPS=[{pct_str}]\n"
            f"{'='*60}",
            flush=True,
        )

        # Skip if already done
        if progress.is_done(experiment_id):
            print(f"  [{_ts()}] SKIP (already completed)",
                  flush=True)
            skipped_count += 1
            continue

        exp_start = time.time()

        # 1) Mutate config
        print(f"  [{_ts()}] [STEP 1/8] Mutating "
              f"config...", flush=True)
        exp_cfg = apply_experiment_config(
            base_cfg, scheduler, mps_pcts,
        )
        save_yaml(exp_cfg, tmp_config)

        # 2) Insert experiment record
        print(f"  [{_ts()}] [STEP 2/8] Inserting DB "
              f"record...", flush=True)
        insert_experiment(
            conn, experiment_id, exp_cfg,
            scheduler, mps_pcts,
        )

        # 3) Stop any previous run FIRST so pods are
        #    dead before we wipe the metrics dir.
        #    (Old pod was still running during clear,
        #    recreating the JSONL → new pod appended to
        #    stale data → >10 iters per experiment.)
        print(f"  [{_ts()}] [STEP 3/8] Stopping "
              f"previous run...", flush=True)
        stop_previous_run(tmp_config)
        time.sleep(5)

        # 4) Clear remote metrics AFTER pods are gone
        print(f"  [{_ts()}] [STEP 4/8] Clearing remote "
              f"metrics...", flush=True)
        clear_remote_metrics(exp_cfg)

        # 5) Deploy
        print(f"  [{_ts()}] [STEP 5/8] Deploying "
              f"(plan -> mps -> redeploy)...",
              flush=True)
        deploy_ok, deploy_output = run_deploy(
            tmp_config, scheduler,
        )
        if not deploy_ok:
            print(
                f"  [{_ts()}] FAILED (deploy): "
                f"{deploy_output[:500]}",
                flush=True,
            )
            conn.execute(
                "UPDATE experiments SET status=?,"
                " failure_reason=? WHERE experiment_id=?",
                ("failed", deploy_output[:1000],
                 experiment_id),
            )
            conn.commit()
            failed_count += 1
            continue

        # 6) Wait for training
        print(f"  [{_ts()}] [STEP 6/8] Waiting for "
              f"training to complete...", flush=True)
        train_ok, train_msg = wait_for_training(
            exp_cfg, timeout_s=args.train_timeout,
        )
        if not train_ok:
            print(
                f"  [{_ts()}] FAILED (training): "
                f"{train_msg}",
                flush=True,
            )
            conn.execute(
                "UPDATE experiments SET status=?,"
                " failure_reason=? WHERE experiment_id=?",
                ("failed", train_msg[:1000],
                 experiment_id),
            )
            conn.commit()
            failed_count += 1
            continue

        # 7) Collect JSONL
        print(f"  [{_ts()}] [STEP 7/8] Collecting JSONL "
              f"files...", flush=True)
        exp_data_dir = (
            Path(args.output_dir) / experiment_id
        )
        collected = collect_jsonl(exp_cfg, exp_data_dir)
        print(
            f"  [{_ts()}] Collected "
            f"{len(collected)}/{NUM_RANKS} JSONL files",
            flush=True,
        )

        # 8) Mark completed
        print(f"  [{_ts()}] [STEP 8/8] Marking "
              f"completed...", flush=True)
        conn.execute(
            "UPDATE experiments SET status=?"
            " WHERE experiment_id=?",
            ("completed", experiment_id),
        )
        conn.commit()
        progress.mark_done(experiment_id)

        exp_elapsed = time.time() - exp_start
        experiment_durations.append(exp_elapsed)
        print(
            f"  [{_ts()}] DONE in "
            f"{_fmt_duration(exp_elapsed)} "
            f"({progress.num_completed} completed,"
            f" {failed_count} failed)",
            flush=True,
        )

    # Cleanup tmp config
    if tmp_config.exists():
        tmp_config.unlink()

    # Summary
    sweep_elapsed = time.time() - sweep_start
    avg_dur = (
        sum(experiment_durations)
        / len(experiment_durations)
        if experiment_durations else 0.0
    )
    min_dur = min(experiment_durations) if experiment_durations else 0.0
    max_dur = max(experiment_durations) if experiment_durations else 0.0

    print(
        f"\n{'='*60}\n"
        f"[{_ts()}] SWEEP COMPLETE\n"
        f"{'='*60}\n"
        f"  Total experiments:  {total}\n"
        f"  Completed:          {progress.num_completed}\n"
        f"  Failed:             {failed_count}\n"
        f"  Skipped:            {skipped_count}\n"
        f"  Pass rate:          "
        f"{progress.num_completed}/{progress.num_completed + failed_count}"
        f" ({(progress.num_completed / max(progress.num_completed + failed_count, 1) * 100):.1f}%)\n"
        f"{'─'*60}\n"
        f"  Total wall time:    {_fmt_duration(sweep_elapsed)}\n"
        f"  Avg per experiment: {_fmt_duration(avg_dur)}\n"
        f"  Fastest experiment: {_fmt_duration(min_dur)}\n"
        f"  Slowest experiment: {_fmt_duration(max_dur)}\n"
        f"{'─'*60}\n"
        f"  DB:    {args.db}\n"
        f"  Data:  {args.output_dir}\n"
        f"{'='*60}",
        flush=True,
    )

    conn.close()


if __name__ == "__main__":
    main()
