#!/usr/bin/env python3
"""Planner-only experiment runner for MPS sweep.

Generates unique (scheduler, R0%, R1%, ..., R7%)
experiment tuples, modifies the YAML config, invokes
deploy_asteroid.sh with --phase plan, and stores each
generated hpp_plan.json into SQLite.

Usage
-----
    python3 baselines/scripts/run_experiments_plan_only.py \
        --config baselines/configs/asteroid_default.yaml \
        --num-experiments 20 \
        --db experiments.db

Env vars
--------
    ASTEROID_METRICS=1     — passed through to deploy script
    ASTEROID_METRICS_DIR   — passed through to deploy script
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

MPS_CHOICES = [20, 40, 60 , 70 , 80]
SCHEDULERS = ["confident", "dtfm", "asteroid"]
DEFAULT_SCHEDULERS = ["asteroid"]
GPU_TOTAL_MB = 45459   # L40S usable memory
NUM_RANKS = 8
METRICS_REMOTE_DIR = "/tmp/asteroid_metrics"

# Subnet pairs: nodes within a pair share fast links and
# are always grouped together by the planner.  Assigning
# the same MPS% to both nodes in a pair reduces sweep size
# from rank-level multisets C(K+7,8) to pair-level multisets
# C(K+3,4) per scheduler, where K=len(MPS_CHOICES).
SUBNET_PAIRS = [
    (0, 1),   # subnet 54  (R0↔R1)
    (2, 5),   # subnet 53  (R2↔R5)
    (3, 4),   # subnet 56  (R3↔R4)
    (6, 7),   # subnet 57  (R6↔R7)
]

BASELINES_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASELINES_DIR / "scripts"
DEPLOY_SCRIPT = SCRIPTS_DIR / "deploy_asteroid.sh"
PLAN_FILE = BASELINES_DIR / "hpp_plan.json"


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
    schedulers: list[str] | None = None,
) -> list[tuple[str, list[int]]]:
    """Generate (scheduler, [R0%, ... R7%]) tuples.

    Nodes inside each subnet pair are tied to the same MPS%.
    We therefore enumerate MPS multisets across pair-slots
    (len(SUBNET_PAIRS)) and expand each pair assignment to
    rank-level [R0..R7].

    If num_experiments is given, samples randomly.
    """
    def _expand_pair_combo(
        pair_combo: tuple[int, ...],
    ) -> list[int]:
        rank_pcts: list[int | None] = [None] * NUM_RANKS
        for (r0, r1), pct in zip(
            SUBNET_PAIRS, pair_combo
        ):
            rank_pcts[r0] = pct
            rank_pcts[r1] = pct
        if any(v is None for v in rank_pcts):
            raise ValueError(
                "SUBNET_PAIRS must cover all ranks"
            )
        return [int(v) for v in rank_pcts]

    all_combos: list[tuple[str, list[int]]] = []
    scheduler_list = schedulers if schedulers else SCHEDULERS
    for sched in scheduler_list:
        for pair_combo in itertools.combinations_with_replacement(
            MPS_CHOICES, len(SUBNET_PAIRS)
        ):
            all_combos.append(
                (
                    sched,
                    _expand_pair_combo(pair_combo),
                )
            )

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
    _migrate_rank_config_check(conn)
    conn.commit()
    return conn


def _migrate_rank_config_check(
    conn: sqlite3.Connection,
) -> None:
    """Upgrade rank_config mps_pct CHECK for legacy DBs.

    Older schema only allowed (20,40,60,80), which breaks
    sweeps that include values like 70.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master "
        "WHERE type='table' AND name='rank_config'"
    ).fetchone()
    if not row or not row[0]:
        return

    table_sql = " ".join(str(row[0]).split()).lower()
    legacy = "mps_pct in (20, 40, 60, 80)"
    if legacy not in table_sql:
        return

    # Recreate table with a broader percentage range.
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS rank_config_new (
            experiment_id       TEXT NOT NULL REFERENCES experiments(experiment_id),
            rank                INTEGER NOT NULL,
            hostname            TEXT NOT NULL,
            ip                  TEXT NOT NULL,
            gpu_id              INTEGER NOT NULL DEFAULT 0,
            pp_stage            INTEGER NOT NULL,
            dp_group            INTEGER NOT NULL,
            mps_pct             INTEGER NOT NULL
                                CHECK (mps_pct >= 1 AND mps_pct <= 100),
            memory_limit_mb     INTEGER NOT NULL,
            PRIMARY KEY (experiment_id, rank)
        )"""
    )
    conn.execute(
        """INSERT INTO rank_config_new (
            experiment_id, rank, hostname, ip, gpu_id,
            pp_stage, dp_group, mps_pct, memory_limit_mb
        )
        SELECT experiment_id, rank, hostname, ip, gpu_id,
               pp_stage, dp_group, mps_pct, memory_limit_mb
        FROM rank_config"""
    )
    conn.execute("DROP TABLE rank_config")
    conn.execute(
        "ALTER TABLE rank_config_new RENAME TO rank_config"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_rank_config_mps "
        "ON rank_config(mps_pct)"
    )
    conn.execute("PRAGMA foreign_keys=ON")


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
                        CHECK (mps_pct >= 1 AND mps_pct <= 100),
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
    pp_send_cumulative_ms REAL,
    pp_recv_cumulative_ms REAL,
    forward_compute_ms  REAL,
    backward_compute_ms REAL,
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

CREATE TABLE IF NOT EXISTS microbatch_op_metrics (
    experiment_id       TEXT NOT NULL,
    rank                INTEGER NOT NULL,
    iter                INTEGER NOT NULL,
    microbatch          INTEGER NOT NULL,
    op_seq              INTEGER NOT NULL,
    op_kind             TEXT NOT NULL,
    stream              TEXT NOT NULL,
    start_offset_ms     REAL,
    end_offset_ms       REAL,
    duration_ms         REAL,
    nbytes              BIGINT,
    peer_rank           INTEGER,
    PRIMARY KEY (experiment_id, rank, iter, microbatch, op_seq),
    FOREIGN KEY (experiment_id, rank, iter)
        REFERENCES step_metrics(experiment_id, rank, iter)
);

CREATE TABLE IF NOT EXISTS microbatch_metrics (
    experiment_id           TEXT NOT NULL,
    rank                    INTEGER NOT NULL,
    iter                    INTEGER NOT NULL,
    microbatch              INTEGER NOT NULL,
    compute_nonoverlap_ms   REAL,
    comm_nonoverlap_ms      REAL,
    total_union_ms          REAL,
    compute_sum_ms          REAL,
    comm_sum_ms             REAL,
    total_sum_ms            REAL,
    PRIMARY KEY (experiment_id, rank, iter, microbatch),
    FOREIGN KEY (experiment_id, rank, iter)
        REFERENCES step_metrics(experiment_id, rank, iter)
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
    total_pp_send_cumulative_ms REAL,
    total_pp_recv_cumulative_ms REAL,
    total_pp_bytes      BIGINT,
    max_pp_send_ms      REAL,
    max_pp_recv_ms      REAL,
    max_pp_send_cumulative_ms REAL,
    max_pp_recv_cumulative_ms REAL,
    total_forward_compute_ms REAL,
    total_backward_compute_ms REAL,
    total_dp_allreduce_ms   REAL,
    max_dp_allreduce_ms     REAL,
    dp_allreduce_bytes      BIGINT,
    slowest_rank        INTEGER,
    fastest_rank        INTEGER,
    straggler_ratio     REAL,
    PRIMARY KEY (experiment_id, iter)
);

CREATE TABLE IF NOT EXISTS global_microbatch_metrics (
    experiment_id                   TEXT NOT NULL
        REFERENCES experiments(experiment_id),
    iter                            INTEGER NOT NULL,
    microbatch                      INTEGER NOT NULL,
    total_compute_nonoverlap_ms     REAL,
    total_comm_nonoverlap_ms        REAL,
    total_union_ms                  REAL,
    total_compute_sum_ms            REAL,
    total_comm_sum_ms               REAL,
    total_sum_ms                    REAL,
    max_compute_nonoverlap_ms       REAL,
    max_comm_nonoverlap_ms          REAL,
    max_total_union_ms              REAL,
    slowest_rank                    INTEGER,
    fastest_rank                    INTEGER,
    straggler_ratio                 REAL,
    PRIMARY KEY (experiment_id, iter, microbatch)
);

CREATE INDEX IF NOT EXISTS idx_experiments_scheduler ON experiments(scheduler);
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_step_metrics_exp_rank ON step_metrics(experiment_id, rank);
CREATE INDEX IF NOT EXISTS idx_step_metrics_duration ON step_metrics(duration_ms DESC);
CREATE INDEX IF NOT EXISTS idx_micro_op_exp_iter ON microbatch_op_metrics(experiment_id, iter);
CREATE INDEX IF NOT EXISTS idx_micro_op_rank_iter ON microbatch_op_metrics(experiment_id, rank, iter);
CREATE INDEX IF NOT EXISTS idx_micro_metrics_exp_iter ON microbatch_metrics(experiment_id, iter);
CREATE INDEX IF NOT EXISTS idx_micro_metrics_rank_iter ON microbatch_metrics(experiment_id, rank, iter);
CREATE INDEX IF NOT EXISTS idx_rank_config_mps ON rank_config(mps_pct);
CREATE INDEX IF NOT EXISTS idx_global_step_exp ON global_step_metrics(experiment_id, iter);
CREATE INDEX IF NOT EXISTS idx_global_step_straggler ON global_step_metrics(straggler_ratio DESC);
CREATE INDEX IF NOT EXISTS idx_global_micro_exp_iter ON global_microbatch_metrics(experiment_id, iter);

CREATE TABLE IF NOT EXISTS planner_results (
    experiment_id           TEXT PRIMARY KEY REFERENCES experiments(experiment_id),
    planner_start_ts        TIMESTAMP,
    planner_end_ts          TIMESTAMP,
    planner_duration_ms     REAL,
    plan_path               TEXT NOT NULL,
    estimated_latency_ms    REAL,
    num_stages              INTEGER,
    world_size              INTEGER,
    partition_points_json   TEXT,
    device_groups_json      TEXT,
    micro_batch_alloc_json  TEXT,
    plan_json               TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS planner_rank_config (
    experiment_id   TEXT NOT NULL REFERENCES experiments(experiment_id),
    rank            INTEGER NOT NULL,
    planner_stage   INTEGER,
    dp_slot         INTEGER,
    hostname        TEXT,
    ip              TEXT,
    nic             TEXT,
    gpu_id          INTEGER,
    memory_mb       INTEGER,
    architecture    TEXT,
    PRIMARY KEY (experiment_id, rank)
);

CREATE INDEX IF NOT EXISTS idx_planner_results_latency
    ON planner_results(estimated_latency_ms DESC);
CREATE INDEX IF NOT EXISTS idx_planner_rank_stage
    ON planner_rank_config(experiment_id, planner_stage);
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


def _safe_int(value: Any) -> int | None:
    """Convert value to int when possible; else None."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def insert_planner_result(
    conn: sqlite3.Connection,
    experiment_id: str,
    plan: dict[str, Any],
    planner_start_ts: str | None = None,
    planner_end_ts: str | None = None,
    planner_duration_ms: float | None = None,
) -> None:
    """Persist planner output from hpp_plan.json."""
    partition_points = plan.get("partition_points")
    device_groups = plan.get("device_groups")
    micro_batch_alloc = plan.get("micro_batch_alloc")

    conn.execute(
        """INSERT OR REPLACE INTO planner_results VALUES (
            ?,?,?,?,?,?,?,?,?,?,?,?
        )""",
        (
            experiment_id,
            planner_start_ts,
            planner_end_ts,
            planner_duration_ms,
            str(PLAN_FILE),
            plan.get("estimated_latency_ms"),
            plan.get("num_stages"),
            plan.get("world_size"),
            json.dumps(partition_points, sort_keys=True),
            json.dumps(device_groups, sort_keys=True),
            json.dumps(micro_batch_alloc, sort_keys=True),
            json.dumps(plan, sort_keys=True),
        ),
    )

    # Rebuild rank-level planner mapping for this experiment.
    conn.execute(
        "DELETE FROM planner_rank_config WHERE experiment_id=?",
        (experiment_id,),
    )

    stage_map: dict[int, int] = {}
    dp_slot_map: dict[int, int] = {}
    if isinstance(device_groups, dict):
        for stage_raw, rank_list in device_groups.items():
            stage_id = _safe_int(stage_raw)
            if stage_id is None or not isinstance(rank_list, list):
                continue
            for slot, rank_raw in enumerate(rank_list):
                rank = _safe_int(rank_raw)
                if rank is None:
                    continue
                stage_map[rank] = stage_id
                dp_slot_map[rank] = slot

    node_mapping_raw = plan.get("node_mapping")
    node_mapping: dict[int, dict[str, Any]] = {}
    if isinstance(node_mapping_raw, dict):
        for rank_raw, info_raw in node_mapping_raw.items():
            rank = _safe_int(rank_raw)
            if rank is None:
                continue
            if isinstance(info_raw, dict):
                node_mapping[rank] = info_raw
            else:
                node_mapping[rank] = {}

    all_ranks = sorted(
        set(stage_map.keys()) | set(node_mapping.keys())
    )
    for rank in all_ranks:
        info = node_mapping.get(rank, {})
        conn.execute(
            """INSERT OR REPLACE INTO planner_rank_config VALUES (
                ?,?,?,?,?,?,?,?,?,?
            )""",
            (
                experiment_id,
                rank,
                stage_map.get(rank),
                dp_slot_map.get(rank),
                info.get("hostname"),
                info.get("ip"),
                info.get("nic"),
                _safe_int(info.get("gpu_id")),
                _safe_int(info.get("memory_mb")),
                info.get("architecture"),
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


def run_plan_phase(
    config_path: str | Path,
    strategy: str,
    timeout_s: int = 600,
) -> tuple[bool, str, dict[str, Any] | None]:
    """Run deploy_asteroid.sh in planner-only mode."""
    env = os.environ.copy()
    env["ASTEROID_METRICS"] = "1"
    env["ASTEROID_METRICS_DIR"] = METRICS_REMOTE_DIR

    # deploy_asteroid.sh short-circuits if the plan already exists.
    # Remove it so each sweep point always regenerates the plan.
    try:
        PLAN_FILE.unlink(missing_ok=True)
    except Exception:
        pass

    plan_cmd = [
        "bash", str(DEPLOY_SCRIPT),
        "--config", str(config_path),
        "--strategy", strategy,
        "--phase", "plan",
    ]
    print(f"  [{_ts()}] [CMD] {' '.join(plan_cmd)}",
          flush=True)

    planner_start_ts = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    try:
        plan_result = subprocess.run(
            plan_cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
            cwd=str(BASELINES_DIR),
        )
    except subprocess.TimeoutExpired as e:
        return False, f"Plan phase timeout: {e}", None
    except Exception as e:
        return False, f"Plan phase error: {e}", None

    planner_end_ts = datetime.now(timezone.utc).isoformat()
    planner_duration_ms = (time.time() - t0) * 1000.0
    output = plan_result.stdout + plan_result.stderr
    if plan_result.returncode != 0:
        return False, output, None

    if not PLAN_FILE.exists():
        return False, (
            "Plan phase completed but hpp_plan.json was not found at "
            f"{PLAN_FILE}"
        ), None

    try:
        with open(PLAN_FILE) as f:
            plan = json.load(f)
    except Exception as e:
        return False, f"Failed to read {PLAN_FILE}: {e}", None

    return True, output, {
        "plan": plan,
        "planner_start_ts": planner_start_ts,
        "planner_end_ts": planner_end_ts,
        "planner_duration_ms": planner_duration_ms,
    }


def run_deploy(
    config_path: str | Path,
    strategy: str,
    timeout_s: int = 1800,
) -> tuple[bool, str, dict[str, Any] | None]:
    """Run deploy: plan + MPS phase + redeploy."""
    env = os.environ.copy()
    env["ASTEROID_METRICS"] = "1"
    env["ASTEROID_METRICS_DIR"] = METRICS_REMOTE_DIR

    output_parts: list[str] = []

    # 0) Generate HPP plan for this strategy.
    print(
        f"  [{_ts()}] [DEPLOY 1/3] Generating HPP "
        f"plan (strategy={strategy})...",
        flush=True,
    )
    plan_ok, plan_output, plan_meta = run_plan_phase(
        config_path=config_path,
        strategy=strategy,
        timeout_s=600,
    )
    output_parts.append(plan_output)
    if not plan_ok:
        return False, "Plan phase failed:\n" + plan_output, None

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
            ), plan_meta
    except subprocess.TimeoutExpired as e:
        return False, f"MPS phase timeout: {e}", plan_meta
    except Exception as e:
        return False, f"MPS phase error: {e}", plan_meta

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
            return False, output, plan_meta
        return True, output, plan_meta
    except subprocess.TimeoutExpired as e:
        return False, f"Timeout after {timeout_s}s: {e}", plan_meta
    except Exception as e:
        return False, str(e), plan_meta


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
        if experiment_id not in self._data["completed"]:
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


def cleanup_plan_file() -> None:
    """Remove only hpp_plan.json (planner artifact)."""
    try:
        PLAN_FILE.unlink(missing_ok=True)
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
        description="Planner-only MPS sweep runner",
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
            "(default: full sweep)"
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
        "--schedulers",
        default="asteroid",
        help=(
            "Comma-separated scheduler list to run. "
            "Default: asteroid"
        ),
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
        "--skip-existing",
        action="store_true",
        help=(
            "Skip experiments already in progress.json. "
            "Default behavior is to rerun and update DB rows."
        ),
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
    planner_only = True

    # Handle --reset before anything else
    if args.reset:
        _reset_experiment_artifacts(args)
        return

    # Load base config
    base_cfg = load_yaml(args.config)
    config_path = Path(args.config).resolve()

    selected_schedulers = [
        token.strip().lower()
        for token in args.schedulers.split(",")
        if token.strip()
    ]
    if not selected_schedulers:
        selected_schedulers = list(DEFAULT_SCHEDULERS)
    invalid = sorted(
        set(selected_schedulers) - set(SCHEDULERS)
    )
    if invalid:
        raise ValueError(
            "Invalid scheduler(s): "
            + ",".join(invalid)
            + f". Valid: {','.join(SCHEDULERS)}"
        )

    # Generate sweep
    sweep = generate_sweep(
        num_experiments=args.num_experiments,
        seed=args.seed,
        schedulers=selected_schedulers,
    )
    total = len(sweep)
    print(
        f"[{_ts()}] === MPS Sweep: {total} "
        f"experiments ===",
        flush=True,
    )
    if planner_only:
        print(
            f"[{_ts()}] Mode: planner-only "
            f"(phase plan only, no training)",
            flush=True,
        )

    if args.dry_run:
        for i, (sched, pcts) in enumerate(sweep):
            eid = make_experiment_id(sched, pcts)
            pct_str = ",".join(str(p) for p in pcts)
            print(f"  {i+1:>5d}. {eid} {sched:>10s}"
                  f" [{pct_str}]")
        return

    # Planner-only mode intentionally avoids the full deployment
    # pipeline. We only run per-experiment:
    #   delete old plan -> --phase plan -> delete plan
    print(
        f"[{_ts()}] === Planner-only path: skipping "
        f"infra setup/build/deploy phases ===",
        flush=True,
    )
    print(
        f"[{_ts()}] Schedulers: "
        f"{','.join(selected_schedulers)}",
        flush=True,
    )
    print(
        f"[{_ts()}] Skip existing: "
        f"{'yes' if args.skip_existing else 'no (rerun + DB update)'}",
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
    completed_count = 0
    sweep_start = time.time()
    experiment_durations: list[float] = []

    for i, (scheduler, mps_pcts) in enumerate(sweep):
        experiment_id = make_experiment_id(
            scheduler, mps_pcts
        )
        pct_str = ",".join(str(p) for p in mps_pcts)

        # Progress bar
        print(
            f"\n{'='*60}\n"
            f"[{_ts()}] [{i+1}/{total}] Experiment "
            f"{experiment_id}\n"
            f"  scheduler={scheduler}"
            f"  MPS=[{pct_str}]\n"
            f"{'='*60}",
            flush=True,
        )

        # Skip if already done
        if args.skip_existing and progress.is_done(experiment_id):
            print(f"  [{_ts()}] SKIP (already completed)",
                  flush=True)
            skipped_count += 1
            continue

        exp_start = time.time()

        # 1) Mutate config
        print(f"  [{_ts()}] [STEP 1/6] Mutating "
              f"config...", flush=True)
        exp_cfg = apply_experiment_config(
            base_cfg, scheduler, mps_pcts,
        )
        save_yaml(exp_cfg, tmp_config)

        # 2) Insert experiment record
        print(f"  [{_ts()}] [STEP 2/6] Inserting DB "
              f"record...", flush=True)
        insert_experiment(
            conn, experiment_id, exp_cfg,
            scheduler, mps_pcts,
        )

        # 3) Ensure stale planner output is removed before
        #    this experiment's planner run.
        print(f"  [{_ts()}] [STEP 3/6] Cleaning stale "
              f"plan file...", flush=True)
        cleanup_plan_file()

        if planner_only:
            # 4) Run planner only.
            print(f"  [{_ts()}] [STEP 4/6] Running planner "
                  f"(phase=plan only)...", flush=True)
            plan_ok, plan_output, plan_meta = run_plan_phase(
                tmp_config, scheduler,
            )
            if not plan_ok:
                print(
                    f"  [{_ts()}] FAILED (planner): "
                    f"{plan_output[:500]}",
                    flush=True,
                )
                conn.execute(
                    "UPDATE experiments SET status=?,"
                    " failure_reason=? WHERE experiment_id=?",
                    ("failed", plan_output[:1000],
                     experiment_id),
                )
                conn.commit()
                failed_count += 1
                continue

            # 5) Persist planner output + mark complete
            print(f"  [{_ts()}] [STEP 5/6] Saving planner "
                  f"result + marking completed...",
                  flush=True)
            assert plan_meta is not None
            insert_planner_result(
                conn=conn,
                experiment_id=experiment_id,
                plan=plan_meta["plan"],
                planner_start_ts=plan_meta.get(
                    "planner_start_ts"
                ),
                planner_end_ts=plan_meta.get(
                    "planner_end_ts"
                ),
                planner_duration_ms=plan_meta.get(
                    "planner_duration_ms"
                ),
            )
            conn.execute(
                "UPDATE experiments SET status=?,"
                " failure_reason=? WHERE experiment_id=?",
                ("completed", None, experiment_id),
            )
            conn.commit()
            progress.mark_done(experiment_id)
            completed_count += 1

            # Remove plan artifact between experiments.
            print(f"  [{_ts()}] [STEP 6/6] Post-plan cleanup "
                  f"(delete hpp_plan.json)...", flush=True)
            cleanup_plan_file()

            exp_elapsed = time.time() - exp_start
            experiment_durations.append(exp_elapsed)
            print(
                f"  [{_ts()}] DONE in "
                f"{_fmt_duration(exp_elapsed)} "
                f"({completed_count} completed,"
                f" {failed_count} failed)",
                flush=True,
            )
            continue

        # 4) Clear remote metrics AFTER pods are gone
        print(f"  [{_ts()}] [STEP 4/8] Clearing remote "
              f"metrics...", flush=True)
        clear_remote_metrics(exp_cfg)

        # 5) Deploy
        print(f"  [{_ts()}] [STEP 5/8] Deploying "
              f"(plan -> mps -> redeploy)...",
              flush=True)
        deploy_ok, deploy_output, plan_meta = run_deploy(
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

        # Persist planner output captured during deploy.
        if plan_meta is not None:
            insert_planner_result(
                conn=conn,
                experiment_id=experiment_id,
                plan=plan_meta["plan"],
                planner_start_ts=plan_meta.get(
                    "planner_start_ts"
                ),
                planner_end_ts=plan_meta.get(
                    "planner_end_ts"
                ),
                planner_duration_ms=plan_meta.get(
                    "planner_duration_ms"
                ),
            )

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
        completed_count += 1

        exp_elapsed = time.time() - exp_start
        experiment_durations.append(exp_elapsed)
        print(
            f"  [{_ts()}] DONE in "
            f"{_fmt_duration(exp_elapsed)} "
            f"({completed_count} completed,"
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
        f"  Completed:          {completed_count}\n"
        f"  Failed:             {failed_count}\n"
        f"  Skipped:            {skipped_count}\n"
        f"  Pass rate:          "
        f"{completed_count}/{completed_count + failed_count}"
        f" ({(completed_count / max(completed_count + failed_count, 1) * 100):.1f}%)\n"
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
