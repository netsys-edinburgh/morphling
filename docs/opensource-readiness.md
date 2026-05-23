# PRD — Open-source readiness for DeviceEmulator (Morphling)

**Reference:** [`batchgen-project/batchgen`](https://github.com/batchgen-project/batchgen) (local clone: `~/batchgen`)
**Context:** This release accompanies the **EdgeSys '26** paper (camera-ready,
going public). Treat the repo as the canonical companion code; not a research
artifact requiring formal Artifact Evaluation, but it must look like a paper's
code release: real authors, paper citation, figures the paper references.
**Status:** Draft v2 — incorporates user answers on EdgeSys scope.
**Owner:** Bessus

---

## 1. Goal

Bring DeviceEmulator to the same "production-grade open-source" bar as BatchGen,
**plus** the conventions a reader of the EdgeSys '26 paper expects when they
click the GitHub link in the camera-ready PDF:

1. Understand what it is in under 30 seconds (compact README + badges + tagline).
2. See the paper citation, link, and authors immediately ("Paper" section near
   the top of the README).
3. Install it with one command (`make install` / `docker build`).
4. Find contribution norms (`CONTRIBUTING.md`, PR template, issue templates).
5. Open a well-scoped PR/issue and have CI tell them whether it builds.
6. Pull a clean sdist/wheel that actually contains all C++/CUDA sources.

**Out of scope** (explicitly):
- Formal Artifact Evaluation (no REPRODUCE.md, no kick-the-tires script, no
  claim-to-experiment table). User decision.
- LDPC trace pipeline — the current README's "Paper evaluation pipeline"
  section references `data/ldpc_trace_*.csv` which the user confirmed is
  unrelated to EdgeSys '26. **Delete from README**, do not migrate.
- Reproducing figures — user will provide the figure list and reproduction
  commands in a later pass; this PRD only adds a placeholder section.
- Semantic-version releases, PyPI publishing, Docker Hub publishing, logo
  design.

## 2. Non-goals

- No code refactor inside `morphling/` or `csrc/`.
- No proto changes.
- No CI matrix expansion (single Ubuntu 22.04 / Python 3.10 target stays).
- No new dependencies in `requirements.txt`.

## 3. Acceptance criteria (whole PRD)

A reader of the EdgeSys '26 paper clicking the repo link can answer YES to all
of:

- [ ] `README.md` ≤ 180 lines, has badges, tagline, "About", **"Paper"**,
      "Install", "Quick Start", "Documentation" index, "Citation",
      "Acknowledgements".
- [ ] "Paper" section appears near the top with: paper title, venue
      ("EdgeSys '26"), authors, link/DOI placeholder, BibTeX-ready citation
      block. **Owner fills in the figure-reproduction commands later** — PRD
      only adds the section skeleton.
- [ ] LDPC paper-evaluation pipeline (currently README lines 115–208) is
      **deleted** from README. Scripts in `scripts/` stay untouched (separate
      cleanup, not in this PRD).
- [ ] `LICENSE` (not `LICENSE.txt`) exists; GitHub UI shows "Apache 2.0".
- [ ] `CONTRIBUTING.md` exists with commit format + merge policy.
- [ ] `CITATION.cff` exists at repo root so GitHub's "Cite this repository"
      button works.
- [ ] `.github/CODEOWNERS`, `.github/PULL_REQUEST_TEMPLATE.md`,
      `.github/ISSUE_TEMPLATE/bug_report.yml`,
      `.github/ISSUE_TEMPLATE/feature_request.yml` exist.
- [ ] `Makefile` at repo root with at least: `help install install-dev test
      docker-build docker-test format clean`.
- [ ] `MANIFEST.in` includes every C++/CUDA/proto/CMake source under `csrc/`,
      `external/`, `proto/`, `cmake/`.
- [ ] `requirements-lint.txt` exists; lint deps removed from `requirements.txt`.
- [ ] `.github/workflows/build.yml` builds the Docker image on PR and reports
      green/red.
- [ ] No agent-tooling artefacts tracked: `package.json`,
      `package-lock.json` removed from index.
- [ ] `docker build -t device-emulator:latest .` still succeeds.
- [ ] `docker run --rm --gpus all device-emulator:latest python3 -m pytest
      tests -v` still passes (no regression vs. current main).

## 4. Workstreams

Six workstreams, executed in order. Each is independently mergeable.
W1.5 is new in v2 (paper metadata); W2 is reduced (delete LDPC pipeline rather
than migrate it).

---

### W1 — Meta files (mechanical)

**Goal:** Add the standard set of open-source community files so GitHub's
"Community Standards" check turns green and contributors know the rules.

**Files to add:**

| Path | Source/template |
|------|-----------------|
| `CONTRIBUTING.md` | Adapt from `~/batchgen/CONTRIBUTING.md`. Keep: merge policy, Angular commit format, pre-commit setup. Adjust: project name, dev environment uses `docker build` + `docker run pytest` per `CLAUDE.md` §1–2 (not `pip install -r requirements-lint.txt` directly), reference to `tests/cpp/README.md` for C++ tests. |
| `.github/CODEOWNERS` | `* @drunkcoding` as default owner. |
| `.github/PULL_REQUEST_TEMPLATE.md` | Adapt from batchgen; checklist references our `CONTRIBUTING.md`, the Docker-only test policy, and "rebuild image after code change" rule. |
| `.github/ISSUE_TEMPLATE/bug_report.yml` | Adapt from batchgen; ask for GPU model, CUDA version, Docker image hash, `nvidia-smi` output, reproducible command. |
| `.github/ISSUE_TEMPLATE/feature_request.yml` | Adapt verbatim from batchgen (problem / proposed solution / alternatives / importance). |
| `Makefile` | New. Targets below. |
| `MANIFEST.in` | New. Globs below. |
| `requirements-lint.txt` | Extract from `requirements.txt`: `clang-format`, `pre-commit`. |
| `scripts/install_deps.sh` | Optional helper for non-Docker users; thin wrapper around `pip install -e .` with `CUDA_HOME` guards. **DEFER** to a follow-up; current `dev.sh` plus the Docker path already cover the canonical install. Note in PRD that we explicitly do NOT mirror batchgen's heavyweight `install_deps.sh` (no flash-attn/FlashMLA/DeepGEMM dependencies in this project). |

**`Makefile` target inventory:**

```
help          Show all targets
install       pip install -e . (no isolation, matches README current path)
install-dev   install + lint deps
test          docker build && docker run pytest tests -v       (matches CLAUDE.md §2)
docker-build  docker build -t device-emulator:latest .
docker-test   docker run --rm --gpus all device-emulator:latest python3 -m pytest tests -v
format        pre-commit run --all-files
clean         rm -rf build dist *.egg-info __pycache__ .pytest_cache .ruff_cache
```

**`MANIFEST.in` content (must include every artefact the sdist needs to build):**

```
include LICENSE README.md requirements.txt pyproject.toml CMakeLists.txt
include _setup_helpers.py
recursive-include csrc *.h *.hpp *.cpp *.cc *.cu *.cuh
recursive-include external *.h *.hpp *.cpp *.cc *.cu *.cuh
recursive-include cmake *.cmake CMakeLists.txt
recursive-include proto *.proto
recursive-include morphling *.py *.pyi *.proto py.typed
recursive-include config *.ini *.json *.yaml
recursive-include scripts *.py *.sh
prune build
prune build-verify-msg
prune .qa-vtime-qa
prune logs
prune __pycache__
global-exclude __pycache__ *.py[cod] *.so *.o *.a
```

**Rename:** `LICENSE.txt` → `LICENSE`. GitHub's auto-detection requires the
canonical name; the file contents stay identical.

**Acceptance for W1:**
- `python -m build --sdist` produces a tarball that contains every file under
  `csrc/`, `external/`, `proto/`.
- `gh repo view --json licenseInfo` (or equivalent UI) shows Apache-2.0.
- Opening a PR draft shows the new template.
- `git ls-files | grep LICENSE.txt` returns empty.

---

### W1.5 — Paper metadata (new in v2; EdgeSys '26 specific)

**Goal:** Make the repo recognisable as the EdgeSys '26 paper's companion
code from the first scroll, without committing to figure reproduction yet.

**Files to add:**

| Path | Content |
|------|---------|
| `CITATION.cff` | Citation File Format v1.2.0 — enables GitHub's "Cite this repository" button. Fields: `title`, `authors` (placeholder list, owner fills), `version`, `date-released`, `repository-code`, `license: Apache-2.0`, plus a `preferred-citation` block of `type: conference-paper` with `conference: name: "EdgeSys '26"`. Placeholders use `TODO(owner)` markers so a grep finds them. |
| `docs/paper.md` | Stub with: paper title (TBD), authors (TBD), abstract (TBD or paste from paper), link to camera-ready PDF (TBD), BibTeX block (TBD), and a **figure inventory table** with columns `Figure | Caption | Script | Expected output path | Reproduction notes` — owner fills the rows later. |

**README hooks (placed by W2):**

- New `## Paper` section near the top (between About and Key Innovations) with:
  - One-line paper title + "Accepted at **EdgeSys '26**"
  - "Authors: TODO(owner)"
  - Link: `[Paper PDF](TODO)` · `[arXiv](TODO)` · `[BibTeX](#citation)`
  - One-paragraph plain-English summary of the paper's contribution (≈3–4
    sentences, owner can refine).
- New `## Citation` section at the bottom with a fenced BibTeX block whose
  fields are all `TODO(owner)`. References `CITATION.cff` for the structured
  version.
- New `## Figures` subsection inside `## Paper` (or below it) that says:
  > Figures referenced in the paper are reproduced by the scripts listed in
  > [`docs/paper.md`](docs/paper.md). Reproduction commands and expected
  > outputs are documented there.

  No claim-to-experiment table in this PRD — owner specified figures will be
  added later.

**Why a stub, not a full reproduction guide?** User explicitly said "reproduce
not needed, I will provide what figure to add later, just add reminder". W1.5
adds the *scaffolding* so the owner pastes content into named files instead of
hunting for the right place.

**Acceptance for W1.5:**
- `CITATION.cff` parses with `cffconvert --validate` (if cffconvert available;
  otherwise YAML-validate).
- GitHub's "Cite this repository" sidebar item appears after push.
- `docs/paper.md` exists with all required headings and `TODO(owner)` markers.
- `grep -rn "TODO(owner)" README.md CITATION.cff docs/paper.md` lists every
  owner-fillable field — used as a checklist when the paper goes camera-ready.

---

### W2 — README rewrite

**Goal:** Compact, scannable README that opens with "this is an EdgeSys '26
paper's code", mirrors batchgen's structure for everything else.

**Target outline (in order, hard cap ~180 lines — extra ~30 lines vs v1
because of Paper + Citation sections):**

```
1.  H1 + tagline + center-aligned badges
        - License: Apache 2.0
        - Build: ![Build](https://github.com/drunkcoding/DeviceEmulator/actions/workflows/build.yml/badge.svg)
        - Format: ![Format](.../format.yaml/badge.svg)
        - Paper: [![EdgeSys '26](https://img.shields.io/badge/EdgeSys-2026-blue)](TODO)
2.  ## About               (3–4 sentences: what it is, key innovation)
3.  ## Paper               (NEW — see W1.5; title, venue, authors, link, summary)
4.  ## Key Innovations     (bulleted, 3–5 items)
        - Per-GEMM CUDA green-context switching with trace-driven controller
        - Zero-copy scatter-gather buffers for inter-device data transfer
        - Worker pool (XtGemm / CPU MKL) with pluggable scheduling policies
        - Virtual + physical device emulation under one runtime
5.  ## Application Scenarios  (edge inference research, RAN co-location experiments)
6.  ## Hardware Requirements   (Linux + Docker + NVIDIA Container Toolkit; CUDA 12.x recommended)
7.  ## Installation
        - 1-liner: docker build -t device-emulator:latest .
        - 1-liner: docker run --rm --gpus all device-emulator:latest python3 -m pytest tests -v
        - Optional non-Docker path: link to docs/INSTALL.md (deferred — TBD)
8.  ## Quick Start          (3–5 lines max — morphling_cmd save / morphling_emulator)
9.  ## Documentation        (links table)
        - docs/paper.md             (NEW — figure list, BibTeX, abstract)
        - docs/DEV_README.md
        - docs/DOCKER.md
        - docs/green-context.md     (extracted from current README)
        - docs/troubleshooting.md   (extracted from current README)
        - tests/cpp/README.md
10. ## Citation             (NEW — BibTeX block + link to CITATION.cff)
11. ## Acknowledgements     (PyTorch, libevent, MKL, vendored muduo_base, batchgen-project for reference repo structure)
```

**Content moved out of README:**

| Current README section | Action |
|------|---------|
| "Paper evaluation pipeline (single-node RAN control + training)" (lines 115–208) — **LDPC pipeline** | **DELETE from README.** User confirmed LDPC traces are unrelated to EdgeSys '26. Scripts in `scripts/` (e.g., `run_paper_experiments.sh`, `eval_greenctx_training.py`) are NOT touched by this PRD — they remain on disk, untouched, for separate cleanup. |
| "Per-GEMM green context switching" (lines 242–339) | **MOVE verbatim** → `docs/green-context.md`. Still relevant runtime feature documentation. |
| "Troubleshooting" (lines 210–240) | **MOVE verbatim** → `docs/troubleshooting.md`. |
| "Physical Device Usage" inline bash (lines 53–113) | **MOVE** → `docs/deployment.md`. Keep a 1-line pointer in README. |

**Files newly referenced** must exist before the README links land — Edit order:
new docs first, then README.

**Risk of accidental information loss in the LDPC deletion:** any reference to
LDPC scripts elsewhere in the repo (e.g., from `tests/`, `docs/`, other
scripts) will become a dangling link if those callers expect the README to
explain them. Mitigation: before deleting, `grep -rn 'paper_data\.json\|ldpc_trace\|run_paper_experiments' --include='*.md' --include='*.py'` and confirm nothing user-facing depends on the README text. The scripts themselves are unaffected; only the README explanation is removed.

**Acceptance for W2:**
- `wc -l README.md` ≤ 180.
- Every link in README resolves to an existing file (`grep -oE '\[.*\]\(.*\)' README.md` audit).
- Extracted docs files exist and contain the content removed from README.
- LDPC section is gone from README; `grep -in 'ldpc\|paper_data' README.md` is empty.
- No information loss for *kept* sections: the diff is "move", not "delete",
  for green-context, troubleshooting, deployment.

---

### W3 — CI hardening

**Goal:** PRs must verify the canonical "rebuild image, run tests" workflow,
not just lint.

**Files to add:**

`/.github/workflows/build.yml`:

```yaml
name: Docker Build
on:
  push:
    branches: [main]
    paths-ignore: ['**.md', 'docs/**']
  pull_request:
    branches: ['**']
    paths-ignore: ['**.md', 'docs/**']
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
jobs:
  docker-build:
    runs-on: ubuntu-22.04
    timeout-minutes: 60
    steps:
      - uses: actions/checkout@v4
        with: { submodules: recursive }
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      - name: Build image (no test stage — runner is CPU-only)
        run: docker build -t device-emulator:ci .
```

**Note:** GitHub-hosted runners are CPU-only — we cannot run the GPU pytest
suite there. The build step alone catches Dockerfile breakage and most
compilation errors. GPU tests stay gated behind a self-hosted runner (defer
until/if we add one, per W3-future).

**Files to keep:**
- `.github/workflows/format.yaml` — unchanged.
- `.github/workflows/numerical_consistency.yml` — unchanged.

**Files explicitly NOT added (vs. batchgen):**
- `docker-release.yml` — no version-release plan (user said so).
- `publish-test.yml` — no PyPI plan.
- `release-test.yml` — no self-hosted GPU node yet.

**Acceptance for W3:**
- New `build.yml` shows up as a required check on PRs.
- `act -j docker-build` (or manual PR) builds successfully.
- README "Build" badge resolves green.

---

### W4 — Repo hygiene

**Goal:** Remove agent-tooling and one-off artefacts that leak into a public
push and don't belong to the project.

**Tracked-but-shouldn't-be (`git ls-files` audit results):**

| Path | Action | Reason |
|------|--------|--------|
| `package.json` | `git rm` | Only contains `"task-master-ai"` — agent tooling, not project dep. |
| `package-lock.json` (790 KB) | `git rm` | Lockfile for above; bloats clone. |
| `VTIME_DATA_INVENTORY.md` | Move to `docs/internal/vtime-data-inventory.md` OR delete | Internal investigation note; not user-facing. **Decision:** move under `docs/internal/` so context is preserved without polluting root. |
| `morphling.yml` (conda env export) | Move to `docs/morphling-conda-env.yml` and reference from `docs/troubleshooting.md` | Useful as alternative install hint; doesn't belong at root. |

**Working-dir cruft to verify is .gitignored (no action if already ignored):**

`build/`, `build-verify-msg/`, `.qa-vtime-qa/`, `logs/`, `__pycache__/`,
`.pytest_cache/`, `.ruff_cache/`, `nohup.out`, `.env.tmp`, `.codex`,
`.sisyphus/`, `.omo/`, `.taskmaster/` (this last one is a judgment call — see
note below).

**`.gitignore` audit:**
Add (or verify present): `.codex`, `.env.tmp`, `nohup.out`, `.qa-*/`,
`.sisyphus/`, `.omo/`, `__pycache__/`, `*.pyc`.

**`.taskmaster/` decision:** keep tracked. The user explicitly uses taskmaster
for the PRD→tasks workflow (CLAUDE.md §6). Treating it as project metadata
rather than agent tooling is the right call. Same reasoning for `.github/`.

**Acceptance for W4:**
- `git ls-files | grep -E '^(package(-lock)?\.json|VTIME_DATA_INVENTORY|morphling\.yml)$'` returns empty.
- New locations resolve (`docs/internal/`, `docs/morphling-conda-env.yml`).
- `git status` clean after a fresh `docker build` (i.e., no new untracked
  files leak in).

---

### W5 — Deferred / out of scope

Captured for visibility, NOT executed in this PRD:

- **Logo / `assets/`** — needs design input from owner.
- **PyPI publishing workflow** — no release plan.
- **Docker Hub publishing workflow** — no release plan.
- **Self-hosted GPU CI runner** — infra decision, not code.
- **`install_deps.sh` for non-Docker users** — `dev.sh` plus Docker covers it;
  add only if user demand appears.
- **Reproducibility scaffolding** (REPRODUCE.md, kick-the-tires script,
  claim-to-experiment table) — user opted out of AE-style deliverables. If a
  reviewer later requests reproduction guidance, the figure list in
  `docs/paper.md` is the natural place to expand.
- **LDPC pipeline cleanup** — scripts `run_paper_experiments.sh`,
  `eval_greenctx_training.py`, `aggregate_paper_results.py`,
  `validate_traces.py` plus `scripts/run_*ldpc*` family will outlive this
  PRD. They're not deleted here (out of scope), only un-documented in README.
  Track as a follow-up: "decide which paper-evaluation scripts to keep vs
  delete now that LDPC is severed from EdgeSys '26".
- **`figures/` directory audit** — currently tracks 3.4 MB of PDF/PNG outputs
  including LDPC-related figures (`deadline_compliance`, `decode_latency_cdf`,
  `violation_inefficiency_events`, `workload_vs_latency`,
  `timeline_decode_and_sm`, `fig_e2e_latency`, plus `figures/comparison/*` and
  `figures/evaluation/*`). Owner must decide which figures are EdgeSys '26
  paper figures (keep, list in `docs/paper.md`) and which are LDPC leftovers
  (delete from git). Tracked as open question §7.6.

## 5. Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| `MANIFEST.in` misses a file → sdist can't compile. | After W1, run `python -m build --sdist`, extract the tarball into a scratch dir, run `docker build` from there. Add globs until it succeeds. |
| Renaming `LICENSE.txt` → `LICENSE` breaks an external tool that grepped the old name. | `grep -rn LICENSE.txt` repo-wide before renaming; update any references. |
| README rewrite drops content users rely on. | All extracted sections move verbatim into `docs/*.md`. The Documentation index in README links every one. |
| `build.yml` CI fails because the Dockerfile assumes GPU at build time. | Current Dockerfile (`Dockerfile`, 76 lines) is CPU-buildable — verified by reading it before writing this PRD. If a GPU step sneaks in, mark it `RUN ... || true` for CI, or split into a separate stage. |
| `package.json` removal breaks someone's local taskmaster install. | It's tracked but not used by anything in `morphling/` or `scripts/`. The agent already has `task-master-ai` installed globally via npm. Confirmed: no `import` or `require` references in source. |

## 6. Delivery plan

1. **Convert this PRD into tasks via `taskmaster parse_prd`** (CLAUDE.md §6).
2. Execute task-by-task, **in order W1 → W1.5 → W2 → W3 → W4**:
   - W1 commit:   `feat(oss): add CONTRIBUTING, templates, Makefile, MANIFEST`
   - W1.5 commit: `docs(paper): add CITATION.cff and EdgeSys '26 paper stub`
   - W2 commit:   `docs(readme): rewrite for EdgeSys '26 release; drop LDPC pipeline`
   - W3 commit:   `ci: add Docker build workflow on PR`
   - W4 commit:   `chore(repo): remove agent-tooling files, relocate internal notes`
3. After each workstream, run the verification block from its Acceptance
   section. Report results back before the next workstream starts.
4. Final step: `docker build -t device-emulator:latest .` and `docker run
   --gpus all ... pytest tests -v` to prove no regression.
5. **Do NOT commit unless the user explicitly asks** (per system rules).

## 7. Open questions for the owner

1. CODEOWNERS default: confirm `@drunkcoding` is correct (it's the org of the
   existing `git remote -v`). Should anyone else be listed?
2. README badges: keep current repo URL (`drunkcoding/DeviceEmulator`) for badge
   URLs, or do you plan to move the repo to an organization before going public?
3. CONTRIBUTING merge policy: BatchGen uses "owner-only merge button". Keep
   that, or relax to "any maintainer merges after one approval"?
4. `VTIME_DATA_INVENTORY.md` — keep at `docs/internal/` or delete entirely? My
   bias: keep, since `.git` already has it forever.
5. Project tagline (one line). Suggested:
   *"GPU device & network emulator for distributed inference research, with
   per-GEMM CUDA green-context control."*
   Better wording?

---

*End of PRD.*
