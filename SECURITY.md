# Security Policy

## Supported versions

Morphling is a research artefact released alongside the EdgeSys '26 paper.
Security fixes are applied to the current `main` and `dev` branches only;
there are no long-term-support branches.

| Version | Supported          |
|---------|--------------------|
| `main`  | :white_check_mark: |
| `dev`   | :white_check_mark: |
| older   | :x:                |

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security problems. Use
GitHub's private Security Advisories instead:

  https://github.com/drunkcoding/DeviceEmulator/security/advisories/new

This routes the report to the maintainers privately and lets us
coordinate a fix before disclosure.

Include in the report:

- A description of the issue and its impact (data exfiltration, RCE,
  privilege escalation, denial of service, etc.).
- Steps to reproduce, ideally a minimal test case.
- Your environment (OS, CUDA driver, Docker image hash).

## Response process

- **Acknowledgement:** within 7 calendar days of the report.
- **Triage and fix:** target window depends on severity; we aim for a
  patch within 30 days for high-severity issues.
- **Disclosure:** a public advisory and patched release follow the fix.
  Reporters are credited unless they prefer to remain anonymous.

## Scope

In-scope: code under `morphling/`, `csrc/`, `scripts/`, `tests/`, and the
build/CI configuration.

Out of scope: vulnerabilities in vendored third-party code under
`external/` — please report those upstream. See
[`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md) for upstream
project links.
