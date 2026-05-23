## Description
Briefly describe your changes.

## Motivation
Explain why this change is needed and what problem it solves.
If it fixes an issue, link it (e.g., `closes #123`).

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Refactor (no behavior change)
- [ ] Breaking change
- [ ] Documentation update

## Checklist
- [ ] I have read the [CONTRIBUTING](../CONTRIBUTING.md) guide.
- [ ] My commit messages follow the
      [Angular Commit Format](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#-commit-message-format).
- [ ] I rebuilt the Docker image after my code change
      (`docker build -t device-emulator:latest .` or `make docker-build`).
- [ ] I ran the test suite in Docker
      (`make docker-test` or
      `docker run --rm --gpus all device-emulator:latest python3 -m pytest tests -v`).
- [ ] I updated the tests when behavior changed.
- [ ] I updated documentation under `docs/` when public APIs or workflows
      changed.
- [ ] I did NOT modify `proto/*.proto` without prior discussion
      (treat the wire format as a public contract).

## Reproduction / Verification
Paste the commands and output that verify your change.
