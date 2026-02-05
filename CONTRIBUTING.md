# Contributing to Nanovec

Thank you for your interest in contributing to Nanovec! We welcome contributions from everyone.

## How to Contribute

1. **Fork the repository** on GitHub.
2. **Clone your fork** locally.
3. **Create a new branch** for your feature or bug fix (`git checkout -b feature/amazing-feature`).
4. **Commit your changes** with clear messages.
5. **Push to your fork** (`git push origin feature/amazing-feature`).
6. **Open a Pull Request** to the `main` branch.

## Coding Guidelines

* **Go Style:** We follow standard Go conventions. Please run `go fmt ./...` before committing.
* **Testing:** New features must include unit tests. Ensure `go test -race ./...` passes.
* **Performance:** Nanovec prioritizes performance. If you change critical paths (HNSW, Flat Index), please include benchmark results in your PR description.

## Reporting Bugs

Please use the GitHub Issue Tracker to report bugs. Include:

* Your OS and Go version.
* A minimal reproduction code snippet.
* Expected vs. actual behavior.
