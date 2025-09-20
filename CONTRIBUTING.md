

# Contributing to Nerion Selfcoder

Thank you for your interest in contributing!  
This document outlines the basic steps and expectations when submitting contributions.

---

## 1. Fork and Clone
1. Fork the repository to your own GitHub account.
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/nerion-selfcoder.git
   cd nerion-selfcoder
   ```

---

## 2. Branch Naming
- Use short, descriptive branch names.
- Format:  
  ```
  feature/<short-description>
  fix/<short-description>
  docs/<short-description>
  ```
  Example:  
  ```
  feature/add-logging-config
  fix/docstring-bug
  ```

---

## 3. Run Tests & Health Checks Before PR
Before opening a pull request:
```bash
pytest -q
selfcoder healthcheck --verbose
```
Ensure **all tests pass** and **no health check failures** occur.

---

## 4. Pull Request Guidelines
- Write a clear title and description explaining **what** the PR changes and **why**.
- Reference related issues if applicable (e.g., `Closes #123`).
- Keep PRs focused on a single change whenever possible.

---

## 5. Coding Style
- Follow **PEP 8** style guidelines.
- Use `ruff` for linting:
  ```bash
  ruff .
  ```
- Write clear, concise docstrings for new functions, classes, and modules.
- Keep changes minimal and targeted.

---

## 6. Project Overview
Please read the [README.md](README.md) for a full project overview before contributing.

---

Thank you for helping improve Nerion Selfcoder!