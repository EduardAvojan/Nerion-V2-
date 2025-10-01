# Development Setup

This guide will help you set up a development environment for Nerion V2.

## 1. Install Dependencies

To install all the dependencies required for development, including the tools for testing and development, run the following command:

```bash
pip install -e ".[dev,voice,web,docs]"
```

## 2. Running the Tests

To run the tests, use the following command:

```bash
pytest
```

This will run all the tests in the `tests` directory.

## 3. Code Style

We use `ruff` for linting and code formatting. To check your code for style issues, run the following command:

```bash
ruff check .
```

To automatically fix style issues, run:

```bash
ruff check . --fix
```
