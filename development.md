# Development Guide

## Running Tests

Tests require an Anthropic API key since they make real API calls.

```bash
# Install dependencies
uv sync --dev

# Set API key
export ANTHROPIC_API_KEY="your-api-key"

# Run all tests
uv run pytest

# Run tests with verbose output
uv run pytest -v

# Run a specific test file
uv run pytest tests/test_ai_batch.py

# Run a specific test
uv run pytest tests/test_ai_batch.py::test_batch_empty_messages
```

## Upgrading Version

1. Edit `pyproject.toml` and update the version number:
   ```toml
   version = "0.0.2"  # Change this
   ```

2. Commit the change:
   ```bash
   git add pyproject.toml
   git commit -m "Bump version to 0.0.2"
   ```

3. Create a release on GitHub:
   - Go to Releases → Create new release
   - Create a new tag matching the version (e.g., `v0.0.2`)
   - Publish the release
   - The GitHub Action will automatically publish to PyPI

## GitHub Secrets Setup

For tests to run in GitHub Actions, add your API key as a secret:
1. Go to Settings → Secrets and variables → Actions
2. Add new secret: `ANTHROPIC_API_KEY`