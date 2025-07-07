# CLAUDE.md

## Process
1. Check SPEC.md for current feature specification
2. Write tests for the main functionality
3. Implement the feature to make tests pass
4. Update SPEC.md with progress
5. When complete, move SPEC.md to specs/ for archival with a relevant name.

## General rules
- Less output is better, try to be succinct
- Less code is better
- Simple readable code is better
- No comments unless absolutely necessary
- Ask user before proceeding if multiple approaches exist

## Testing
- Test main functionality, not every detail
- Tests must be real and verify actual behavior
- Focus on core behavior

## Project structure
(Update when it changes)
```
ai-batch/
├── CLAUDE.md          # This file
├── SPEC.md            # Current feature specification
├── pyproject.toml     # Project configuration
├── ai_batch.py        # Main module with batch() function
├── .env.example       # Environment variable template
├── examples/          # Example usage scripts
│   ├── __init__.py
│   └── spam_detection.py
├── specs/             # Archived feature specifications
├── tests/             # Test files
│   ├── __init__.py
│   ├── test_ai_batch.py
│   ├── test_hello.py
│   └── e2e/
│       └── test_batch_integration.py
└── hello.py           # Legacy file
```

## Commands
- `uv add <package>` - Add dependency
- `uv run pytest` - Run tests
- `uv run python -m examples.spam_detection` - Run example

