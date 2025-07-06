# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Process
1. Check SPEC.md for current feature specification
2. Write tests for the main functionality you want to build
3. Run tests to confirm they fail appropriately
4. Implement the feature to make tests pass
5. Test other parts of codebase that might be related to ensure no regression
6. Update SPEC.md with progress
7. When feature is complete, move SPEC.md to specs/ for archival

## Current workflow
- Active spec: SPEC.md (next to this file)
- Use `/project:new-spec <feature-name>` to create a new spec
- When done, completed specs are archived in specs/ directory

## Archiving specs
When moving SPEC.md to specs/, update it with:
- **Learnings**: What insights were gained during implementation
- **Problems encountered**: Document challenges and how they were solved
- **Edge cases**: Note any unique situations or special handling required
- **Key decisions**: Why certain approaches were chosen
This preserves valuable context for future reference.

## General rules
- Less code is better
- Simple readable code is better
- Don't add comments, unless absolutely necessary to explain a "why" (e.g. if something is non-trivial and needs explanation)
- If there are several ways to go about something, ask the user before proceeding.
- If project structure or workflows change, ask user before updating this CLAUDE.md file

## Testing principles
- Tests should test the main idea/functionality, not be overly detailed
- Cover edge cases we discuss together, but don't overwhelm the codebase
- Tests must be REAL and actually test the functionality
- No fake/placeholder tests - every test must verify actual behavior
- Focus on testing the core behavior, not every possible scenario

## Project structure
```
ai-batch/
├── CLAUDE.md          # This file - Claude Code guidance
├── README.md          # Project documentation
├── pyproject.toml     # Python project configuration
├── hello.py           # Main application code
├── specs/             # Feature specifications
└── tests/             # Test files (pytest)
    ├── __init__.py
    └── test_*.py      # Test modules
```

## Development commands
### Package management
- Always use `uv` for Python package management
- `uv add <package>` - Add a production dependency
- `uv add --dev <package>` - Add a development dependency

### Testing
- `uv run pytest` - Run all tests
- `uv run pytest tests/test_specific.py` - Run specific test file
- `uv run pytest -v` - Run tests with verbose output
- `uv run pytest --cov` - Run tests with coverage (if coverage installed)

