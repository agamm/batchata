# Fix Packaging and Import Issues

## Problem
The batchata package on PyPI had import issues due to incorrect packaging structure. Users couldn't import `from batchata import BatchManager` because the package was incorrectly structured with a `src/` prefix.

## Root Cause
The `pyproject.toml` packaging configuration was set to:
```toml
[tool.hatch.build.targets.wheel]
packages = ["src"]
```

This created a package structure where the installed package had `src/...` instead of `batchata/...`, causing import failures.

## Solution
1. **Restructured source code**: Moved all files from `src/batchata/` to `batchata/` for cleaner packaging
2. **Updated packaging config**: Changed `pyproject.toml` to use `packages = ["batchata"]`
3. **Fixed all test imports**: Updated all test files to import from `batchata` instead of `src`
4. **Version bump**: Incremented to 0.2.3 to reflect the packaging fix

## Changes Made
- Moved source files from `src/batchata/` to `batchata/`
- Updated `pyproject.toml` packaging configuration
- Fixed 30+ test files with incorrect import statements
- Updated mock patches in tests to use correct module paths

## Verification
- All 131 tests pass (excluding e2e tests)
- Package builds correctly with `uv build`
- Imports work correctly: `from batchata import BatchManager`
- Created and tested with isolated environment to verify imports

## Status
✅ **COMPLETE** - Ready for PyPI release as version 0.2.3