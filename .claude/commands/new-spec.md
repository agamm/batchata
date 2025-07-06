---
description: Create a new specification file for a feature or bugfix
---

# New Spec: $ARGUMENTS

Create a new specification file in the `specs/` directory and update CLAUDE.md to reference it as the current spec.

## Template

```markdown
# $ARGUMENTS

## Goal
Brief description of what we're trying to achieve.

## Current state
Description of what exists now in the codebase.

## Requirements
- [ ] Requirement 1
- [ ] Requirement 2
- [ ] Requirement 3

## Progress
### Completed
- None yet

### In Progress
- Starting spec

### Next Steps
1. Write tests for the main functionality
2. Implement the feature
3. Test and refactor

## Tests
### Tests to write
- [ ] Test case 1
- [ ] Test case 2

### Tests passing
- None yet

## Notes
Additional context, decisions, or considerations.
```

Please:
1. Create a new spec file at `specs/$ARGUMENTS.md` with the above template
2. Update CLAUDE.md to set the current spec to: `$ARGUMENTS`