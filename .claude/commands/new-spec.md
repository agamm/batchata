---
description: Create a new specification for a feature or bugfix
---

# New Spec: $ARGUMENTS

Create a new SPEC.md file (replacing any existing one) for the current feature. When the current feature is complete, the existing SPEC.md will be moved to specs/ for archival.

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

## URL References
Links to external resources, documentation, or codebases that help understand the requirements:
- [Example](https://example.com) - Description of what this helps with

## Learnings
Any ideas or edge cases and how you decided to go about them.

## Notes
Additional context, decisions, or considerations.
```

Please:
1. If SPEC.md exists, move it to `specs/` directory with a descriptive name
2. Create a new SPEC.md with the above template for `$ARGUMENTS`