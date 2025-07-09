# Add tests in github workflow

## Goal
Set up automated testing in GitHub Actions to run tests on every push and pull request, with branch protection to prevent merging to main or creating releases without passing tests.

## Current state
- Project has a comprehensive test suite using pytest
- Tests exist in `tests/` directory including unit tests and e2e tests
- Current GitHub workflow only handles PyPI publishing on release
- No automated test runs in CI/CD pipeline
- No branch protection rules enforced

## Requirements
- [x] Create GitHub workflow that runs on push and pull requests
- [x] Execute all tests using `uv run pytest`
- [x] Test on Python 3.12 only (project requires >=3.12)
- [x] Fail the build if any tests fail
- [x] Cache dependencies for faster runs
- [x] Run on Ubuntu latest
- [ ] Set up branch protection rules for main branch
- [ ] Require tests to pass before allowing merge to main
- [ ] Require tests to pass before allowing version/release tags

## Progress
### Completed
- Created SPEC.md
- Reviewed existing test structure and commands
- Created `.github/workflows/test.yml`
- Fixed raw response saving bug (dict.model_dump() AttributeError)
- Created DEVELOPMENT.md with test and release instructions

### In Progress
- None

### Completed Tasks
- Created GitHub workflow for automated testing
- Fixed test failures related to raw response saving
- Updated citations parsing logic
- Created DEVELOPMENT.md documentation

### Next Steps
1. Review existing tests and test commands
2. Create `.github/workflows/test.yml`
3. Test the workflow with a push
4. Set up branch protection rules in GitHub settings

## Tests
### Tests to write
- [ ] No new tests needed - workflow will run existing test suite

### Tests passing
- Need to verify all existing tests pass locally first

## URL References
- [GitHub Actions Python testing](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python) - Official guide for Python testing in GitHub Actions
- [uv GitHub Actions](https://docs.astral.sh/uv/guides/integration/github/) - Integration guide for using uv in GitHub Actions
- [Branch protection rules](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches) - GitHub branch protection documentation

## Learnings
- Found and fixed a bug in batch_job.py where raw_response was treated as a Pydantic model when it's actually a dict
- E2E tests may timeout in CI due to real API calls - consider adding test timeouts or mocking for CI
- Fixed citations parsing logic to handle single content blocks, but discovered API returns citations=None in batch mode
- The citations e2e test failure appears to be an API limitation where citations aren't returned in batch requests

## Notes
- Using `uv` as the package manager (consistent with existing publish workflow)
- Testing on Python 3.12 (project requires >=3.12 per pyproject.toml)
- Workflow should be similar to publish.yml but focused on testing
- Branch protection rules need to be configured in GitHub repository settings after workflow is created
- Consider making the test workflow a required status check for PRs
- **IMPORTANT**: Tests use real API calls, so `ANTHROPIC_API_KEY` must be added as a GitHub secret