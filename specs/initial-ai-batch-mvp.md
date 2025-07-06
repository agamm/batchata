# Initial ai-batch mvp

## Goal
Build a minimal batch() function that processes multiple messages through Anthropic's Message Batches API, using instructor for Pydantic model handling. Create a simple wrapper around instructor to enable batch processing with Claude models.

## Current state
- Basic Python project structure with hello.py
- pytest configured and working
- uv for dependency management
- Empty specs and tests directories

## Requirements
- [ ] Core batch() function that accepts messages, model, response_model, and system prompt
- [ ] Integration with instructor library for structured output
- [ ] Support for Anthropic Claude models
- [ ] Environment variable handling for API keys using python-dotenv
- [ ] Pydantic model validation for responses
- [ ] Error handling for API failures and invalid inputs
- [ ] Example spam detection implementation

## Progress
### Completed
- Project structure setup
- Basic testing framework

### In Progress
- Specification planning

### Next Steps
1. Set up project structure (ai_batch/, examples/)
2. Add dependencies: instructor, anthropic, python-dotenv
3. Write failing tests for batch() function
4. Implement minimal batch() wrapper
5. Create spam detection example
6. Add environment setup documentation

## Tests
### Tests to write
- [ ] test_batch_basic_functionality() - Test core batch processing
- [ ] test_batch_spam_detection() - Test with SpamResult model
- [ ] test_batch_error_handling() - Test invalid inputs and API errors
- [ ] test_batch_multiple_messages() - Test processing multiple messages
- [ ] test_pydantic_model_validation() - Test response model validation
- [ ] test_env_variable_loading() - Test API key loading from .env

### Tests passing
- None yet

## Notes
- API Example: batch(messages=[[{"role": "user", "content": f"Is this spam? {email}"}] for email in emails], model="claude-3-haiku-20240307", response_model=SpamResult, system="You are a spam detection expert.")
- Use instructor's existing batch processing capabilities
- Focus on simple, clean API that matches the provided example
- Ensure proper error handling and validation