- [x] Citation feature via anthropic

- [x] Add github action tests before merge.

- [x] Raw results in dir.

- [x] Add pricing metadata and max_spend

- [x] Auto batch maanger (run several batches in parallel, retry failed ones, control spend, resume batch from state file)

- [x] Better errors (citation not supported on image?)
- [x] Add github workflow to update model names in providers.
- [ ] Get description of pydantic from field description to help llm.
- [x] Abort batch option (also when ctrl+c)
- [ ] Test dry-mode to run on 1% from batch to check if works before big one
- [x] OpenAI support
- [ ] Test web use 
- [ ] Eval example / embedding / 
---
- Fix batch_files to work with text/other files that provider supports not just pdf.