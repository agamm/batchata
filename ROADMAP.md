- [x] Citation feature via anthropic

- [x] Add github action tests before merge.

- [x] Raw results in dir.

- [x] Add pricing metadata and max_spend

- [x] Auto batch maanger (run several batches in parallel, retry failed ones, control spend, resume batch from state file)

- [ ] Better errors (citation not supported on image?)
- [ ] Get description of pydantic from field description to help llm.
- [ ] Abort batch option (also when ctrl+c)
- [ ] Test dry-mode to run on 1% from batch to check if works before big one
- [ ] OpenAI support
- [ ] Test web use 
- [ ] Eval example / embedding / 


---
- Fix batch_files to work with text/other files that provider supports not just pdf.