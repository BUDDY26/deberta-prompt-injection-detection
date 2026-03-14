# Refactor Playbook

> **Trigger:** "Refactor [component / file / function]"
> All refactoring follows a proposal-first workflow. No code is changed until a proposal is approved.

---

## Rule: Proposal Before Any Change

Before writing a single line of changed code:

1. Read the target code in full
2. Read its tests
3. Read any related section of `docs/architecture.md`
4. Write a proposal using the template below
5. Present the proposal and wait for explicit approval

This prevents wasted effort and ensures the user controls all structural decisions.

---

## Proposal Template

```
## Refactor Proposal: [component or function name]

### What
[One sentence describing what will be changed]

### Why
[One paragraph explaining the problem with the current code]

### How
[Step-by-step description of the planned changes — no code yet]

### Risks
- [Risk 1 — what could break]
- [Risk 2 — what needs re-testing]

### Scope
Files to be modified:
- [file 1]
- [file 2]

Files that must NOT be modified:
- [file — reason]

### Tests
[How the refactor will be validated — existing tests, new tests, or both]

### Reversibility
[How to undo this change if it introduces a regression]
```

---

## Execution Steps (post-approval only)

### Step 1 — Confirm the test suite passes before touching anything

Run the test command from CLAUDE.md Section 3. If it fails, surface the failures to the user before proceeding. Never start a refactor on a broken baseline.

### Step 2 — Make one change at a time

Do not batch unrelated refactors in a single pass. Each discrete change must be small enough to review in isolation and validated by re-running the test suite.

### Step 3 — Keep interfaces stable

Unless the proposal explicitly changes a public API or function signature, external interfaces must remain identical after the refactor.

### Step 4 — Update documentation

After the refactor, update any affected:

- Inline comments and docstrings
- `docs/architecture.md` if a component's design changed
- CLAUDE.md Section 5 if new sharp edges were discovered

### Step 5 — Report completion

Produce a brief summary:

- What changed and why
- Test results before → after
- Any follow-up tasks identified

---

## When to Stop and Start Over

Stop and write a new proposal if:

- The scope grows beyond what was approved
- A test failure requires structural changes not in the original plan
- A dependency needs to be added or removed
- A public API needs to change
