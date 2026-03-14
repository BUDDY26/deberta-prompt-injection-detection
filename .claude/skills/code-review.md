# Code Review Skill

> **Trigger:** "Review [file / PR / change]"
> Produces a structured, severity-labeled review report.

---

## Severity Labels

| Label | Meaning |
|-------|---------|
| **BLOCKER** | Must fix before merge — correctness, security, or data integrity issue |
| **MAJOR** | Should fix — design flaw, significant performance problem, or confusing logic |
| **MINOR** | Nice to fix — style, naming, or small inefficiency |
| **NOTE** | Observation only — no action required |

---

## Review Procedure

### Step 1 — Understand the change

Before reviewing, read:

1. The file(s) being reviewed in full
2. Any related test files
3. The relevant section of `docs/architecture.md` if the change touches a core component

### Step 2 — Check correctness

- Does the code do what it claims to do?
- Are edge cases handled (empty input, null, zero, boundary values)?
- Are errors raised or returned at the right boundaries?
- Are there off-by-one errors, race conditions, or unhandled exceptions?

### Step 3 — Check security

- Is user input validated at system boundaries?
- Are secrets handled through environment variables, never hardcoded?
- Is there potential for SQL injection, XSS, or command injection?
- Does the code follow the principle of least privilege?

### Step 4 — Check tests

- Is there a test for the new behavior?
- Do tests assert outcomes, not just that code runs without error?
- Are edge cases tested?
- Are tests independent (no shared mutable state)?

### Step 5 — Check readability

- Are names descriptive and consistent with the surrounding codebase?
- Is the logic self-evident, or does it need a comment?
- Is each function doing one thing?
- Is there dead code or commented-out code to remove?

---

## Output Format

```
## Code Review: [filename or description]

### Summary
[2–3 sentence overall assessment]

### Findings

**[BLOCKER]** [location] — [description]
**[MAJOR]**   [location] — [description]
**[MINOR]**   [location] — [description]
**[NOTE]**    [location] — [description]

### Verdict
APPROVE | REQUEST CHANGES | DISCUSS
[One sentence justifying the verdict]
```

If there are no findings at a given severity level, omit that section.

---

## Quick Checklist

- [ ] Correctness verified
- [ ] Security checked
- [ ] Tests present and meaningful
- [ ] No hardcoded secrets
- [ ] No dead code
- [ ] Names are clear and consistent
- [ ] Errors handled at system boundaries
- [ ] CLAUDE.md permission rules respected
