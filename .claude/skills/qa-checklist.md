# QA Checklist

> **Trigger:** "Run QA" or "Audit the tests"
> Produces a test coverage and portfolio readiness report.

---

## Part 1 — Test Coverage Audit

### 1.1 Inventory

List every file in `src/`. For each, identify:

- Whether a unit test exists in `tests/unit/`
- Whether integration test coverage exists in `tests/integration/`
- Whether the file is tested at all

Flag any source file with no coverage as **UNCOVERED**.

### 1.2 Test Quality

For each test file, check:

| Check | Pass? |
|-------|-------|
| Tests assert specific outcomes, not just absence of errors | |
| Tests are independent — no shared mutable state between test cases | |
| Edge cases are covered: empty input, boundary values, error paths | |
| Test names describe the behavior being verified | |
| Fixtures are minimal and purposeful | |

### 1.3 Coverage Report

Run the test command from CLAUDE.md Section 3 and capture coverage output. Report:

- Total coverage percentage
- Files below 80% (flag as **LOW COVERAGE**)
- Files at 0% (flag as **UNCOVERED**)

---

## Part 2 — Documentation Audit

| Item | Status |
|------|--------|
| `README.md` overview matches current implementation | |
| All README commands tested and working | |
| `docs/architecture.md` reflects current code structure | |
| At least one ADR exists in `docs/adr/` | |
| `docs/qa/qa-plan.md` is filled in | |
| `docs/runbooks/operations.md` is filled in | |
| No unfilled template tokens in any operational file | |

---

## Part 3 — Portfolio Readiness Audit

| Criterion | Status |
|-----------|--------|
| Project has a clear purpose stated in README | |
| README explains what problem is solved | |
| Architecture is documented | |
| Tests exist and pass | |
| CI pipeline is green | |
| No placeholder tokens remain | |
| No dead code or commented-out blocks | |
| Commit history is professional (descriptive messages, logical progression) | |
| No secrets or credentials in any file | |
| CLAUDE.md Sections 4–9 are filled in | |

---

## Part 4 — Structure Validation

Run:

```bash
bash scripts/validate-structure.sh --strict
```

All items must pass in portfolio-ready state. Warnings are not acceptable for a submitted portfolio repository.

---

## QA Report Output Format

```
## QA Report: [project name] — [date]

### Test Coverage
- Total: [X]%
- Uncovered files: [list or "None"]
- Low-coverage files: [list or "None"]

### Documentation
[Pass / list of gaps]

### Portfolio Readiness
[Pass / list of blockers]

### Structure Validation
[Pass / FAIL with details]

### Recommended Actions
1. [Highest priority item]
2. [Next item]
```
