# QA Plan

**Project:** {{PROJECT_NAME}}
**Test framework:** {{TEST_FRAMEWORK}}
**Last updated:** {{LAST_UPDATED}}

---

## Test Strategy

<!-- Describe the overall approach to testing for this project. -->
<!-- What confidence level is targeted? What is explicitly out of scope? -->

*(Fill in the test strategy here.)*

---

## Test Layers

### Unit Tests — `tests/unit/`

- **Scope:** Individual functions and classes in isolation
- **Mocking policy:** Mock external I/O (database, network, filesystem); do not mock internal domain logic
- **Coverage target:** 80% minimum per source file
- **Framework:** {{TEST_FRAMEWORK}}

### Integration Tests — `tests/integration/`

- **Scope:** Component interactions, database queries, full request/response cycles
- **Environment:** Requires real dependencies (database running, environment variables set)
- **When to run:** Before every merge; always in CI

---

## Test Command

```bash
{{TEST_COMMAND}}
```

---

## Coverage Policy

| Result | Threshold |
|--------|-----------|
| Pass | ≥ 80% overall |
| Warning | 60–79% |
| Fail | < 60% |

---

## Test File Inventory

<!-- Update this table as source files are added. -->

| Source File | Unit Test | Integration Test | Coverage |
|-------------|-----------|-----------------|----------|
| *(fill in)* | *(yes/no)* | *(yes/no)* | *(%)*  |

---

## CI Integration

Tests run automatically on every push and pull request via `.github/workflows/ci.yml`.
See the workflow file for the exact commands and matrix configuration.

---

## Known Gaps

<!-- Document any areas deliberately not covered by automated tests, and why. -->

*(Fill in known test gaps here.)*
