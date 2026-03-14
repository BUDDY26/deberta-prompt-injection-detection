# CLAUDE.md — Repository Memory File

> **READ THIS FIRST.** This is your operating guide for this repository.
> Do not modify any code, rename any files, or restructure any directories
> until you have completed the Repository Entry Protocol in
> `.claude/skills/entry-protocol.md`.

---

## 1. Project Identity

**Project Name:** `{{PROJECT_NAME}}`
**Purpose (WHY):** `{{PROJECT_DESCRIPTION}}`
**Status:** `{{STATUS}}`  <!-- Active Development | Maintenance | Portfolio | Archived -->
**Primary Language(s):** `{{LANGUAGE}}`
**Framework(s):** `{{FRAMEWORK}}`
**Owner / Portfolio:** `{{GITHUB_USERNAME}}`

---

## 2. Repository Map (WHAT)

```
{{REPO_TREE}}
```

<!-- Run `tree -L 3 --gitignore` and paste the output above after first scan -->

**Key Entry Points:**
- `{{ENTRY_POINT_1}}`
- `{{ENTRY_POINT_2}}`

**Configuration Files:**
- `.env.example` — environment variable reference (never commit `.env`)
- `{{CONFIG_FILE_2}}`

**Test Suite:**
- `tests/` — {{TEST_FRAMEWORK}}, run with `{{TEST_COMMAND}}`

---

## 3. Rules + Commands (HOW)

### ✅ Allowed Without Asking
- Read any file
- Improve documentation (docstrings, comments, README, CLAUDE.md)
- Fix formatting and style inconsistencies
- Add or improve inline comments
- Add new test files in `tests/`
- Update `.env.example` with new variable names (never values)

### ⚠️ Requires Explicit Approval Before Executing
- Renaming or moving any file or directory
- Changing function signatures or public APIs
- Adding, removing, or upgrading dependencies
- Modifying database schemas or migration files
- Editing files in `src/auth/`, `src/billing/`, or `infra/`
- Deleting any file
- Creating new top-level directories

### 🚫 Never Do
- Commit or push to any branch
- Execute `rm -rf` or any irreversible destructive command
- Modify `.env` files or embed secrets in source code
- Run `DROP TABLE`, truncate databases, or execute destructive SQL
- Merge branches or create releases

### Common Commands
```bash
# Install dependencies
{{INSTALL_COMMAND}}

# Run the application
{{RUN_COMMAND}}

# Run tests
{{TEST_COMMAND}}

# Run linter + formatter
{{LINT_COMMAND}}

# Validate repository structure
bash scripts/validate-structure.sh
```

---

## 4. Repository Governance Rules

Documentation is the source of truth for this repository. Code follows documentation — never the reverse.

### Authority Hierarchy

```
Paper / External Sources
         ↓
   Evidence Ledger
         ↓
 Architecture Document
         ↓
    ADR Decisions
         ↓
 Implementation Plan
         ↓
        Code
```

Each layer is authoritative over everything below it. If code and documentation disagree, documentation wins and the code must be corrected — or an explicit change request must be approved before documentation is updated.

### Layer Responsibilities

| Layer | Location | Role |
|-------|----------|------|
| External sources | Research papers, specs, reports | Primary evidence; facts extracted here are non-negotiable |
| Evidence ledger | `docs/evidence.md` *(if applicable)* | Confirmed facts extracted from external sources; separates evidence from assumptions |
| Architecture | `docs/architecture.md` | System design, component map, data flow |
| ADRs | `docs/adr/*.md` | Binding architectural decisions with documented rationale |
| Implementation plan | `docs/implementation-plan.md` *(if applicable)* | Coding order, module scope, deliverables |
| Code | `src/` | Implementation — must conform to all layers above |

### Rules

- ADRs are binding once accepted. Do not re-litigate an accepted ADR without creating a superseding one.
- The implementation plan defines what gets built and in what order. Code must follow it.
- An AI assistant must not modify ADRs or the implementation plan automatically.

### Conflict Resolution Protocol

If a conflict is discovered — the plan cannot be followed exactly as written — the assistant must:

1. **Report** the conflict: what the plan specifies vs. what the implementation requires.
2. **Explain** why the current plan cannot be followed exactly.
3. **Propose** a specific, minimal change to the plan or ADR.
4. **Wait** for explicit approval before modifying any documentation.

Do not silently deviate from the plan. Do not edit governance documents without completing this protocol.

---

## 5. Implementation Plan Authority

`docs/implementation-plan.md` is the authoritative coding guide for this repository when present.

### Status During Coding Passes

The implementation plan is **read-only during coding passes**. It defines what to build and in what order. An AI assistant must not edit it while implementing code — not to mark progress, not to add notes, not to correct phrasing.

### Progress Reporting

Report implementation progress in responses rather than by editing the file:

> "Completed: `src/config.py`, `src/data.py`. Next: `src/env.py`."

### Conflict Protocol

If a true implementation conflict is discovered during a coding pass:

1. **Stop** the current coding pass.
2. **Report** the conflict clearly: plan specification vs. what the code requires.
3. **Propose** a minimal, targeted change to the plan.
4. **Wait** for explicit approval.

Once approved: update the plan first, then update the code to match.

---

## 6. Architecture Summary

`{{ARCHITECTURE_SUMMARY}}`

> Full system design, component breakdown, and data flow are documented in
> `docs/architecture.md`. Key technical decisions are in `docs/adr/`.

---

## 7. Known Issues / Sharp Edges

<!-- Fill this in after running the entry protocol. Examples below: -->
- `{{SHARP_EDGE_1}}` — e.g., `src/auth/` token logic is tightly coupled to legacy session model. Do not refactor without approval.
- `{{SHARP_EDGE_2}}` — e.g., `migrations/` must be applied in strict order. Never reorder.

---

## 8. Skills Available

| Skill | File | Purpose |
|-------|------|---------|
| Entry Protocol | `.claude/skills/entry-protocol.md` | **Run first** — mandatory scan before any changes |
| Code Review | `.claude/skills/code-review.md` | Structured review with severity-labeled findings |
| Refactor Playbook | `.claude/skills/refactor-playbook.md` | Safe, proposal-first refactoring workflow |
| Documentation | `.claude/skills/documentation.md` | Docstrings, README, architecture docs, ADRs |
| QA Checklist | `.claude/skills/qa-checklist.md` | Test coverage + portfolio readiness audit |
| Release Procedure | `.claude/skills/release-procedure.md` | Steps before tagging a version |

---

## 9. Hooks Active

| Hook | Trigger | Action |
|------|---------|--------|
| `post-edit-format` | After editing `.py` / `.ts` / `.js` files | Suggest running formatter |
| `pre-delete-guard` | Before any file deletion | Halt and require explicit confirmation |
| `test-on-core-change` | After editing files in `src/` | Remind to run test suite |
| `block-sensitive-dirs` | Before modifying `auth/`, `billing/`, `infra/`, `migrations/` | Halt and require approval |
| `no-secrets-in-code` | Before writing string literals resembling keys/tokens | Replace with env variable pattern |
| `proposal-before-refactor` | Before renaming, moving, or changing signatures | Write proposal first |

---

## 10. Documentation Index

| Document | Location | Description |
|----------|----------|-------------|
| Architecture Overview | `docs/architecture.md` | Full system design and component breakdown |
| ADR Index | `docs/adr/` | All architectural decision records |
| Implementation Plan | `docs/implementation-plan.md` | Coding order and module scope (create when applicable) |
| QA Plan | `docs/qa/qa-plan.md` | Test strategy and coverage map |
| Operations Runbook | `docs/runbooks/operations.md` | Setup, deployment, and troubleshooting |
| API Reference | `docs/api-reference.md` | Endpoint documentation (create when applicable) |

---

## 11. Portfolio Context

**Target Audience:** Graduate admissions reviewers (UT Austin MSCS), software engineering employers
**Demonstrates:** `{{DEMONSTRATES}}`  <!-- e.g., REST API design, async processing, TDD -->
**Key Technical Decisions:** See `docs/adr/` for documented rationale
**Portfolio Repository:** Yes — maintain professional commit history and documentation standards

---

*Last updated by Claude: `{{LAST_UPDATED}}`*
*Entry protocol completed: `no — run on first session`*
