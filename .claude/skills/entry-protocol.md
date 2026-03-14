# Entry Protocol

> **Trigger:** Run this at the start of every new session before modifying any code.
> Say "Run the entry protocol" to invoke.

This is a nine-phase read-only scan. Do not write, edit, or propose code changes until Phase 9 is complete.

---

## Phase 1 — Read CLAUDE.md

Read `CLAUDE.md` in full. Extract:

- Project name, purpose, and status (Section 1)
- Permitted and forbidden actions (Section 3)
- Active hooks (Section 7)
- Known sharp edges (Section 5)

Confirm: "CLAUDE.md read. Rules loaded."

---

## Phase 2 — Check Bootstrap Status

Check for `.bootstrap-complete`. If absent, warn:

> Bootstrap has not been run. Run `bash scripts/bootstrap.sh` before proceeding.

If present, read and summarize when bootstrap was run and for which project.

---

## Phase 3 — Validate Repository Structure

Run:

```bash
bash scripts/validate-structure.sh
```

Summarize failures and warnings. Do not proceed past Phase 3 if there are hard failures — surface them to the user first.

---

## Phase 4 — Inventory Source Files

List all files in `src/`. For each file, note:

- File name and approximate purpose (from name and a quick read)
- Whether a corresponding test file exists in `tests/`

---

## Phase 5 — Inventory Test Files

List all files in `tests/unit/` and `tests/integration/`. For each:

- Note what module it tests
- Flag any source files in `src/` with no test coverage

---

## Phase 6 — Read Documentation

Skim the following in order:

1. `docs/architecture.md` — understand the system design
2. `docs/adr/` — identify all recorded decisions
3. `docs/qa/qa-plan.md` — understand test strategy

Note any documentation that is incomplete or references unfilled placeholders.

---

## Phase 7 — Identify Unfilled Placeholders

Scan all `.md`, `.yml`, `.sh` files for remaining `{{PLACEHOLDER}}` tokens.
List any found. If none, confirm: "No unfilled placeholders."

---

## Phase 8 — Build Session Summary

Write a brief summary (5–10 lines) covering:

- What this project does
- Current implementation status
- Test coverage state
- Documentation state
- Any blockers or urgent issues

---

## Phase 9 — Propose Prioritized Work Items

Produce a numbered list of suggested work items in priority order. For each:

- One-sentence description
- Estimated effort: Low / Medium / High
- Which CLAUDE.md permission tier applies (allowed / requires approval / forbidden)

Do not begin any work. Present the list and wait for the user to select an item.

---

*Entry protocol complete. Await user instruction.*
