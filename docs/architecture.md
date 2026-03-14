# Architecture Overview

**Project:** {{PROJECT_NAME}}
**Last updated:** {{LAST_UPDATED}}

> Update this document whenever the system design changes.
> Keep it in sync with the actual implementation — do not let it drift.

---

## System Overview

{{ARCHITECTURE_SUMMARY}}

---

## Component Map

| Component | Location | Responsibility |
|-----------|----------|----------------|
| *(fill in)* | *(fill in)* | *(fill in)* |

<!-- List each major component, what directory it owns, and what it is responsible for. -->
<!-- Example: -->
<!-- | API layer | `src/api/` | HTTP request handling and routing | -->
<!-- | Domain logic | `src/core/` | Business rules, independent of transport layer | -->
<!-- | Data access | `src/db/` | Database queries and connection management | -->

---

## Data Flow

<!-- Describe how data moves through the system from input to output. -->
<!-- Use a numbered list or a simple diagram. -->

*(Document the end-to-end data flow here.)*

---

## Key Interfaces

<!-- List the contracts between major components. -->
<!-- These are the seams where changes in one component can propagate to another. -->

*(Document key interfaces, function signatures, or API contracts here.)*

---

## External Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| *(fill in)* | *(fill in)* | *(fill in)* |

---

## Architecture Decision Records

Key decisions are documented in [`docs/adr/`](adr/).

| ADR | Title | Status |
|-----|-------|--------|
| [ADR-001](adr/ADR-001-template.md) | *(first decision title)* | Proposed |

---

## Known Constraints

<!-- Document performance, scalability, or operational constraints. -->
<!-- These should also appear in CLAUDE.md Section 5 as sharp edges. -->

*(Document constraints here.)*
