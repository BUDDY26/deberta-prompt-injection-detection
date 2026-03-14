# Documentation Skill

> **Trigger:** "Document [file / module / project]" or "Update the docs"
> Covers docstrings, README sections, architecture docs, and ADRs.

---

## Docstring Standards

### Python

Use Google-style docstrings for all public functions, classes, and modules.

```python
def function_name(param: type) -> return_type:
    """One-sentence summary.

    Args:
        param: Description of the parameter.

    Returns:
        Description of the return value.

    Raises:
        ValueError: When and why this is raised.
    """
```

### TypeScript / JavaScript

Use JSDoc for all exported functions and classes.

```typescript
/**
 * One-sentence summary.
 *
 * @param param - Description of the parameter.
 * @returns Description of the return value.
 * @throws {Error} When and why this is thrown.
 */
```

---

## README Updates

When updating `README.md`, verify:

- The Overview section matches current functionality
- All Getting Started commands work from a clean environment
- The Repository Structure tree reflects the actual directory layout
- Removed features are removed from the Features list
- Prerequisites list current versions and tools

---

## Architecture Documentation

When updating `docs/architecture.md`, maintain these sections:

1. **System Overview** — what the system does, in one paragraph
2. **Component Map** — what each major component owns and is responsible for
3. **Data Flow** — how data moves through the system end-to-end
4. **Key Interfaces** — the contracts between components
5. **External Dependencies** — services, libraries, infrastructure

Do not let `docs/architecture.md` drift from the actual implementation. Update it in the same session as any significant structural change.

---

## Architecture Decision Records (ADRs)

Use ADRs to document significant decisions — library choices, algorithmic approaches, design tradeoffs.

To create a new ADR:

1. Copy `docs/adr/ADR-001-template.md`
2. Rename to `ADR-NNN-short-title.md` (increment NNN sequentially)
3. Fill in all sections
4. Link the new ADR from `docs/architecture.md`

An ADR is required when:

- A library or framework is added or removed
- An algorithm or approach is chosen over alternatives
- A significant structural decision is made
- A past decision is reversed or superseded

---

## Documentation Review Checklist

Before marking documentation complete:

- [ ] All public functions and classes have docstrings
- [ ] README commands tested and working from a clean environment
- [ ] `docs/architecture.md` reflects the current implementation
- [ ] New decisions have ADRs
- [ ] No unfilled `{{PLACEHOLDER}}` tokens remain in any file
- [ ] Internal links between docs resolve to real files
