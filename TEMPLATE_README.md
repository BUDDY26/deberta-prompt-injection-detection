# portfolio-base

> A reusable GitHub template repository for professional software engineering projects.
> Structured for graduate admissions review (UT Austin MSCS) and employer evaluation.

---

## What This Template Includes

Every repository created from this template gets:

| Category | What's Included |
|----------|----------------|
| **Claude Code** | `CLAUDE.md` (9-section memory file), 6 skills, 6 hooks |
| **CI/CD** | GitHub Actions pipeline (lint + test + security scan) |
| **Documentation** | Architecture doc, ADR template, QA plan, operations runbook |
| **Source structure** | `src/`, `tests/unit/`, `tests/integration/` with working starter files |
| **Python tooling** | `pyproject.toml`, `requirements.txt`, pytest + ruff + black configured |
| **TypeScript tooling** | `package.json`, `tsconfig.json`, jest + eslint + prettier configured |
| **Automation** | `bootstrap.sh` (renames all placeholders), `validate-structure.sh` |
| **GitHub** | PR template, bug report + feature request issue templates |

---

## Using This Template

### Option A — GitHub UI (recommended)

1. Click **Use this template** → **Create a new repository** at the top of this page
2. Name your repository and choose visibility
3. Clone your new repository locally
4. Run the bootstrap script:

```bash
cd your-new-repo
bash scripts/bootstrap.sh
```

The script will prompt for your project name, language, commands, and automatically replace all `{{PLACEHOLDER}}` tokens throughout every file.

### Option B — Manual Setup

If you prefer to set up manually without the script:

1. Clone or download this template
2. Find and replace all `{{PLACEHOLDER}}` tokens (see table below)
3. Delete any language-specific files you don't need
4. Run `bash scripts/validate-structure.sh` to confirm everything is in place

#### All Placeholders

| Placeholder | Replace With | Example |
|-------------|-------------|---------|
| `deberta-prompt-injection-detection` | Your project name | `my-api-service` |
| `Multi-stage fine-tuning pipeline for DeBERTa-v3 to detect prompt injection attacks using Safe-Guard, SPML, and NVIDIA Aegis datasets.` | One-sentence description | `A REST API for managing inventory` |
| `BUDDY26` | Your GitHub username | `jdoe` |
| `deberta-prompt-injection-detection` | Repository name | `inventory-api` |
| `Python 3.11` | Primary language + version | `Python 3.11` |
| `HuggingFace Transformers` | Framework (or `None`) | `FastAPI` |
| `None` | Database (or `None`) | `PostgreSQL` |
| `pytest` | Test framework | `pytest` |
| `pip install -r requirements.txt` | Dependency install command | `pip install -r requirements.txt` |
| `python src/train.py` | Application run command | `python src/main.py` |
| `pytest tests/ -v` | Test run command | `pytest tests/ -v` |
| `ruff check src/ {{LINT_COMMAND}}{{LINT_COMMAND}} black src/` | Lint/format command | `ruff check src/ && black src/` |
| `MIT` | License type | `MIT` |
| `2026-03-14` | Today's date | `2026-03-13` |
| `Active Development` | Project status | `Active Development` |
| `— fill in: what technical skills does this project show?` | Skills demonstrated | `REST API design, TDD, async processing` |
| `— fill in: 3–5 sentence architecture description` | 3–5 sentence architecture description | *(fill in after design)* |
| `— run: tree -L 3 --gitignore` | Output of `tree -L 3 --gitignore` | *(fill in after populating src/)* |

---

## Directory Structure

```
portfolio-base/
├── CLAUDE.md                         # Claude Code operating guide (9 sections)
├── README.md                         # Project README (with {{PLACEHOLDER}} tokens)
├── TEMPLATE_README.md                # This file — template documentation
├── CHANGELOG.md                      # Version history
├── .env.example                      # Environment variable reference
├── .gitignore                        # Python + Node + IDE + OS
│
├── pyproject.toml                    # Python: pytest, ruff, black, coverage config
├── requirements.txt                  # Python: runtime dependencies
├── package.json                      # TypeScript: scripts, dependencies
├── tsconfig.json                     # TypeScript: compiler options
├── jest.config.js                    # TypeScript: test runner config
├── .eslintrc.js                      # TypeScript: linter config
├── .prettierrc                       # TypeScript: formatter config
│
├── src/                              # Application source code
│   ├── CLAUDE.md                     # Module-level sharp-edge notes
│   ├── __init__.py                   # Python package init
│   ├── main.py                       # Python entry point (starter)
│   └── index.ts                     # TypeScript entry point (starter)
│
├── tests/
│   ├── conftest.py                   # pytest fixtures
│   ├── unit/
│   │   ├── test_main.py              # Python unit test starter
│   │   └── index.test.ts            # TypeScript unit test starter
│   └── integration/
│       ├── test_integration_example.py
│       └── integration.test.ts
│
├── docs/
│   ├── architecture.md               # System design template
│   ├── adr/
│   │   └── ADR-001-template.md       # Architecture Decision Record template
│   ├── qa/
│   │   └── qa-plan.md                # Test strategy template
│   └── runbooks/
│       └── operations.md             # Operational runbook template
│
├── scripts/
│   ├── bootstrap.sh                  # First-time setup + placeholder replacement
│   └── validate-structure.sh        # Structure conformance checker
│
├── .github/
│   ├── workflows/
│   │   └── ci.yml                    # Lint + test + security scan pipeline
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug-report.md
│   │   └── feature-request.md
│   ├── pull_request_template.md
│   └── dependabot.yml
│
└── .claude/
    ├── skills/
    │   ├── entry-protocol.md         # Mandatory 9-phase repo scan
    │   ├── code-review.md            # Structured code review
    │   ├── refactor-playbook.md      # Safe refactoring workflow
    │   ├── documentation.md          # Docstrings + docs generation
    │   ├── qa-checklist.md           # Quality + portfolio readiness audit
    │   └── release-procedure.md      # Pre-release checklist
    └── hooks/
        └── hooks.md                  # 6 automatic guardrails
```

---

## Claude Code Workflow

Once your repository is set up, open it in VS Code with Claude Code and say:

> **"Run the entry protocol"**

Claude will scan the repository, build a system summary, and propose a prioritized list of improvements — without touching any code until you approve.

### Available Skills

| Say this... | Claude will... |
|-------------|---------------|
| `"Run the entry protocol"` | Scan, summarize, and propose changes |
| `"Review this file"` | Run structured code review with severity ratings |
| `"Refactor this"` | Safe, proposal-first refactoring |
| `"Document this"` | Generate docstrings, README, architecture docs |
| `"Run QA"` | Audit tests, docs, and portfolio readiness |
| `"Prepare a release"` | Pre-release checklist and release report |

### Project Instructions (Paste into Claude Code)

Open **Claude Code → Project Settings → Custom Instructions** and paste:

```
You are a software engineering assistant maintaining a professional portfolio.
These repositories are reviewed by graduate admissions committees (UT Austin
MSCS) and software engineering employers.

MANDATORY RULES — apply every session:
1. Always read CLAUDE.md before doing anything else.
2. Run the entry protocol (.claude/skills/entry-protocol.md) before modifying code.
3. Respect the three-tier permission model in CLAUDE.md Section 3.
4. Use skills for repeated workflows — read the relevant .claude/skills/*.md first.
5. Respect all hooks in .claude/hooks/hooks.md.
6. Write documentation for a technical reader who has never seen this project.
7. Update CLAUDE.md at the end of every session with new findings.
```

---

## Applying to Existing Repositories

To bring an existing repo up to this standard:

1. Copy `.claude/` into the repo root
2. Copy `docs/` templates (don't overwrite existing docs)
3. Copy `scripts/bootstrap.sh` and `scripts/validate-structure.sh`
4. Create `CLAUDE.md` from the template and fill in sections 1–3
5. Run `bash scripts/validate-structure.sh` to see what's missing
6. In Claude Code: `"Run the entry protocol"` — it will handle the rest

---

## After Bootstrap: What To Do First

After running `bootstrap.sh`, complete these in order:

- [ ] Fill in `CLAUDE.md` sections 4–9 (architecture, sharp edges, portfolio context)
- [ ] Replace starter `src/main.py` / `src/index.ts` with your actual code
- [ ] Write real tests in `tests/unit/` and `tests/integration/`
- [ ] Fill in `docs/architecture.md` with your system design
- [ ] Create your first real ADR in `docs/adr/` (copy `ADR-001-template.md`)
- [ ] Add all required environment variables to `.env.example`
- [ ] Enable GitHub Actions in your repository settings
- [ ] Run `bash scripts/validate-structure.sh --strict` — it should pass clean

---

## How to Enable This as a GitHub Template

1. Push this repository to GitHub
2. Go to **Settings → General**
3. Check **Template repository**
4. Now the **"Use this template"** button appears on the repository page

---

*Built as part of a professional software engineering portfolio system.*
*Designed for UT Austin MSCS graduate admissions and employer review.*
