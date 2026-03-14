#!/usr/bin/env bash
# =============================================================================
# validate-structure.sh — Repository Structure Validator
#
# Purpose:
#   Checks that the repository conforms to the portfolio-base template standard.
#   Prints a pass/fail report. Exits 0 if all required files exist, 1 if not.
#
# Usage:
#   bash scripts/validate-structure.sh
#   bash scripts/validate-structure.sh --strict   # fail on warnings too
# =============================================================================

set -euo pipefail

STRICT=false
[[ "${1:-}" == "--strict" ]] && STRICT=true

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
RESET='\033[0m'

pass()  { echo -e "  ${GREEN}✓${RESET}  $*"; }
fail()  { echo -e "  ${RED}✗${RESET}  $*"; FAILURES=$((FAILURES + 1)); }
warn()  { echo -e "  ${YELLOW}⚠${RESET}  $*"; WARNINGS=$((WARNINGS + 1)); }
header(){ echo -e "\n${BOLD}$*${RESET}"; echo "$(printf '─%.0s' {1..50})"; }

FAILURES=0
WARNINGS=0

# ── Required files ────────────────────────────────────────────────────────────
header "Required Files"

REQUIRED_FILES=(
  "CLAUDE.md"
  "README.md"
  ".gitignore"
  ".env.example"
  ".claude/skills/entry-protocol.md"
  ".claude/skills/code-review.md"
  ".claude/skills/refactor-playbook.md"
  ".claude/skills/documentation.md"
  ".claude/skills/qa-checklist.md"
  ".claude/hooks/hooks.md"
  "docs/architecture.md"
  "docs/adr/ADR-001-template.md"
  "docs/qa/qa-plan.md"
  "docs/runbooks/operations.md"
  "scripts/bootstrap.sh"
  "scripts/validate-structure.sh"
)

for f in "${REQUIRED_FILES[@]}"; do
  if [[ -f "$f" ]]; then
    pass "$f"
  else
    fail "$f  ← MISSING"
  fi
done

# ── Required directories ──────────────────────────────────────────────────────
header "Required Directories"

REQUIRED_DIRS=(
  "src"
  "tests"
  "tests/unit"
  "tests/integration"
  "docs"
  "docs/adr"
  "docs/qa"
  "docs/runbooks"
  "scripts"
  ".claude/skills"
  ".claude/hooks"
  ".github/workflows"
  ".github/ISSUE_TEMPLATE"
)

for d in "${REQUIRED_DIRS[@]}"; do
  if [[ -d "$d" ]]; then
    pass "$d/"
  else
    fail "$d/  ← MISSING"
  fi
done

# ── CI/CD checks ──────────────────────────────────────────────────────────────
header "CI/CD Files"

CI_FILES=(
  ".github/workflows/ci.yml"
  ".github/ISSUE_TEMPLATE/bug-report.md"
  ".github/ISSUE_TEMPLATE/feature-request.md"
  ".github/pull_request_template.md"
  ".github/dependabot.yml"
)

for f in "${CI_FILES[@]}"; do
  if [[ -f "$f" ]]; then
    pass "$f"
  else
    warn "$f  ← missing (recommended)"
  fi
done

# ── Placeholder check ─────────────────────────────────────────────────────────
header "Placeholder Replacement Check"

PLACEHOLDER_COUNT=$(grep -r "{{[A-Z_]*}}" --include="*.md" --include="*.yml" --include="*.yaml" --include="*.sh" . 2>/dev/null | grep -v ".git" | wc -l || true)

if [[ "$PLACEHOLDER_COUNT" -eq 0 ]]; then
  pass "No unfilled {{PLACEHOLDER}} tokens found"
else
  warn "$PLACEHOLDER_COUNT unfilled {{PLACEHOLDER}} tokens remain"
  echo ""
  grep -r "{{[A-Z_]*}}" --include="*.md" --include="*.yml" --include="*.yaml" --include="*.sh" . 2>/dev/null \
    | grep -v ".git" \
    | sed 's/^/      /' \
    | head -20
  if [[ "$PLACEHOLDER_COUNT" -gt 20 ]]; then
    echo "      ... and $((PLACEHOLDER_COUNT - 20)) more"
  fi
fi

# ── Bootstrap check ───────────────────────────────────────────────────────────
header "Bootstrap Status"

if [[ -f ".bootstrap-complete" ]]; then
  pass "Bootstrap has been run"
  cat .bootstrap-complete | sed 's/^/      /'
else
  warn "Bootstrap has not been run — run: bash scripts/bootstrap.sh"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "$(printf '─%.0s' {1..50})"
echo -e "${BOLD}Results:${RESET}"
echo -e "  Required files/dirs:  ${FAILURES} failure(s)"
echo -e "  Warnings:             ${WARNINGS} warning(s)"
echo ""

if [[ $FAILURES -gt 0 ]]; then
  echo -e "${RED}${BOLD}FAIL${RESET} — $FAILURES required item(s) are missing."
  echo "     Run 'bash scripts/bootstrap.sh' to resolve most issues."
  exit 1
elif [[ $WARNINGS -gt 0 && "$STRICT" == "true" ]]; then
  echo -e "${YELLOW}${BOLD}WARN${RESET} — $WARNINGS warning(s) in strict mode."
  exit 1
else
  echo -e "${GREEN}${BOLD}PASS${RESET} — Repository structure is valid."
  exit 0
fi
