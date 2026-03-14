#!/usr/bin/env bash
# =============================================================================
# bootstrap.sh — Portfolio Base Template Setup Script
#
# Purpose:
#   Renames all {{PLACEHOLDER}} values throughout the repository to real project
#   values. Run once immediately after cloning from the template.
#
# Usage:
#   bash scripts/bootstrap.sh
#
# What it does:
#   1. Prompts for project-specific values
#   2. Replaces all {{PLACEHOLDER}} tokens in all text files
#   3. Validates the resulting structure
#   4. Writes the bootstrap completion record
#   5. Prints a checklist of manual steps remaining
# =============================================================================

set -euo pipefail

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
RESET='\033[0m'

info()    { echo -e "${BLUE}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*"; exit 1; }
header()  { echo -e "\n${BOLD}$*${RESET}"; echo "$(printf '─%.0s' {1..60})"; }

# ── Guard: must be run from repo root ─────────────────────────────────────────
if [[ ! -f "CLAUDE.md" ]]; then
  error "Run this script from the repository root (where CLAUDE.md lives)."
fi

# ── Guard: must not have already been bootstrapped ────────────────────────────
if ! grep -q "deberta-prompt-injection-detection" CLAUDE.md 2>/dev/null; then
  warn "This repository appears to have already been bootstrapped."
  read -rp "Run again anyway? This will overwrite previous replacements. (y/N): " confirm
  [[ "${confirm,,}" == "y" ]] || { info "Aborted."; exit 0; }
fi

# ── Collect values ────────────────────────────────────────────────────────────
header "Portfolio Base — Bootstrap Setup"
echo "Answer the following questions. Press Enter to accept defaults."
echo ""

read -rp "  Project name (e.g. my-api-service):         " PROJECT_NAME
read -rp "  One-sentence description:                   " PROJECT_DESCRIPTION
read -rp "  GitHub username:                            " GITHUB_USERNAME
read -rp "  Repo name (default: $PROJECT_NAME):         " REPO_NAME
REPO_NAME="${REPO_NAME:-$PROJECT_NAME}"

echo ""
read -rp "  Primary language (e.g. Python 3.11):        " LANGUAGE
read -rp "  Framework (e.g. FastAPI, or 'None'):        " FRAMEWORK
read -rp "  Database (e.g. PostgreSQL, or 'None'):      " DATABASE
read -rp "  Test framework (e.g. pytest):               " TEST_FRAMEWORK

echo ""
read -rp "  Install command (e.g. pip install -r requirements.txt): " INSTALL_COMMAND
read -rp "  Run command    (e.g. python src/main.py):               " RUN_COMMAND
read -rp "  Test command   (e.g. pytest tests/ -v):                 " TEST_COMMAND
read -rp "  Lint command   (e.g. ruff check src/ && black src/):    " LINT_COMMAND

echo ""
read -rp "  License (e.g. MIT):                         " LICENSE
LICENSE="${LICENSE:-MIT}"

STATUS="Active Development"
TODAY=$(date +%Y-%m-%d)

# ── Build a safe sed-compatible replacement function ─────────────────────────
replace_in_file() {
  local file="$1"
  local placeholder="$2"
  local value="$3"
  # Escape special characters for sed
  local escaped_value
  escaped_value=$(printf '%s\n' "$value" | sed 's/[[\.*^$()+?{|]/\\&/g; s/]/\\]/g')
  sed -i "s|{{${placeholder}}}|${escaped_value}|g" "$file"
}

replace_all() {
  local placeholder="$1"
  local value="$2"
  info "Replacing {{${placeholder}}} → ${value}"
  # Find all text files tracked or untracked (excluding .git)
  while IFS= read -r -d '' file; do
    if file "$file" | grep -q "text"; then
      replace_in_file "$file" "$placeholder" "$value"
    fi
  done < <(find . -not -path "./.git/*" -type f -print0)
}

# ── Execute replacements ──────────────────────────────────────────────────────
header "Replacing placeholder values"

replace_all "PROJECT_NAME"       "$PROJECT_NAME"
replace_all "PROJECT_DESCRIPTION" "$PROJECT_DESCRIPTION"
replace_all "GITHUB_USERNAME"    "$GITHUB_USERNAME"
replace_all "REPO_NAME"          "$REPO_NAME"
replace_all "LANGUAGE"           "$LANGUAGE"
replace_all "FRAMEWORK"          "$FRAMEWORK"
replace_all "DATABASE"           "$DATABASE"
replace_all "TEST_FRAMEWORK"     "$TEST_FRAMEWORK"
replace_all "INSTALL_COMMAND"    "$INSTALL_COMMAND"
replace_all "RUN_COMMAND"        "$RUN_COMMAND"
replace_all "TEST_COMMAND"       "$TEST_COMMAND"
replace_all "LINT_COMMAND"       "$LINT_COMMAND"
replace_all "LICENSE"            "$LICENSE"
replace_all "STATUS"             "$STATUS"
replace_all "LAST_UPDATED"       "$TODAY"

# Remaining placeholders that need manual completion
replace_all "ENTRY_POINT_1"      "src/main.py — replace with actual entry point"
replace_all "ENTRY_POINT_2"      "— replace with secondary entry point or remove"
replace_all "CONFIG_FILE_2"      "— replace with config file or remove"
replace_all "SHARP_EDGE_1"       "— fill in after running entry protocol"
replace_all "SHARP_EDGE_2"       "— fill in after running entry protocol"
replace_all "DEMONSTRATES"       "— fill in: what technical skills does this project show?"
replace_all "ARCHITECTURE_SUMMARY" "— fill in: 3–5 sentence architecture description"
replace_all "PREREQ_1"           "— list prerequisites here"
replace_all "PREREQ_2"           "— list prerequisites here"
replace_all "FEATURE_1"          "— describe a key feature"
replace_all "FEATURE_2"          "— describe a key feature"
replace_all "FEATURE_3"          "— describe a key feature"
replace_all "PROJECT_OVERVIEW"   "— write 2–3 paragraph overview for this project"
replace_all "REPO_TREE"          "— run: tree -L 3 --gitignore"

success "All placeholder replacements complete."

# ── Validate structure ────────────────────────────────────────────────────────
header "Validating repository structure"
bash scripts/validate-structure.sh

# ── Write bootstrap completion record ────────────────────────────────────────
# Write a record that bootstrap was completed
echo "bootstrapped_on: $TODAY" >> .bootstrap-complete
echo "project: $PROJECT_NAME" >> .bootstrap-complete
echo "github_username: $GITHUB_USERNAME" >> .bootstrap-complete

# ── Print remaining manual steps ─────────────────────────────────────────────
header "Bootstrap complete ✓"
echo ""
echo -e "${BOLD}Automated replacements are done. Complete these manual steps:${RESET}"
echo ""
echo "  1. Fill in the architecture summary in CLAUDE.md Section 4"
echo "  2. Add project-specific sharp edges to CLAUDE.md Section 5"
echo "  3. Replace placeholder features in README.md"
echo "  4. Fill in docs/architecture.md with your system design"
echo "  5. Create at least one ADR in docs/adr/ (copy ADR-001-template.md)"
echo "  6. Add your actual source files to src/"
echo "  7. Add your actual test files to tests/"
echo "  8. Add required environment variables to .env.example"
echo "  9. Commit and push your bootstrap changes:"
echo ""
echo -e "     ${BLUE}git add .${RESET}"
echo -e "     ${BLUE}git commit -m 'chore: initialize from portfolio-base template'${RESET}"
echo -e "     ${BLUE}git push${RESET}"
echo ""
echo " 10. In your first Claude Code session, say:"
echo -e "     ${BOLD}\"Run the entry protocol\"${RESET}"
echo ""
success "Repository '${PROJECT_NAME}' is ready."
