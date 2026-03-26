#!/bin/bash
#
# verify_workflows.sh
# Comprehensive workflow verification and health check
#

echo "=========================================="
echo "LinkedIn Automation Workflow Verification"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counter for checks
CHECKS_PASSED=0
CHECKS_FAILED=0

# Helper functions
pass() {
    echo -e "${GREEN}✅${NC} $1"
    ((CHECKS_PASSED++))
}

fail() {
    echo -e "${RED}❌${NC} $1"
    ((CHECKS_FAILED++))
}

warn() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

info() {
    echo -e "${BLUE}ℹ️${NC} $1"
}

echo -e "${BLUE}=== 1. WORKFLOW FILE STRUCTURE ===${NC}\n"

# Check if workflows directory exists
if [ -d ".github/workflows" ]; then
    pass "Workflows directory exists"
else
    fail "Workflows directory not found"
    exit 1
fi

# Check workflow files
WORKFLOW_FILES=(
    ".github/workflows/linkedin-combined-posts.yml"
    ".github/workflows/linkedin_growth_automation.yml"
)

for wf in "${WORKFLOW_FILES[@]}"; do
    if [ -f "$wf" ]; then
        pass "Found $wf"
    else
        fail "Missing $wf"
    fi
done

echo ""
echo -e "${BLUE}=== 2. WORKFLOW SYNTAX VALIDATION ===${NC}\n"

for wf in "${WORKFLOW_FILES[@]}"; do
    if python3 -c "import yaml; yaml.safe_load(open('$wf'))" 2>/dev/null; then
        pass "Valid YAML syntax: $wf"
    else
        fail "Invalid YAML syntax: $wf"
    fi
done

echo ""
echo -e "${BLUE}=== 3. CRON SCHEDULE DEFINITIONS ===${NC}\n"

echo "linkedin-combined-posts.yml:"
grep -E "^\s*- cron:" .github/workflows/linkedin-combined-posts.yml | sed 's/^/  /'
info "Times (UTC): 9:00 AM, 12:00 PM, 6:00 PM | IST: 2:30 PM, 5:30 PM, 11:30 PM"

echo ""
echo "linkedin_growth_automation.yml:"
grep -E "^\s*- cron:" .github/workflows/linkedin_growth_automation.yml | head -3 | sed 's/^/  /'
info "Times (UTC): 03:30, 06:30, 11:30 | IST: 9:00 AM, 12:00 PM, 5:00 PM"

echo ""
echo -e "${BLUE}=== 4. TRIGGER CONFIGURATION ===${NC}\n"

if grep -q "workflow_dispatch:" .github/workflows/linkedin-combined-posts.yml; then
    pass "Manual trigger (workflow_dispatch) enabled in combined-posts"
else
    fail "Manual trigger not found in combined-posts"
fi

if grep -q "workflow_dispatch:" .github/workflows/linkedin_growth_automation.yml; then
    pass "Manual trigger (workflow_dispatch) enabled in growth-automation"
else
    fail "Manual trigger not found in growth-automation"
fi

echo ""
echo -e "${BLUE}=== 5. ENVIRONMENT VARIABLES & SECRETS ===${NC}\n"

if grep -q "LINKEDIN_ACCESS_TOKEN" .github/workflows/linkedin-combined-posts.yml; then
    pass "LINKEDIN_ACCESS_TOKEN referenced in combined-posts"
    warn "Verify secret is configured in GitHub repo settings"
else
    fail "LINKEDIN_ACCESS_TOKEN not found in combined-posts"
fi

if grep -q "LINKEDIN_MEMBER_ID" .github/workflows/linkedin-combined-posts.yml; then
    pass "LINKEDIN_MEMBER_ID referenced in combined-posts"
    warn "Verify secret is configured in GitHub repo settings"
else
    fail "LINKEDIN_MEMBER_ID not found in combined-posts"
fi

if grep -q "FORCE_FORMAT:" .github/workflows/linkedin-combined-posts.yml; then
    pass "FORCE_FORMAT configured in combined-posts"
    echo "  Formats:"
    grep "FORCE_FORMAT:" .github/workflows/linkedin-combined-posts.yml | awk '{print "    - " $NF}' | sort -u
else
    fail "FORCE_FORMAT not configured"
fi

echo ""
echo -e "${BLUE}=== 6. POST CONFIGURATION ===${NC}\n"

if grep -q "FORCE_FORMAT: hiring_pitch" .github/workflows/linkedin-combined-posts.yml; then
    pass "Hiring pitch format configured"
else
    fail "Hiring pitch format not configured"
fi

if grep -q "FORCE_FORMAT: referral_request" .github/workflows/linkedin-combined-posts.yml; then
    pass "Referral request format configured"
else
    fail "Referral request format not configured"
fi

if grep -q "DRY_RUN: false" .github/workflows/linkedin-combined-posts.yml; then
    pass "Live posting mode enabled (DRY_RUN: false)"
else
    warn "DRY_RUN might be enabled (check config)"
fi

echo ""
echo -e "${BLUE}=== 7. RATE LIMITING ===${NC}\n"

if grep -q "sleep 30" .github/workflows/linkedin-combined-posts.yml; then
    pass "30-second delay between posts configured"
else
    warn "No delay between posts found"
fi

echo ""
echo -e "${BLUE}=== 8. DEPENDENCY INSTALLATION ===${NC}\n"

if grep -q "pip install -r requirements.txt" .github/workflows/linkedin-combined-posts.yml; then
    pass "Python dependencies installed via requirements.txt"
else
    fail "Dependency installation not found"
fi

echo ""
echo -e "${BLUE}=== 9. GIT REPOSITORY STATUS ===${NC}\n"

if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    pass "Git repository detected"
    
    # Check for uncommitted changes
    if [ -z "$(git status --short)" ]; then
        pass "No uncommitted changes"
    else
        warn "Uncommitted changes detected"
        git status --short | sed 's/^/  /'
    fi
else
    fail "Not a git repository"
fi

echo ""
echo -e "${BLUE}=== 10. SUMMARY ===${NC}\n"

TOTAL=$((CHECKS_PASSED + CHECKS_FAILED))
echo "Checks Passed: ${GREEN}${CHECKS_PASSED}${NC}/${TOTAL}"

if [ $CHECKS_FAILED -eq 0 ]; then
    echo -e "${GREEN}All workflow configurations are valid!${NC}"
    echo ""
    echo -e "${BLUE}📋 EXECUTION SCHEDULE${NC}"
    echo "  • linkedin-combined-posts.yml: 3x daily (9 AM, 12 PM, 6 PM UTC)"
    echo "  • linkedin_growth_automation.yml: 3x daily (3:30 AM, 6:30 AM, 11:30 AM UTC)"
    echo ""
    echo -e "${BLUE}📝 NEXT STEPS${NC}"
    echo "  1. Confirm GitHub secrets are set:"
    echo "     - LINKEDIN_ACCESS_TOKEN"
    echo "     - LINKEDIN_MEMBER_ID"
    echo "  2. Monitor GitHub Actions tab for scheduled runs"
    echo "  3. Check workflow run logs for success/failure"
    echo "  4. Verify posts appear on LinkedIn feed"
    exit 0
else
    echo -e "${RED}Checks Failed: ${CHECKS_FAILED}${NC}/${TOTAL}"
    echo "⚠️  Some configurations need attention"
    exit 1
fi
