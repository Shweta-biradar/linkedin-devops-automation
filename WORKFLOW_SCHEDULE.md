# LinkedIn Automation Workflow Schedule

## Overview
Two independent workflows handle different aspects of LinkedIn automation:

### 1. **linkedin-combined-posts.yml** (Primary Posting)
**Focus:** Hiring pitches + Referral requests (3x daily)

| Schedule (Cron) | UTC Time | IST Time | Content |
|---|---|---|---|
| `0 9 * * *` | 9:00 AM UTC | 2:30 PM IST | Hiring Pitch + Referral Request |
| `0 12 * * *` | 12:00 PM UTC | 5:30 PM IST | Hiring Pitch + Referral Request |
| `0 18 * * *` | 6:00 PM UTC | 11:30 PM IST | Hiring Pitch + Referral Request |

**Per Run:** 1 hiring_pitch + 1 referral_request (30s apart)

**Daily Total:** 6 posts (3 hiring + 3 referral)

**Trigger:** Automated CRON + Manual workflow_dispatch

---

### 2. **linkedin_growth_automation.yml** (Growth & Engagement)
**Focus:** Growth strategies, engagement, analytics posts

| Schedule (Cron) | UTC Time | IST Time | Mode |
|---|---|---|---|
| `30 3 * * *` | 03:30 UTC | 9:00 AM IST | Growth automation |
| `30 6 * * *` | 06:30 UTC | 12:00 PM IST | Growth automation |
| `30 11 * * *` | 11:30 UTC | 5:00 PM IST | Growth automation |

**Per Run:** Full automation, engagement, analytics posts

**Trigger:** Automated CRON + Manual workflow_dispatch with customizable inputs

---

## Time Zone Reference
- **UTC:** Coordinated Universal Time
- **IST:** Indian Standard Time (UTC+5:30)

---

## Current Configuration Status

✅ **linkedin-combined-posts.yml**
- [x] 3 CRON schedules active
- [x] workflow_dispatch enabled
- [x] FORCE_FORMAT set (hiring_pitch, referral_request)
- [x] LINKEDIN_ACCESS_TOKEN configured
- [x] LINKEDIN_MEMBER_ID configured
- [x] DRY_RUN: false (live posting)

✅ **linkedin_growth_automation.yml**
- [x] 3 CRON schedules active
- [x] workflow_dispatch enabled with inputs
- [x] Rate limit bypass toggleable
- [x] Multiple engagement modes available

---

## Verification Steps

To verify workflows are triggered:
1. Check GitHub Actions tab → Workflows section
2. Look for recent "LinkedIn Combined Posts" and "LinkedIn Growth Automation" runs
3. Verify run timestamps match configured CRON times
4. Check workflow run logs for success/failure indicators

---

## Manual Trigger

From GitHub Repository:
1. Go to **Actions** tab
2. Select workflow: "LinkedIn Combined Posts" or "LinkedIn Growth Automation"  
3. Click **Run workflow**
4. Configure inputs (for growth automation)
5. Click **Run workflow**

Or via GitHub CLI:
```bash
gh workflow run linkedin-combined-posts.yml
gh workflow run linkedin_growth_automation.yml
```

---

## Next Steps (If Workflows Not Triggering)

1. **Verify Secrets:**
   - LINKEDIN_ACCESS_TOKEN (must be valid)
   - LINKEDIN_MEMBER_ID (numeric ID)
   - Action logs should show secrets masked

2. **Check Workflow Syntax:**
   ```bash
   cd .github/workflows
   yamllint linkedin-combined-posts.yml
   ```

3. **Monitor GitHub Actions:**
   - Repository → Actions → All Workflows
   - Check "Scheduled runs" section
   - Look for any disabled workflows (yellow warning)

4. **Test with Manual Trigger:**
   - Use workflow_dispatch to test immediately
   - Verify success/failure in logs
   - Check LinkedIn for actual posts

---

## Debugging

If runs appear but posts don't show on LinkedIn:
- Check workflow run logs for:
  - `Rate limit: Too soon since last post`
  - `LINKEDIN_ACCESS_TOKEN not set` 
  - `Failed to resolve author URN`
  - `DRY_RUN enabled: skipping actual LinkedIn post`

---

Last Updated: March 26, 2026
