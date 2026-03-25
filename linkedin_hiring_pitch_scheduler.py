#!/usr/bin/env python3
"""
LinkedIn Hiring Pitch Post Scheduler
=====================================

Automatically posts hiring pitch posts twice daily to maximize HR visibility.

Schedule:
- Morning Post: 9:00 AM UTC (peak hiring hours)
- Evening Post: 6:00 PM UTC (evening scroll time)

Adjust times via environment variables:
- HIRING_PITCH_MORNING_TIME (default: 09:00)
- HIRING_PITCH_EVENING_TIME (default: 18:00)
"""

import schedule
import time
import os
import sys
import logging
from datetime import datetime, timezone

# Add current directory to path
sys.path.insert(0, '.')

try:
    from post_data_analyst_news import post_to_linkedin, DRY_RUN
    from identity_post_builders import build_hiring_pitch_post
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
MORNING_TIME = os.environ.get("HIRING_PITCH_MORNING_TIME", "09:00")
EVENING_TIME = os.environ.get("HIRING_PITCH_EVENING_TIME", "18:00")
TIMEZONE = os.environ.get("TIMEZONE", "UTC")

def post_hiring_pitch():
    """Generate and post a hiring pitch to LinkedIn."""
    try:
        logger.info("🚀 Generating hiring pitch post...")
        post_text = build_hiring_pitch_post()
        
        logger.info(f"📝 Generated post ({len(post_text)} chars):")
        logger.info("-" * 60)
        logger.info(post_text)
        logger.info("-" * 60)
        
        if DRY_RUN:
            logger.info("✅ DRY_RUN: Post not sent to LinkedIn")
            return True
        
        logger.info("📤 Posting to LinkedIn...")
        post_id = post_to_linkedin(post_text)
        
        if post_id:
            logger.info(f"✅ Successfully posted! Post ID: {post_id}")
            return True
        else:
            logger.error("❌ Failed to post to LinkedIn")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error posting hiring pitch: {e}")
        return False

def schedule_posts():
    """Setup the twice-daily posting schedule."""
    logger.info(f"""
╔════════════════════════════════════════════════════════╗
║  LinkedIn Hiring Pitch Scheduler - 2x Daily Posts      ║
╠════════════════════════════════════════════════════════╣
║  Morning Post:  {MORNING_TIME} {TIMEZONE:20s}║
║  Evening Post:  {EVENING_TIME} {TIMEZONE:20s}║
║  Status:        {'🟢 ACTIVE' if not DRY_RUN else '🟡 DRY_RUN':<25}║
╚════════════════════════════════════════════════════════╝
    """)
    
    # Schedule the jobs
    schedule.every().day.at(MORNING_TIME).do(post_hiring_pitch)
    schedule.every().day.at(EVENING_TIME).do(post_hiring_pitch)
    
    logger.info("✅ Scheduler initialized")
    logger.info("📅 Waiting for scheduled times...")
    
    # Keep scheduler running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    try:
        logger.info("🎯 Starting LinkedIn Hiring Pitch Scheduler...")
        schedule_posts()
    except KeyboardInterrupt:
        logger.info("\n✋ Scheduler stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)
