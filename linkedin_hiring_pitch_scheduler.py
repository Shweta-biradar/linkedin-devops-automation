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
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

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
MIDDAY_TIME = os.environ.get("HIRING_PITCH_MIDDAY_TIME", "12:00")
TIMEZONE = os.environ.get("TIMEZONE", "UTC")

# Post history tracking file
POST_HISTORY_FILE = "hiring_pitch_history.json"
MAX_HISTORY = 30  # Track last 30 posts to prevent repetition

def load_post_history():
    """Load posting history from file."""
    if not Path(POST_HISTORY_FILE).exists():
        return []
    
    try:
        with open(POST_HISTORY_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"⚠️ Could not load post history: {e}")
        return []

def save_post_history(history):
    """Save posting history to file."""
    try:
        with open(POST_HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.error(f"❌ Error saving post history: {e}")

def is_post_duplicate(post_text):
    """Check if post was recently posted."""
    history = load_post_history()
    return post_text in history

def add_to_history(post_text):
    """Add post to history, maintaining MAX_HISTORY limit."""
    history = load_post_history()
    history.insert(0, post_text)  # Add to beginning
    history = history[:MAX_HISTORY]  # Keep only last N posts
    save_post_history(history)

def post_hiring_pitch():
    """Generate and post a hiring pitch to LinkedIn (no duplicates)."""
    max_attempts = 10  # Try up to 10 times to get a unique post
    post_text = None
    
    for attempt in range(max_attempts):
        post_text = build_hiring_pitch_post()
        
        if not is_post_duplicate(post_text):
            break  # Found unique post
        
        if attempt < max_attempts - 1:
            logger.info(f"⚠️ Generated post is duplicate (attempt {attempt + 1}/{max_attempts}), regenerating...")
    
    if is_post_duplicate(post_text):
        logger.warning("⚠️ Could not generate unique post (all attempts resulted in duplicates)")
        return False
    
    try:
        logger.info("🚀 Generating hiring pitch post...")
        
        logger.info(f"📝 Generated post ({len(post_text)} chars):")
        logger.info("-" * 60)
        logger.info(post_text)
        logger.info("-" * 60)
        
        if DRY_RUN:
            logger.info("✅ DRY_RUN: Post not sent to LinkedIn")
            add_to_history(post_text)  # Still track in history
            return True
        
        logger.info("📤 Posting to LinkedIn...")
        post_id = post_to_linkedin(post_text)
        
        if post_id:
            add_to_history(post_text)  # Add to history on success
            logger.info(f"✅ Successfully posted! Post ID: {post_id}")
            return True
        else:
            logger.error("❌ Failed to post to LinkedIn")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error posting hiring pitch: {e}")
        return False

def schedule_posts():
    """Setup the thrice-daily posting schedule."""
    logger.info(f"""
╔══════════════════════════════════════════════════════════╗
║  LinkedIn Hiring Pitch Scheduler - 3x Daily Posts       ║
╠══════════════════════════════════════════════════════════╣
║  Morning Post:  {MORNING_TIME} {TIMEZONE:20s}║
║  Midday Post:   {MIDDAY_TIME} {TIMEZONE:20s}║
║  Evening Post:  {EVENING_TIME} {TIMEZONE:20s}║
║  Status:        {'🟢 ACTIVE' if not DRY_RUN else '🟡 DRY_RUN':<24}║
║  Dedup History: {MAX_HISTORY} posts (no repeats)         ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # Schedule the jobs
    schedule.every().day.at(MORNING_TIME).do(post_hiring_pitch)
    schedule.every().day.at(MIDDAY_TIME).do(post_hiring_pitch)
    schedule.every().day.at(EVENING_TIME).do(post_hiring_pitch)
    
    logger.info("✅ Scheduler initialized with deduplication")
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
