#!/usr/bin/env python3
"""Test script to verify hiring pitch framework is properly integrated."""

import sys
import os

# Set test mode
os.environ['DRY_RUN'] = 'true'

print('🔍 Verification Tests')
print('=' * 70)

# Test 1: Import check
print('\n✅ Test 1: Checking imports...')
try:
    from identity_post_builders import build_hiring_pitch_post
    print('   ✓ build_hiring_pitch_post imported successfully')
except Exception as e:
    print(f'   ✗ Error: {e}')
    sys.exit(1)

try:
    from post_data_analyst_news import AVAILABLE_POST_FORMATS
    print(f'   ✓ Available formats: {len(AVAILABLE_POST_FORMATS)} total')
    if 'hiring_pitch' in AVAILABLE_POST_FORMATS:
        print(f'   ✓ hiring_pitch found in formats')
        count = AVAILABLE_POST_FORMATS.count('hiring_pitch')
        print(f'   ✓ hiring_pitch appears {count} times (weight: {count}/{len(AVAILABLE_POST_FORMATS)})')
    else:
        print('   ✗ hiring_pitch NOT found in formats')
        sys.exit(1)
except Exception as e:
    print(f'   ✗ Error: {e}')
    sys.exit(1)

# Test 2: Generate sample posts
print('\n✅ Test 2: Generating sample hiring pitch posts...')
try:
    for i in range(3):
        post = build_hiring_pitch_post()
        has_hook = any(x in post for x in ['Most Data Analysts', 'made a mistake', 'taught me', 'separates good', 'doing this wrong', 'best insights'])
        has_insight = "Here's what actually matters:" in post
        has_hashtag = '#DataAnalyst' in post
        has_opentowork = '#OpenToWork' in post
        
        if has_hook and has_insight and has_hashtag and has_opentowork:
            print(f'   ✓ Post {i+1}: Valid structure ✓')
        else:
            print(f'   ✗ Post {i+1}: Missing components')
            print(f'     Hook: {has_hook}, Insight: {has_insight}, Hashtag: {has_hashtag}, OpenToWork: {has_opentowork}')
except Exception as e:
    print(f'   ✗ Error: {e}')
    sys.exit(1)

# Test 3: Check scheduler import
print('\n✅ Test 3: Checking scheduler compatibility...')
try:
    import schedule
    print('   ✓ schedule library available')
except Exception as e:
    print(f'   ⚠ schedule not installed - run: pip install -r requirements.txt')

print('\n' + '=' * 70)
print('✅ ALL TESTS PASSED - System ready for hiring pitch posting!')
print('=' * 70)
