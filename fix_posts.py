#!/usr/bin/env python3
"""
Script to fix LinkedIn post engagement issues:
1. Improve duplicate detection
2. Reduce link frequency
3. Better engagement hooks
"""

import re
import json

def fix_footer_questions():
    """Update footer questions to be more engaging."""
    with open('post_data_analyst_news.py', 'r') as f:
        content = f.read()
    
    # Replace quick_tip footer questions
    old_quick_tip_footer = '''    footer_questions = [
        "What's your favorite quick tip?",
        "How do you apply this in your stack?",
        "What would you add to this list?",
        "Which tip saves you the most time?"
    ]
    hook = random.choice([h for h in quick_headers if h not in _USED_INTRO_LINES] or quick_headers)
    _USED_INTRO_LINES.append(hook)
    if len(_USED_INTRO_LINES) > len(quick_headers) // 2:
        _USED_INTRO_LINES = _USED_INTRO_LINES[-len(quick_headers)//2:]
    persona_line = random.choice([p for p in persona_lines if p not in _USED_SUBHEADER_LINES] or persona_lines)
    _USED_SUBHEADER_LINES.append(persona_line)
    if len(_USED_SUBHEADER_LINES) > len(persona_lines) // 2:
        _USED_SUBHEADER_LINES = _USED_SUBHEADER_LINES[-len(persona_lines)//2:]
    footer_question = random.choice([q for q in footer_questions if q not in _USED_FOOTER_QUESTIONS] or footer_questions)
    _USED_FOOTER_QUESTIONS.append(footer_question)
    if len(_USED_FOOTER_QUESTIONS) > len(footer_questions) // 2:
        _USED_FOOTER_QUESTIONS = _USED_FOOTER_QUESTIONS[-len(footer_questions)//2:]
    cta = pick_random_cta(FORMAT_CTAS["quick_tip"]) 
    tip = random.choice(QUICK_TIPS)
    emoji = get_emoji("hook")
    tip_emoji = "💡" if EMOJI_STYLE != "none" else ""
    lines = [hook, persona_line, ""]
    lines.extend([
        f"{tip_emoji} {tip}".strip(),
        "",
        "---",
        "",
        cta,
        "",
        get_subscription_cta(),
        "",
        get_hashtags(),
        "",
        f"❓ {footer_question}"
    ])'''
    
    new_quick_tip_footer = '''    footer_questions = [
        "What's your version of this tip?",
        "How have you used this?",
        "What did I miss?",
        "Your move - how would you apply this?",
        "Which hack saves you the most time?",
        "Tell me your best productivity trick below 👇"
    ]
    hook = random.choice([h for h in quick_headers if h not in _USED_INTRO_LINES] or quick_headers)
    _USED_INTRO_LINES.append(hook)
    if len(_USED_INTRO_LINES) > len(quick_headers) // 2:
        _USED_INTRO_LINES = _USED_INTRO_LINES[-len(quick_headers)//2:]
    persona_line = random.choice([p for p in persona_lines if p not in _USED_SUBHEADER_LINES] or persona_lines)
    _USED_SUBHEADER_LINES.append(persona_line)
    if len(_USED_SUBHEADER_LINES) > len(persona_lines) // 2:
        _USED_SUBHEADER_LINES = _USED_SUBHEADER_LINES[-len(persona_lines)//2:]
    footer_question = random.choice([q for q in footer_questions if q not in _USED_FOOTER_QUESTIONS] or footer_questions)
    _USED_FOOTER_QUESTIONS.append(footer_question)
    if len(_USED_FOOTER_QUESTIONS) > len(footer_questions) // 2:
        _USED_FOOTER_QUESTIONS = _USED_FOOTER_QUESTIONS[-len(footer_questions)//2:]
    cta = pick_random_cta(FORMAT_CTAS["quick_tip"])
    # Improved engagement CTAs
    engaging_cta_options = [
        "This works. Try it today.",
        "Your team will thank you for this.",
        "Implementation takes 5 minutes.",
        "No excuses - start today.",
        "This is a game-changer."
    ]
    tip = random.choice(QUICK_TIPS)
    emoji = get_emoji("hook")
    tip_emoji = "💡" if EMOJI_STYLE != "none" else ""
    lines = [hook, persona_line, ""]
    lines.extend([
        f"{tip_emoji} {tip}".strip(),
        "",
        random.choice(engaging_cta_options),
        "",
        get_subscription_cta(),
        "",
        get_hashtags(),
        "",
        f"❓ {footer_question}"
    ])'''
    
    content = content.replace(old_quick_tip_footer, new_quick_tip_footer)
    
    # Reduce link frequency in various post types
    # Quote style: 40% chance instead of 30%
    old_quote_link = '''    if link and random.random() > 0.7:
        lines.extend(["", f"{link}"])'''
    
    new_quote_link = '''    # Only show link strategically (40% of posts)
    if link and random.random() > 0.6:
        lines.extend(["", f"🔗 Dive deeper: {link}"])'''
    
    content = content.replace(old_quote_link, new_quote_link)
    
    # News flash: reduce link frequency
    old_news_link = '''    if link:
        style = random.choice([f"Breaking: {link}", f"Full story: {link}", f"Details: {link}"])
        lines.extend(["", style])'''
    
    new_news_link = '''    # Add link only 35% of the time
    if link and random.random() > 0.65:
        style = random.choice([f"📰 Read more: {link}", f"🔍 Full story: {link}", f"📄 Learn more: {link}"])
        lines.extend(["", style])'''
    
    content = content.replace(old_news_link, new_news_link)
    
    # Thread style: reduce link frequency
    old_thread_link = '''    if link and random.random() > 0.6:
        lines.extend(["", f"🔗 {link}"])'''
    
    new_thread_link = '''    # Add link only 40% of the time
    if link and random.random() > 0.6:
        lines.extend(["", f"📖 Source: {link}"])'''
    
    # Use a more specific match to avoid confusion
    if '    if link and random.random() > 0.6:\n        lines.extend(["", f"🔗 {link}"])' in content:
        content = content.replace(
            '''    if link and random.random() > 0.6:
        lines.extend(["", f"🔗 {link}"])''',
            new_thread_link,
            1  # Replace only first occurrence
        )
    
    with open('post_data_analyst_news.py', 'w') as f:
        f.write(content)
    
    print("✅ Applied engagement improvements to post_data_analyst_news.py")

if __name__ == "__main__":
    fix_footer_questions()
