import os
import pytest

# Ensure DRY_RUN mode during tests to prevent LinkedIn API authorization checks
os.environ["DRY_RUN"] = "true"

from post_data_analyst_news import (
    detect_topic_category,
    get_lesson_texts_for_category,
    get_lesson_value_for_category,
    get_contextual_hashtags,
    build_lessons_post,
)


def test_detect_topic_category_analytics():
    assert detect_topic_category("15 Power BI Project Ideas") == "analytics"
    assert detect_topic_category("Business Intelligence dashboard strategy") == "analytics"


def test_detect_topic_category_security_and_cloud():
    assert detect_topic_category("SQL Injection vulnerability report") == "security"
    assert detect_topic_category("AWS cost optimization patterns") == "cloud"


def test_get_lesson_texts_for_category_analytics():
    lessons = get_lesson_texts_for_category("analytics")
    assert any("data model clean" in l.lower() for l in lessons)
    assert any("business question" in l.lower() for l in lessons)


def test_get_lesson_value_for_category_analytics():
    value = get_lesson_value_for_category("analytics")
    assert "portfolio" in value


def test_get_contextual_hashtags_powerbi():
    tags = get_contextual_hashtags("15 Power BI Project Ideas", max_count=5)
    assert "#PowerBI" in tags or "#BusinessIntelligence" in tags


def test_build_lessons_post_powerbi_content():
    result = build_lessons_post([
        {"title": "15 Power BI Project Ideas to Build Your Portfolio in 2026", "summary": "Build real world Power BI solutions."}
    ])
    assert "Power BI" in result
    assert "data model" in result.lower() or "dashboard" in result.lower()
    assert "pattern" in result.lower()


def test_build_post_personal_story_uses_topic_hashtags():
    result = build_lessons_post([
        {"title": "Power BI adoption for analytics teams", "summary": "This is a summary."}
    ])
    assert "#PowerBI" in result or "#BusinessIntelligence" in result
