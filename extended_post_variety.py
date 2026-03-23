"""
Enhanced LinkedIn Post Variety System
=====================================

This module adds 20+ diverse post types to help establish your unique identity
and attract different audience segments with varied content.

Post Categories:
- PERSONAL BRAND (identity, values, story)
- THOUGHT LEADERSHIP (opinions, predictions, analysis)
- ENGAGEMENT (polls, questions, conversations)
- EDUCATION (tips, tutorials, explanations)
- STORYTELLING (narratives, case studies, journeys)
- SOCIAL PROOF (wins, achievements, testimonials)
- INDUSTRY INSIGHTS (trends, patterns, observations)
"""

from typing import Dict, List, Optional
import random

# =====================================================
# EXTENDED POST TYPES FOR IDENTITY ESTABLISHMENT
# =====================================================

EXTENDED_POST_TYPES = {
    # ===== PERSONAL BRAND (5 types) =====
    "personal_story": {
        "name": "Personal Story",
        "category": "PERSONAL BRAND",
        "description": "Share your journey, struggles, victories, lessons learned",
        "emoji": "📖",
        "engagement_level": "HIGH"
    },
    "values_statement": {
        "name": "Values & Beliefs",
        "category": "PERSONAL BRAND",
        "description": "What you stand for, your principles, your approach to work",
        "emoji": "💎",
        "engagement_level": "HIGH"
    },
    "about_me": {
        "name": "About Me",
        "category": "PERSONAL BRAND",
        "description": "Who you are, your background, what drives you",
        "emoji": "👋",
        "engagement_level": "MEDIUM"
    },
    "career_journey": {
        "name": "Career Journey",
        "category": "PERSONAL BRAND",
        "description": "How you got here, career milestones, pivotal moments",
        "emoji": "🚀",
        "engagement_level": "HIGH"
    },
    "mission_vision": {
        "name": "Mission & Vision",
        "category": "PERSONAL BRAND",
        "description": "Where you're headed, your big goals, your impact vision",
        "emoji": "🎯",
        "engagement_level": "HIGH"
    },
    
    # ===== THOUGHT LEADERSHIP (5 types) =====
    "bold_prediction": {
        "name": "Bold Prediction",
        "category": "THOUGHT LEADERSHIP",
        "description": "Forecast trends, make controversial predictions, challenge status quo",
        "emoji": "🔮",
        "engagement_level": "VERY HIGH"
    },
    "contrarian_take": {
        "name": "Contrarian Take",
        "category": "THOUGHT LEADERSHIP",
        "description": "Disagree with popular opinion, challenge conventional wisdom",
        "emoji": "⚡",
        "engagement_level": "VERY HIGH"
    },
    "framework": {
        "name": "Framework/Model",
        "category": "THOUGHT LEADERSHIP",
        "description": "Share your unique framework, model, or methodology",
        "emoji": "🎨",
        "engagement_level": "HIGH"
    },
    "research_findings": {
        "name": "Research Findings",
        "category": "THOUGHT LEADERSHIP",
        "description": "Share data, research, or findings that support your expertise",
        "emoji": "📊",
        "engagement_level": "HIGH"
    },
    "future_trends": {
        "name": "Future Trends",
        "category": "THOUGHT LEADERSHIP",
        "description": "What's coming in your industry, emerging patterns",
        "emoji": "🌍",
        "engagement_level": "HIGH"
    },
    
    # ===== ENGAGEMENT (5 types) =====
    "this_or_that": {
        "name": "This or That",
        "category": "ENGAGEMENT",
        "description": "Poll-style posts: choose between two options",
        "emoji": "🤔",
        "engagement_level": "VERY HIGH"
    },
    "question_thread": {
        "name": "Question Thread",
        "category": "ENGAGEMENT",
        "description": "Ask provocative questions to spark discussion",
        "emoji": "❓",
        "engagement_level": "VERY HIGH"
    },
    "fill_in_blank": {
        "name": "Fill in the Blank",
        "category": "ENGAGEMENT",
        "description": "Prompts audience to complete a sentence",
        "emoji": "📝",
        "engagement_level": "VERY HIGH"
    },
    "poll_post": {
        "name": "Poll Post",
        "category": "ENGAGEMENT",
        "description": "Multi-option polls to get audience opinion",
        "emoji": "📈",
        "engagement_level": "VERY HIGH"
    },
    "open_discussion": {
        "name": "Open Discussion",
        "category": "ENGAGEMENT",
        "description": "Start a conversation, share your thoughts first",
        "emoji": "💬",
        "engagement_level": "HIGH"
    },
    
    # ===== EDUCATION (5 types) =====
    "breakdown": {
        "name": "Breakdown",
        "category": "EDUCATION",
        "description": "Break down complex concepts into simple parts",
        "emoji": "🧩",
        "engagement_level": "MEDIUM"
    },
    "how_to": {
        "name": "How To",
        "category": "EDUCATION",
        "description": "Step-by-step tutorial or guide",
        "emoji": "📋",
        "engagement_level": "MEDIUM"
    },
    "myth_busting": {
        "name": "Myth Busting",
        "category": "EDUCATION",
        "description": "Debunk common misconceptions in your field",
        "emoji": "🚫",
        "engagement_level": "HIGH"
    },
    "lessons_learned": {
        "name": "Lessons Learned",
        "category": "EDUCATION",
        "description": "What you learned from mistakes or experiences",
        "emoji": "🎓",
        "engagement_level": "HIGH"
    },
    "mini_course": {
        "name": "Mini Course",
        "category": "EDUCATION",
        "description": "Share knowledge in a structured series",
        "emoji": "📚",
        "engagement_level": "MEDIUM"
    },
    
    # ===== STORYTELLING (5 types) =====
    "before_after": {
        "name": "Before & After",
        "category": "STORYTELLING",
        "description": "Show transformation, change, improvement",
        "emoji": "✨",
        "engagement_level": "VERY HIGH"
    },
    "client_win": {
        "name": "Client Win/Success Story",
        "category": "STORYTELLING",
        "description": "Share client success, victories, transformations",
        "emoji": "🏆",
        "engagement_level": "HIGH"
    },
    "behind_scenes": {
        "name": "Behind the Scenes",
        "category": "STORYTELLING",
        "description": "Show your process, workspace, team, or journey",
        "emoji": "🎬",
        "engagement_level": "HIGH"
    },
    "failure_story": {
        "name": "Failure Story",
        "category": "STORYTELLING",
        "description": "Share failures and what you learned",
        "emoji": "📉",
        "engagement_level": "VERY HIGH"
    },
    "journey_update": {
        "name": "Journey Update",
        "category": "STORYTELLING",
        "description": "Updates on your projects, progress, milestones",
        "emoji": "📍",
        "engagement_level": "MEDIUM"
    },
    
    # ===== SOCIAL PROOF (3 types) =====
    "milestone": {
        "name": "Milestone",
        "category": "SOCIAL PROOF",
        "description": "Celebrate achievements, numbers, impact",
        "emoji": "🎉",
        "engagement_level": "MEDIUM"
    },
    "testimonial": {
        "name": "Testimonial",
        "category": "SOCIAL PROOF",
        "description": "Share praise, reviews, kind words from others",
        "emoji": "⭐",
        "engagement_level": "LOW"
    },
    "top_post_reflection": {
        "name": "Top Post Reflection",
        "category": "SOCIAL PROOF",
        "description": "Reflect on why a post resonated, analyze engagement",
        "emoji": "📌",
        "engagement_level": "MEDIUM"
    },
    
    # ===== INDUSTRY INSIGHTS (3 types) =====
    "market_observation": {
        "name": "Market Observation",
        "category": "INDUSTRY INSIGHTS",
        "description": "What you're seeing in your industry",
        "emoji": "👁️",
        "engagement_level": "MEDIUM"
    },
    "pattern_recognition": {
        "name": "Pattern Recognition",
        "category": "INDUSTRY INSIGHTS",
        "description": "Patterns you've noticed across companies/teams",
        "emoji": "🔗",
        "engagement_level": "MEDIUM"
    },
    "industry_shift": {
        "name": "Industry Shift",
        "category": "INDUSTRY INSIGHTS",
        "description": "How your industry is changing, what's different",
        "emoji": "🌊",
        "engagement_level": "MEDIUM"
    }
}

# Post content templates for each type
POST_TEMPLATES = {
    "personal_story": {
        "openers": [
            "I wasn't always successful at this...",
            "A few years ago, I almost gave up on...",
            "The biggest turning point in my career was...",
            "Nobody knew this about me until...",
            "Here's a story I rarely share...",
        ],
        "middles": [
            "Everything changed when I realized...",
            "That's when I learned the hard way that...",
            "The breakthrough came from...",
            "What I didn't expect was...",
            "The lesson stuck with me...",
        ],
        "closers": [
            "Now I always...",
            "That experience taught me that...",
            "If I could go back, I'd tell myself...",
            "The takeaway for me was...",
            "Today, I apply this by...",
        ]
    },
    
    "bold_prediction": {
        "openers": [
            "Hot take: Within 3 years...",
            "Mark my words: The future of [industry] is...",
            "Unpopular opinion: I predict...",
            "I'm calling it now: The next big shift will be...",
            "Controversial prediction: [Assumption] will become...",
        ],
        "supporting_points": [
            "Here's why I think this: The data shows...",
            "Look at the patterns: We're already seeing...",
            "Three reasons this is inevitable:",
            "Evidence is mounting that...",
            "The signals are everywhere - here's what I see...",
        ],
        "closers": [
            "What's your take? Do you agree or disagree?",
            "Am I wrong? I'd love to hear your thoughts.",
            "Let me know if you think this is bold or obvious.",
            "Comment if you've already seen this trend.",
            "Who else sees this coming?",
        ]
    },
    
    "contrarian_take": {
        "openers": [
            "Everyone's saying X, but actually...",
            "The conventional wisdom is wrong about...",
            "Most people believe [myth], but what I've found is...",
            "This might be controversial, but...",
            "I disagree with the popular take that...",
        ],
        "reasoning": [
            "Here's why the mainstream opinion is incomplete:",
            "The real story is more nuanced:",
            "What people miss is...",
            "The evidence suggests something different:",
            "From my experience, the truth is...",
        ],
        "closers": [
            "Prove me wrong - what am I missing?",
            "Does this match your experience?",
            "Help me understand the other perspective.",
            "What would you add to this?",
            "Is this controversial or just obvious?",
        ]
    },
    
    "values_statement": {
        "openers": [
            "If you want to work with me, know this about me...",
            "These are the principles I stand by:",
            "I've always believed that...",
            "My non-negotiables are...",
            "In everything I do, I prioritize...",
        ],
        "values": [
            "Authenticity over perfection",
            "People over process",
            "Impact over visibility",
            "Learning over knowing",
            "Growth over comfort",
        ],
        "closers": [
            "Do these values align with yours?",
            "What values are non-negotiable for you?",
            "Let me know if these resonate with you.",
            "Share your core principle below 👇",
            "What would you add to this list?",
        ]
    },
    
    "myth_busting": {
        "openers": [
            "MYTH: [Common belief]",
            "This is often said: [Wrong statement]",
            "❌ WRONG: [Popular misconception]",
            "I used to believe [myth] too, but...",
            "The biggest misconception about [topic] is...",
        ],
        "reality": [
            "✅ REALITY: [Truth]",
            "What I've actually found is...",
            "The reality is more nuanced:",
            "Here's what the data actually shows:",
            "What really happens is...",
        ],
        "closers": [
            "Have you fallen for this myth too?",
            "What other myths should we bust?",
            "Did this change your perspective?",
            "Agree or disagree? Drop your take below 👇",
            "What myths exist in your field?",
        ]
    },
    
    "before_after": {
        "openers": [
            "BEFORE: I was stuck with...",
            "This changed everything for me/us/our team...",
            "The transformation happened when...",
            "What a difference when we...",
            "Then → Now: Here's what changed...",
        ],
        "transformation": [
            "The turning point was...",
            "We decided to...",
            "Everything shifted when we...",
            "The key change was...",
            "We implemented...",
        ],
        "results": [
            "AFTER: Now we...",
            "The impact has been...",
            "Today, the difference is...",
            "The results speak for themselves:",
            "We went from X to Y by...",
        ]
    },
    
    "this_or_that": {
        "openers": [
            "Quick poll - choose wisely 👇",
            "This has sparked debate - team's divided",
            "Which would you choose?",
            "Settle this debate for us:",
            "This or that? (and why?)",
        ],
        "options": [
            ["Speed", "Perfection"],
            ["Theory", "Practice"],
            ["Startup energy", "Stability"],
            ["Specialization", "Broad knowledge"],
            ["Remote work", "Office presence"],
        ],
        "closers": [
            "Drop your choice + reasoning below 👇",
            "Tell me which one and why!",
            "Comment: left or right?",
            "Your take? Which wins?",
            "What's driving your choice?",
        ]
    },
    
    "framework": {
        "openers": [
            "Here's a framework I use for...",
            "After years of doing this, I developed...",
            "The [Name] Model for...",
            "My approach to [topic] in 3 steps:",
            "Here's the system I use...",
        ],
        "framework_parts": [
            "Step 1: [Foundation/Understanding]",
            "Step 2: [Action/Development]",
            "Step 3: [Optimization/Results]",
        ],
        "closers": [
            "Have you tried this approach?",
            "What's your framework for this?",
            "Do you follow a similar process?",
            "What would you add to this?",
            "Share your version of this framework 👇",
        ]
    },
    
    "failure_story": {
        "openers": [
            "I failed spectacularly at...",
            "The biggest failure of my career...",
            "This didn't go as planned, and here's what happened...",
            "I made a massive mistake when...",
            "The failure that changed everything...",
        ],
        "what_happened": [
            "Here's what went wrong:",
            "The warning signs I missed...",
            "In hindsight, I should have...",
            "What I didn't see coming...",
            "The domino effect was...",
        ],
        "lessons": [
            "But here's what I learned:",
            "The silver lining was...",
            "This taught me that...",
            "Now I know to...",
            "The value of this failure was...",
        ]
    },
    
    "question_thread": {
        "openers": [
            "Question for you...",
            "I'm genuinely curious:",
            "Help me understand something:",
            "Can we talk about [topic]?",
            "Serious question for the fold:",
        ],
        "questions": [
            "What stops most people from...?",
            "Why do we still...?",
            "What would need to change for...?",
            "What's the real barrier to...?",
            "Why isn't more people doing...?",
        ],
        "follow_ups": [
            "Follow-up: What would it take?",
            "Real talk: Is it really that hard?",
            "Or am I overthinking this?",
            "What's your honest take?",
            "Help me think through this...",
        ]
    },
    
    "market_observation": {
        "openers": [
            "Observation: I'm seeing a lot of...",
            "Pattern in the market right now...",
            "What I'm noticing across companies...",
            "Trend alert: More teams are...",
            "I've noticed something interesting...",
        ],
        "observation": [
            "Here's what stands out:",
            "The common thread is...",
            "Every successful team I see is...",
            "The difference between winners and the rest...",
            "What's changed recently...",
        ],
        "implications": [
            "Why does this matter?",
            "My take on what this means:",
            "The implications are significant...",
            "Here's how I'd respond to this shift...",
            "If you see this trend, consider...",
        ]
    },
    
    "client_win": {
        "openers": [
            "So proud of [Client] for...",
            "Love seeing our clients succeed at...",
            "Shoutout to [Client] - they just...",
            "[Client] just hit an incredible milestone...",
            "One of my favorite success stories...",
        ],
        "story": [
            "Here's what made the difference:",
            "The key to their success was...",
            "They transformed by...",
            "What amazed me was their...",
            "The breakthrough for them was...",
        ],
        "takeaway": [
            "The lesson for others:",
            "If you're facing X, here's how they solved it...",
            "What this teaches us...",
            "The principle at play here...",
            "What everyone can learn from this...",
        ]
    },
}

# =====================================================
# DATA ANALYST EXPERTISE POST TYPES
# =====================================================

DATA_EXPERTISE_POST_TYPES = {
    # ===== DATA SKILLS & TOOLS (6 types) =====
    "sql_tip": {
        "name": "SQL Tip/Trick",
        "category": "DATA EXPERTISE",
        "description": "Share SQL optimization tricks, window functions, CTEs, indexing tips",
        "emoji": "💡",
        "engagement_level": "HIGH",
        "audience": ["practitioners", "peers"]
    },
    "power_bi_insight": {
        "name": "Power BI Best Practice",
        "category": "DATA EXPERTISE",
        "description": "Share Power BI tips, DAX patterns, RLS, incremental refresh strategies",
        "emoji": "🎨",
        "engagement_level": "HIGH",
        "audience": ["practitioners", "peers"]
    },
    "data_modeling_lesson": {
        "name": "Data Modeling Lesson",
        "category": "DATA EXPERTISE",
        "description": "Teach about star schema, dimension tables, grain definition",
        "emoji": "📊",
        "engagement_level": "MEDIUM",
        "audience": ["practitioners", "executives"]
    },
    "etl_challenge": {
        "name": "ETL Challenge Solved",
        "category": "DATA EXPERTISE",
        "description": "Share how you solved data quality, pipeline, or transformation issues",
        "emoji": "🔧",
        "engagement_level": "HIGH",
        "audience": ["practitioners", "peers"]
    },
    "query_optimization": {
        "name": "Query Optimization",
        "category": "DATA EXPERTISE",
        "description": "Share query performance tuning, indexing strategies, execution plans",
        "emoji": "⚡",
        "engagement_level": "MEDIUM",
        "audience": ["practitioners", "peers"]
    },
    "tool_comparison": {
        "name": "Tool Comparison",
        "category": "DATA EXPERTISE",
        "description": "Compare Power BI vs Tableau, Looker, and other BI platforms",
        "emoji": "🛠️",
        "engagement_level": "HIGH",
        "audience": ["executives", "practitioners"]
    },
    
    # ===== STRATEGY & LEADERSHIP (4 types) =====
    "kpi_strategy": {
        "name": "KPI Strategy",
        "category": "DATA EXPERTISE",
        "description": "Share KPI frameworks, metrics definitions, business metrics hierarchy",
        "emoji": "📈",
        "engagement_level": "HIGH",
        "audience": ["executives", "practitioners"]
    },
    "stakeholder_management": {
        "name": "Stakeholder Management",
        "category": "DATA EXPERTISE",
        "description": "Share insights on managing expectations, requirements gathering, communication",
        "emoji": "🤝",
        "engagement_level": "HIGH",
        "audience": ["executives", "peers"]
    },
    "data_governance": {
        "name": "Data Governance",
        "category": "DATA EXPERTISE",
        "description": "Share data governance frameworks, security, compliance, policies",
        "emoji": "🔐",
        "engagement_level": "MEDIUM",
        "audience": ["executives", "practitioners"]
    },
}

# Merge both post type systems
EXTENDED_POST_TYPES.update(DATA_EXPERTISE_POST_TYPES)

# Audience segments by post type (updated)
AUDIENCE_SEGMENTS = {
    "executives": [
        "bold_prediction", "research_findings", "framework", "values_statement",
        "kpi_strategy", "data_governance", "tool_comparison"
    ],
    "practitioners": [
        "myth_busting", "how_to", "breakdown", "lessons_learned",
        "sql_tip", "power_bi_insight", "data_modeling_lesson", "etl_challenge",
        "query_optimization", "kpi_strategy", "data_governance"
    ],
    "peers": [
        "contrarian_take", "personal_story", "career_journey", "failure_story",
        "sql_tip", "power_bi_insight", "etl_challenge", "query_optimization",
        "stakeholder_management"
    ],
    "industry": [
        "market_observation", "pattern_recognition", "industry_shift", "future_trends",
        "tool_comparison", "data_governance"
    ],
    "community": [
        "this_or_that", "question_thread", "open_discussion", "fill_in_blank",
        "kpi_strategy"
    ],
    "all": []  # Posts that appeal to everyone
}

print(f"✅ Extended post types loaded: {len(EXTENDED_POST_TYPES)} types")
print(f"📊 Available categories: {len(set(t['category'] for t in EXTENDED_POST_TYPES.values()))}")
