"""
Identity-Driven Post Builders
==============================

Functions to build each of the 20+ post types to help establish your unique identity.
Each function is designed to attract specific audience segments.
"""

import random
from extended_post_variety import POST_TEMPLATES, EXTENDED_POST_TYPES

def build_personal_story_post() -> str:
    """Build a personal story that builds emotional connection and identity."""
    template = POST_TEMPLATES["personal_story"]
    opener = random.choice(template["openers"])
    middle = random.choice(template["middles"])
    closer = random.choice(template["closers"])
    
    story_details = [
        "I spent weeks optimizing a query, only to realize a simple index would have solved it in minutes.",
        "My first ETL pipeline failed in production because I didn't account for data quality issues.",
        "I presented a dashboard to executives that nobody understood because I didn't listen to their needs.",
        "I thought raw SQL was all I needed. Then I learned Power BI DAX changed how I think about problems.",
        "The turning point came when I realized BI isn't about dashboards - it's about enabling decisions.",
        "I was drowning in ad-hoc reports until I built proper data governance and self-service BI.",
    ]
    
    lines = [
        f"📖 {opener}",
        "",
        f"My data analytics journey: {random.choice(story_details)}",
        "",
        f"{middle}",
        f"When I shifted from just building dashboards to understanding business impact, everything changed.",
        "",
        f"{closer}",
        "",
        "👇 What was your breakthrough moment in data/analytics?",
    ]
    
    return "\n".join(lines)


def build_bold_prediction_post() -> str:
    """Build a controversial prediction to spark discussion."""
    template = POST_TEMPLATES["bold_prediction"]
    opener = random.choice(template["openers"])
    support = random.choice(template["supporting_points"])
    closer = random.choice(template["closers"])
    
    predictions = [
        "Self-service BI will fail without proper data governance",
        "SQL is more valuable than any BI tool right now",
        "Most Power BI dashboards create noise, not insight",
        "Row-level security + incremental refresh will become non-negotiable",
        "Query optimization skills will become rarer and more valuable",
        "Data warehouse design is more important than the tool itself",
    ]
    
    evidence = [
        "- I'm seeing organizations struggle with governance first",
        "- Companies hiring for both SQL expertise and BI",
        "- Poor dashboard design directly impacts decision quality",
        "- Performance issues kill adoption faster than bad features",
        "- Most query problems come from poor indexing strategy",
        "- Tool choice matters less than schema design",
    ]
    
    lines = [
        f"🔮 {opener}",
        f"{random.choice(predictions)}.",
        "",
        f"{support}",
        random.choice(evidence),
        random.choice(evidence),
        "",
        f"❓ {closer}",
    ]
    
    return "\n".join(lines)


def build_contrarian_take_post() -> str:
    """Build a contrarian opinion to establish unique perspective."""
    template = POST_TEMPLATES["contrarian_take"]
    opener = random.choice(template["openers"])
    reasoning = random.choice(template["reasoning"])
    closer = random.choice(template["closers"])
    
    contrarian_takes = [
        ("More BI tools won't fix bad data modeling", "Design matters before tools"),
        ("Most dashboards are overengineered", "Simple usually wins"),
        ("CTEs aren't always the answer", "Indexing and execution plans matter more"),
        ("Power BI isn't the problem - execution is", "Great dashboards come from great requirements"),
        ("Self-service BI often fails because of governance", "Guardrails enable adoption"),
    ]
    
    take, reality = random.choice(contrarian_takes)
    
    supporting = [
        "I've seen teams with fancy tools fail due to poor schema design.",
        "The most performant queries I've built were simpler than expected.",
        "Bloated DAX measures kill adoption faster than missing features.",
        "Organizations trying self-service BI without governance hit walls fast.",
        "The best BI initiatives started with stakeholder conversations, not dashboards.",
    ]
    
    lines = [
        f"⚡ {opener}",
        f"'{take}' → {reality}",
        "",
        f"{reasoning}",
        random.choice(supporting),
        "",
        f"Your take on this? {closer}",
    ]
    
    return "\n".join(lines)


def build_values_statement_post() -> str:
    """Build a values statement to establish your principles."""
    template = POST_TEMPLATES["values_statement"]
    opener = random.choice(template["openers"])
    closer = random.choice(template["closers"])
    
    values_sets = [
        [
            "✅ Authenticity over perfection - I'll share failures, not just wins",
            "✅ People over process - Systems serve humans, not the other way around",
            "✅ Impact over visibility - Results matter more than likes",
            "✅ Learning over knowing - Curiosity beats confidence",
        ],
        [
            "✅ Progress over perfection",
            "✅ Experiments over theory",
            "✅ Collaboration over competition",
            "✅ Sustainability over hustle",
        ],
        [
            "✅ Radical honesty",
            "✅ Continuous improvement",
            "✅ Generosity with knowledge",
            "✅ High standards, high support",
        ],
    ]
    
    chosen_values = random.choice(values_sets)
    
    lines = [
        f"💎 {opener}",
        "",
    ]
    lines.extend(chosen_values)
    lines.extend([
        "",
        f"{closer}",
    ])
    
    return "\n".join(lines)


def build_failure_story_post() -> str:
    """Build a failure story to show vulnerability and learning."""
    template = POST_TEMPLATES["failure_story"]
    opener = random.choice(template["openers"])
    what_happened = random.choice(template["what_happened"])
    lessons = random.choice(template["lessons"])
    
    failures = [
        ("Not asking stakeholders what they actually needed", "Requirements beats assumptions"),
        ("Skipping data quality checks in ETL", "Validate early, debug late"),
        ("Over-optimizing queries without understanding the business context", "Performance without purpose is wasted effort"),
        ("Building complex DAX without documentation", "Simplicity and clarity win"),
        ("Ignoring index maintenance and query performance tuning", "Prevention beats firefighting"),
    ]
    
    situation, lesson = random.choice(failures)
    
    lines = [
        f"📉 {opener}",
        "",
        f"I was all-in on {situation}.",
        "",
        f"{what_happened}:",
        "- Presented dashboards that nobody used",
        "- Spent hours building features nobody asked for",
        "- Performance tanked in production",
        "- Had to rebuild from scratch",
        "",
        f"{lessons}:",
        f"**{lesson}**",
        "",
        "I apply this now by:",
        "- Starting with stakeholder workshops, not designs",
        "- Building data validation pipelines upfront",
        "- Focusing on query plans before optimization",
        "- Writing clear DAX with comments",
        "- Scheduling regular index maintenance",
        "",
        "👇 What data/analytics lesson came from your mistakes?",
    ]
    
    return "\n".join(lines)


def build_myth_busting_post() -> str:
    """Build a myth-busting post to establish expertise."""
    template = POST_TEMPLATES["myth_busting"]
    opener = random.choice(template["openers"])
    reality = random.choice(template["reality"])
    closer = random.choice(template["closers"])
    
    myths = [
        ("SQL expertise is becoming obsolete with modern BI tools", "SQL depth is more valuable now than ever"),
        ("More dashboards = Better insights", "Smart KPIs > Dashboard volume"),
        ("Star schema is outdated, snowflake schema is superior", "Schema choice depends on use case, not trends"),
        ("DAX can solve any Power BI performance issue", "Query folding and source optimization matter first"),
        ("Row-level security is optional in BI", "Security was table stakes, now it's structural"),
    ]
    
    myth, truth = random.choice(myths)
    
    lines = [
        f"🚫 MYTH: {myth}",
        "",
        f"✅ {reality}:",
        f"{truth}",
        "",
        "Here's what I've actually seen:",
        "- Best BI professionals combine deep SQL + BI tool expertise",
        "- Organizations with strong KPI frameworks outperform dashboard-heavy ones",
        "- Schema design is about understanding your query patterns",
        "- Most performance issues originate in the source system",
        "- Elite teams build governance, not just dashboards",
        "",
        f"{closer}",
    ]
    
    return "\n".join(lines)


def build_this_or_that_post() -> str:
    """Build an engaging this-or-that poll to drive engagement."""
    template = POST_TEMPLATES["this_or_that"]
    opener = random.choice(template["openers"])
    closer = random.choice(template["closers"])
    
    polls = [
        ("Raw talent with bad attitude", "Average talent with great culture fit"),
        ("Fast and messy", "Slow and clean"),
        ("Comfortable but stagnant", "Uncomfortable but growing"),
        ("Talk about it", "Do it"),
        ("Specialize deep", "Stay broad"),
    ]
    
    option1, option2 = random.choice(polls)
    
    lines = [
        f"🤔 {opener}",
        "",
        f"LEFT 👈  {option1}",
        f"RIGHT 👉  {option2}",
        "",
        "Your choice (and why?) 👇",
    ]
    
    return "\n".join(lines)


def build_question_thread_post() -> str:
    """Build a question thread to spark meaningful discussion."""
    template = POST_TEMPLATES["question_thread"]
    opener = random.choice(template["openers"])
    question = random.choice(template["questions"])
    follow_up = random.choice(template["follow_ups"])
    
    lines = [
        f"❓ {opener}",
        "",
        f"{question}",
        "",
        f"{follow_up}",
        "",
        "Drop your thoughts below 👇",
    ]
    
    return "\n".join(lines)


def build_framework_post() -> str:
    """Build a framework post to share your methodology."""
    frameworks = [
        {
            "name": "The Requirements-Schema-Optimization Framework",
            "steps": [
                "REQUIREMENTS: Ask stakeholders what they need to decide, not what reports they want",
                "SCHEMA: Design a clean, normalized dimensional model (Star Schema focus)",
                "OPTIMIZATION: Index strategically, optimize queries, measure performance"
            ],
            "why": "This cycle ensures every dashboard drives actual decisions"
        },
        {
            "name": "The ETL-Validate-Serve Framework",
            "steps": [
                "ETL: Extract from source, transform logic-heavy in warehouse, load gradually",
                "VALIDATE: Data quality checks at every stage, expectations defined upfront",
                "SERVE: Expose clean data through self-service with proper governance"
            ],
            "why": "Garbage in = Garbage out. Validation prevents downstream disasters"
        },
        {
            "name": "The SQL-DAX-Design Framework",
            "steps": [
                "SQL: Push heavy lifting to the source with complex CTEs and window functions",
                "DAX: Keep measures simple and semantic, focus on user experience",
                "DESIGN: Create dashboards that tell stories, not just display numbers"
            ],
            "why": "Performance on source saves DAX complexity and improves adoption"
        },
    ]
    
    framework = random.choice(frameworks)
    
    lines = [
        f"🎨 My framework for Data Analytics success:",
        f"**{framework['name']}**",
        "",
    ]
    
    for i, step in enumerate(framework['steps'], 1):
        lines.append(f"{i}️⃣  {step}")
    
    lines.extend([
        "",
        f"Why this works: {framework['why']}",
        "",
        "What framework works for you? Share below 👇",
    ])
    
    return "\n".join(lines)


def build_data_drift_post() -> str:
    """Build a detailed data drift post with practical context and hashtags."""
    lines = [
        "🔍 Data drift is the silent killer for analytics teams, and it usually starts with small, invisible changes.",
        "",
        "What is data drift?",
        "- The data distribution changes over time compared to training or expected historical patterns.",
        "- Examples include new input sources, schema modifications, or value shifts (e.g., free-text to numeric).",
        "",
        "Why it matters:",
        "1) Model drift: A scoring model built in Jan can fail by June if input features shift. (Example: new product category added without retraining)",
        "2) Analytics drift: KPI definitions change under the hood and dashboard numbers silently move, leading stakeholders to the wrong conclusion.",
        "3) Data source drift: A previously reliable API starts returning different values (format changes, null patterns), breaking ETL logic.",
        "",
        "Common counterargument:",
        "- 'Data drift is only an ML problem.' That's not true. Business intelligence pipelines, direct query dashboards, and ETL health checks suffer first.",
        "",
        "Personal experience:",
        "- I once had a report showing an 18% revenue spike weeks after a software rename propagated into the fact table. It wasn't real growth, it was poorly managed source drift.",
        "- Fix: Implement automated data quality checks, versioned schemas, and drift alerts in CI/CD for all data pipelines.",
        "",
        "Actionable checklist:",
        "- Set up daily distribution checks for key fields (mean, median, null %, unique values)",
        "- Add data contract monitoring (versioned schema + change agreement with source owners)",
        "- Review KPI definitions monthly with business, not just data engineering",
        "",
        "If you’re not tracking it, it will silently erode trust. ",
        "",
        "➡️ What’s your data drift story? Leave one practical tip below. 👇",
        "",
        "#ai #businessintelligence #datavisualization #python #rstats",
    ]
    return "\n".join(lines)


def build_market_observation_post() -> str:
    """Build a market observation post."""
    template = POST_TEMPLATES["market_observation"]
    opener = random.choice(template["openers"])
    observation = random.choice(template["observation"])
    implications = random.choice(template["implications"])
    
    observations = [
        ("Every company now wants self-service BI, but few invest in data governance", "We're seeing 40% dashboard abandonment rates because of it"),
        ("SQL skills are becoming rarer despite being more important than ever", "Companies are desperately seeking advanced SQL developers"),
        ("Data warehouse optimization is shifting from ETL to query performance", "The game is changing from data movement to query efficiency"),
        ("Power BI adoption is rapid, but stakeholder management is the real bottleneck", "Technical implementation rarely is the constraint anymore"),
        ("Most BI failures stem from poor schema design, not tool limitations", "Fundamentals matter more today than tool sophistication"),
    ]
    
    observation_text, context = random.choice(observations)
    
    lines = [
        f"👁️ {opener}",
        f"{observation_text}",
        "",
        f"Context: {context}",
        "",
        f"{observation}",
        "",
        f"{implications}",
        "",
        "What market shifts are you seeing in data/BI? 👇",
    ]
    
    return "\n".join(lines)



def build_before_after_post() -> str:
    """Build a before-and-after transformation post."""
    template = POST_TEMPLATES["before_after"]
    opener = random.choice(template["openers"])
    transformation = random.choice(template["transformation"])
    results = random.choice(template["results"])
    
    scenarios = [
        {
            "before": "Slow dashboards, ad-hoc queries running for 10 minutes, frustrated users",
            "change": "Implemented proper indexing strategy + optimized join logic + query plan analysis",
            "after": "Dashboard queries running in <5 seconds, users actually adopt self-service BI",
        },
        {
            "before": "ETL pipelines breaking every week, data quality issues, stakeholder trust eroding",
            "change": "Built data validation layer + automated quality checks + clear SLAs",
            "after": "Reliable pipelines, high confidence in data, faster reporting cycles",
        },
        {
            "before": "Hundreds of one-off dashboards, nobody reused anything, chaos",
            "change": "Built shared semantic layer + documented star schema + governance framework",
            "after": "Centralized BI platform, 60% efficiency gain, consistency across org",
        },
    ]
    
    scenario = random.choice(scenarios)
    
    lines = [
        f"✨ {opener}",
        "",
        f"BEFORE: {scenario['before']}",
        f"AFTER: {scenario['after']}",
        "",
        f"{transformation}",
        scenario['change'],
        "",
        "🎯 Key changes:",
        "- Started with diagnostic analysis, not assumptions",
        "- Built reusable components, then scaled",
        "- Focused on data quality as foundation",
        "- Involved stakeholders early in requirements",
        "",
        f"{results}",
        "",
        "What's your data/BI transformation story? 👇",
    ]
    
    return "\n".join(lines)


def build_career_journey_post() -> str:
    """Build a career journey reflection post."""
    career_stages = [
        {
            "then": "Started as SQL developer, writing queries, heads down on tickets",
            "pivot": "Realized impact multiplies when I understand the business problem first",
            "now": "Love being the bridge between business needs and technical solutions"
        },
        {
            "then": "Focused only on tool expertise (Power BI), chasing certifications",
            "pivot": "Learned that deep SQL + data modeling basics matter more than tool proficiency",
            "now": "Platform-agnostic, can pick the right tool because I understand the fundamentals"
        },
        {
            "then": "Built dashboards as requested without understanding real user needs",
            "pivot": "Started asking why before building what",
            "now": "Partner with stakeholders to drive actual decisions, not just dashboards"
        },
    ]
    
    stage = random.choice(career_stages)
    
    lines = [
        "🚀 My data analytics career journey:",
        "",
        f"📍 THEN: {stage['then']}",
        f"🔄 THE PIVOT: {stage['pivot']}",
        f"🎯 NOW: {stage['now']}",
        "",
        "These shifts from execution → strategy → impact changed everything.",
        "",
        "What pivots have shaped your BI/analytics journey? 👇",
    ]
    
    return "\n".join(lines)


def build_sql_tip_post() -> str:
    """Build a SQL tip/trick post."""
    sql_tips = [
        {
            "title": "Window Functions can replace complex joins",
            "snippet": "ROW_NUMBER() OVER (PARTITION BY id ORDER BY date DESC) replaces self-joins",
            "benefit": "Cleaner logic, better performance"
        },
        {
            "title": "CTEs make complex queries readable",
            "snippet": "WITH cleaned_data AS (...) SELECT * FROM cleaned_data WHERE condition",
            "benefit": "Easier to debug, modify, and optimize"
        },
        {
            "title": "Query execution plans reveal performance issues",
            "snippet": "Set Statistics IO ON; Run query; Check reads vs logical operations",
            "benefit": "Find expensive operations before they hit production"
        },
        {
            "title": "Proper indexing > Query optimization",
            "snippet": "Index on frequently filtered/joined columns, not every column",
            "benefit": "60%+ performance improvement without code changes"
        },
    ]
    
    tip = random.choice(sql_tips)
    
    lines = [
        f"💡 SQL TIP: {tip['title']}",
        "",
        f"Instead of writing complex subqueries, use CTEs for readability",
        f"'{tip['snippet']}'",
        "",
        f"Result: {tip['benefit']}",
        "",
        "🔍 Pro tip: Check execution plans before submitting PR 👇",
    ]
    
    return "\n".join(lines)


def build_power_bi_insight_post() -> str:
    """Build a Power BI insight/best practice post."""
    insights = [
        {
            "topic": "DAX Measures",
            "problem": "Complex DAX measures slow down your data model",
            "solution": "Push heavy lifting to SQL source, keep DAX simple",
            "win": "Faster dashboards + easier maintenance"
        },
        {
            "topic": "Row-Level Security",
            "problem": "Self-service BI without RLS is a security nightmare",
            "solution": "Implement RLS at table level + test with multiple roles",
            "win": "Users see only data they're supposed to see"
        },
        {
            "topic": "Incremental Refresh",
            "problem": "Full refresh of 100GB dataset every day = waste",
            "solution": "Set up incremental refresh with rolling windows",
            "win": "Load time drops from 1hr to 5min"
        },
    ]
    
    insight = random.choice(insights)
    
    lines = [
        f"🎨 Power BI Best Practice: {insight['topic']}",
        "",
        f"Problem: {insight['problem']}",
        f"Solution: {insight['solution']}",
        "",
        f"Result: {insight['win']}",
        "",
        "What Power BI challenges are you solving? 👇",
    ]
    
    return "\n".join(lines)


def build_data_modeling_lesson_post() -> str:
    """Build a data modeling lesson post."""
    lessons = [
        {
            "concept": "Star Schema vs Snowflake",
            "mistake": "Over-normalizing your data warehouse",
            "lesson": "Star schema: simple queries, fewer joins, denormalize strategically",
            "impact": "30% improvement in dashboard query speed"
        },
        {
            "concept": "Fact & Dimension Tables",
            "mistake": "Mixing transactional and dimensional logic",
            "lesson": "Clear separation: facts are measures, dimensions are context",
            "impact": "Cleaner data model, easier reporting"
        },
        {
            "concept": "Grain Definition",
            "mistake": "Ambiguous fact table granularity = incorrect aggregations",
            "lesson": "Define grain explicitly (daily? hourly? transaction level?)",
            "impact": "Accurate KPIs, trust in data"
        },
    ]
    
    lesson = random.choice(lessons)
    
    lines = [
        f"📊 Data Modeling: {lesson['concept']}",
        "",
        f"What I got wrong: {lesson['mistake']}",
        "",
        f"The lesson: {lesson['lesson']}",
        "",
        f"Impact: {lesson['impact']}",
        "",
        "Share your data modeling lessons 👇",
    ]
    
    return "\n".join(lines)


def build_etl_challenge_post() -> str:
    """Build an ETL challenge/solution post."""
    challenges = [
        {
            "issue": "Data quality issues discovered after dashboard goes live",
            "cause": "No validation in ETL pipeline",
            "fix": "Add validation layer: null checks, range checks, referential integrity",
        },
        {
            "issue": "ETL pipeline takes 4 hours, business needs fresh data by 6am",
            "cause": "Full reload every time, poor query optimization",
            "fix": "Incremental load logic + optimized source queries",
        },
        {
            "issue": "Dashboard showing incorrect aggregated values",
            "cause": "Missing deduplication logic in transformation",
            "fix": "Identify and handle duplicates before aggregation",
        },
    ]
    
    challenge = random.choice(challenges)
    
    lines = [
        f"🔧 ETL CHALLENGE SOLVED",
        "",
        f"Issue: {challenge['issue']}",
        f"Root cause: {challenge['cause']}",
        f"Solution: {challenge['fix']}",
        "",
        "Prevention > firefighting. Build validation upfront 🎯",
        "",
        "What ETL issues have you solved? 👇",
    ]
    
    return "\n".join(lines)


def build_kpi_strategy_post() -> str:
    """Build a KPI strategy/framework post."""
    lines = [
        "📈 KPI Strategy: From Dashboards to Decisions",
        "",
        "LEVEL 1 - Business KPIs (What matters to leadership)",
        "Revenue, Growth Rate, Customer Satisfaction, Retention",
        "",
        "LEVEL 2 - Process KPIs (What drives business metrics)",
        "Lead Quality, Conversion Rate, Cost per Acquisition, Customer Lifetime Value",
        "",
        "LEVEL 3 - Operational KPIs (What we directly control)",
        "Page Load Time, Data Freshness, Dashboard Adoption, Query Performance",
        "",
        "Most teams focus on Level 3 only. The breakthrough comes from connecting all 3 levels.",
        "",
        "If dashboard load time (L3) impacts adoption (L3) which impacts decision quality (L2+L1) - NOW it matters. 🎯",
        "",
        "What KPI strategy drives your org? 👇",
    ]
    
    return "\n".join(lines)


def build_query_optimization_post() -> str:
    """Build a query optimization post."""
    lines = [
        "⚡ Query Optimization: The 80/20 Rule",
        "",
        "80% of query problems come from:",
        "1. Missing indexes (most common)",
        "2. Unnecessary joins or subqueries",
        "3. Implicit conversions in WHERE clause",
        "4. Scanning large tables instead of seeking",
        "5. Not using execution plans",
        "",
        "BEFORE optimizing DAX or code, check your SQL query plan.",
        "",
        "My approach:",
        "✓ Identify expensive operations (clustered index scan = red flag)",
        "✓ Add indexes on frequently filtered columns",
        "✓ Rewrite joins to use seek instead of scan",
        "✓ Test with actual data volumes",
        "✓ Monitor with Query Store",
        "",
        "Result: 60% performance improvement, zero code changes. 🚀",
        "",
        "What query optimization trick has saved you the most time? 👇",
    ]
    
    return "\n".join(lines)


def build_stakeholder_management_post() -> str:
    """Build a stakeholder management/communication post."""
    lines = [
        "🤝 Stakeholder Management: The Real Skill in BI",
        "",
        "Building dashboards is the easy part.",
        "Managing expectations is the hard part.",
        "",
        "What actually works:",
        "❌ Showing up with 50 dashboard ideas",
        "✅ Asking: 'What decision do you need to make this week?'",
        "",
        "❌ Building everything they ask for",
        "✅ Saying: 'Let's start with the top 3 priorities'",
        "",
        "❌ Waiting until dashboard is perfect",
        "✅ Getting feedback at 60% complete",
        "",
        "❌ Technical presentations",
        "✅ Showing business impact, not technical details",
        "",
        "The teams who get BI adoption right? They prioritize communication over features.",
        "",
        "How do you manage stakeholder expectations? 👇",
    ]
    
    return "\n".join(lines)


def build_data_governance_post() -> str:
    """Build a data governance post."""
    lines = [
        "🔐 Data Governance: The Foundation Nobody Builds",
        "",
        "Self-service BI is broken without governance.",
        "I've seen it 100 times:",
        "",
        "Year 1: 'Let's empower users to build their own dashboards'",
        "Year 1.5: 'Why are there 500 conflicting KPI definitions?'",
        "Year 2: 'Everyone's confused, nobody trusts the data'",
        "Year 3: 'Let's go back to BI team building everything'",
        "",
        "Real governance looks like:",
        "✓ Clear data dictionary (what each field means)",
        "✓ Standardized KPI definitions (not 10 ways to count 'revenue')",
        "✓ Access control policies (who can see what)",
        "✓ Quality SLAs (when data should be fresh)",
        "✓ Change management (not random schema updates)",
        "",
        "Boring? Yes. Necessary? Absolutely.",
        "",
        "What governance frameworks have you found effective? 👇",
    ]
    
    return "\n".join(lines)


def build_tool_comparison_post() -> str:
    """Build a tool comparison/recommendation post."""
    lines = [
        "🛠️ Power BI vs Tableau vs Looker (My take after 2.5+ years)",
        "",
        "POWER BI:",
        "✅ Seamless Excel integration, great DAX, Microsoft ecosystem",
        "⚠️ Governance can be tricky at scale, query performance needs tuning",
        "",
        "TABLEAU:",
        "✅ Beautiful visualizations, intuitive for business users, mature platform",
        "⚠️ Higher licensing cost, performance depends on data source",
        "",
        "LOOKER:",
        "✅ Strong governance model, semantic layer built-in, SQL-first approach",
        "⚠️ Steeper learning curve, requires Looker expertise",
        "",
        "Reality: The tool doesn't matter as much as:",
        "→ Data quality",
        "→ Clear requirements",
        "→ Strong schema design",
        "→ User adoption strategy",
        "",
        "Any of these tools will fail without those fundamentals.",
        "",
        "Which tool are you using? What would you change? 👇",
    ]
    
    return "\n".join(lines)


def build_hiring_pitch_post() -> str:
    """
    Build a viral hiring pitch post following LinkedIn's proven engagement framework.
    
    Structure:
    1. HOOK - Stop the scroll, create curiosity (1-2 lines)
    2. PROBLEM/STORY - Relatable situation (2-4 lines)
    3. INSIGHT/VALUE - Core value with numbered points (3-5 points)
    4. POSITIONING - Skills & focus area (1-2 lines)
    5. CTA - Clear ask (1-2 lines)
    6. HASHTAGS - Relevant hashtags (5-8)
    
    Focus: Data Analyst recruitment positioning with #OpenToWork
    """
    
    # HOOK OPTIONS
    hooks = [
        "Most Data Analysts focus on tools.",
        "I made a mistake most analysts make.",
        "3 years in data taught me this…",
        "Here's what separates good analysts from great ones.",
        "Most teams are doing this wrong with their analytics.",
        "The best insights come from understanding the business.",
    ]
    
    # PROBLEM/STORY: Used to X, but then realized Y
    problems = [
        {
            "problem": "I used to think better dashboards = better impact",
            "realization": "But I was wrong",
            "setup": ""
        },
        {
            "problem": "I spent years building perfect dashboards",
            "realization": "Nobody was using them",
            "setup": ""
        },
        {
            "problem": "I optimized queries until they were perfect",
            "realization": "But the real problem was asking the wrong questions",
            "setup": ""
        },
        {
            "problem": "I thought SQL expertise was all I needed",
            "realization": "Communication turned out to be equally important",
            "setup": ""
        },
    ]
    
    # INSIGHTS/VALUE POINTS
    insights_pool = [
        {
            "title": "Asking the right questions",
            "details": "Understanding business needs comes before building"
        },
        {
            "title": "Understanding business needs",
            "details": "The context matters more than the tool"
        },
        {
            "title": "Communicating insights clearly",
            "details": "A dashboard nobody uses isn't a dashboard"
        },
        {
            "title": "Data quality foundations",
            "details": "Garbage in = garbage out, always"
        },
        {
            "title": "SQL expertise",
            "details": "Still the backbone of real analytics"
        },
        {
            "title": "Translating data to business decisions",
            "details": "That's where analysts add real value"
        },
    ]
    
    # POSITIONING OPTIONS
    positioning = [
        "With 2.5+ years of experience in SQL, Power BI & data warehousing\nI focus on solving real business problems through data",
        "With experience in ETL, data modeling & BI tool development\nI focus on enabling data-driven decisions",
        "Specialized in Power BI, SQL optimization & data governance\nI solve analytics problems that matter",
        "With expertise in query optimization, data modeling & reporting\nI build analytics solutions that drive action",
    ]
    
    # CTAs - Job seeking focused
    ctas = [
        "I'm open to Data Analyst roles 📊 (Immediate Joiner)\nWould appreciate referrals 🙏",
        "Currently looking for Data Analyst opportunities 🎯\nLet's connect if you have an opening",
        "Open to Data Analyst/BI Developer roles 🚀\nDrop a note if your team is hiring",
        "Actively seeking Data Analyst positions 💼\nConnected and ready to contribute immediately",
        "Looking for Data Analyst opportunities with growth potential 📈\nHappy to discuss roles in your organization",
    ]
    
    # Build the post
    hook = random.choice(hooks)
    problem_item = random.choice(problems)
    insights_selected = random.sample(insights_pool, k=random.randint(3, 5))
    position = random.choice(positioning)
    cta = random.choice(ctas)
    
    # Core hashtags
    hashtags = "#DataAnalyst #SQL #PowerBI #Analytics #OpenToWork #Hiring #DataJobs"
    
    # Construct the post with proper spacing
    lines = [
        hook,
        "",
        problem_item["problem"],
        problem_item["realization"],
        "",
        "Here's what actually matters:",
        "",
    ]
    
    # Add insights as numbered points
    for i, insight in enumerate(insights_selected, 1):
        lines.append(f"{i}. {insight['title']}")
    
    lines.extend([
        "",
        position,
        "",
        cta,
        "",
        hashtags,
    ])
    
    return "\n".join(lines)


def build_referral_request_post() -> str:
    """Build a post asking for referrals and job opportunities."""
    hooks = [
        "🚀 Actively seeking new opportunities in Data Analytics & BI",
        "📊 Looking to connect with teams doing impactful data work",
        "🔍 Open to referrals for Data Analyst/BI Developer roles",
        "💼 Seeking data-driven organizations to join",
        "🎯 Looking for my next challenge in analytics",
    ]
    
    value_props = [
        "3+ years building data pipelines, dashboards, and insights that drive business decisions",
        "Expert in SQL, Power BI, Python, and ETL processes with a track record of optimizing performance",
        "Passionate about turning complex data into actionable insights for stakeholders",
        "Experienced in data modeling, KPI development, and stakeholder management",
        "Skilled in modern BI tools and cloud data platforms (AWS, Azure, GCP)",
    ]
    
    asks = [
        "If you know of any opportunities or can make an introduction, I'd greatly appreciate it!",
        "Please reach out if you hear of relevant roles or want to discuss potential fits",
        "Open to connecting with hiring managers or teams in data/analytics",
        "Happy to chat about how my skills could contribute to your organization",
        "Let's connect if you're hiring or know someone who is",
    ]
    
    hashtags = [
        "#DataAnalyst #BusinessIntelligence #PowerBI #SQL #DataScience",
        "#Analytics #DataEngineering #BI #DataVisualization #Python",
        "#DataAnalytics #ETL #DataModeling #Tableau #Azure",
        "#BigData #DataWarehouse #KPI #DataGovernance #CareerOpportunity",
    ]
    
    lines = [
        random.choice(hooks),
        "",
        random.choice(value_props),
        "",
        random.choice(asks),
        "",
        "#OpenToWork",
        random.choice(hashtags),
    ]
    
    return "\n".join(lines)

# Export all builders
POST_BUILDERS = {
    "personal_story": build_personal_story_post,
    "bold_prediction": build_bold_prediction_post,
    "contrarian_take": build_contrarian_take_post,
    "values_statement": build_values_statement_post,
    "failure_story": build_failure_story_post,
    "myth_busting": build_myth_busting_post,
    "this_or_that": build_this_or_that_post,
    "question_thread": build_question_thread_post,
    "framework": build_framework_post,
    "market_observation": build_market_observation_post,
    "data_drift_insight": build_data_drift_post,
    "before_after": build_before_after_post,
    "career_journey": build_career_journey_post,
    "sql_tip": build_sql_tip_post,
    "power_bi_insight": build_power_bi_insight_post,
    "data_modeling_lesson": build_data_modeling_lesson_post,
    "etl_challenge": build_etl_challenge_post,
    "kpi_strategy": build_kpi_strategy_post,
    "query_optimization": build_query_optimization_post,
    "stakeholder_management": build_stakeholder_management_post,
    "data_governance": build_data_governance_post,
    "tool_comparison": build_tool_comparison_post,
    "hiring_pitch": build_hiring_pitch_post,
    "referral_request": build_referral_request_post,
}

print(f"✅ Post builders ready: {len(POST_BUILDERS)} data-focused formats loaded")
