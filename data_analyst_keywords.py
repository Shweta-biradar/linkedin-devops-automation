"""
Data Analyst Brand Keywords and Configuration
==============================================

This file contains brand keywords and topics that will be used to ensure all generated posts
are related to the user's professional identity as a Data Analyst/BI Expert.

Professional Profile:
- Role: Results-driven Data Analyst with 2.5+ years experience
- Expertise: Data Analytics, Business Intelligence, Data Warehousing, SQL Development
- Tools: Power BI, DAX, SQL, Excel, Power Query
- Core Skills: Data modeling, ETL, Dashboard development, Query optimization, KPI reporting
- Experience: Agile/Scrum, UAT testing, Stakeholder management, 40% reporting efficiency gains
"""

# Core professional keywords - all posts should relate to these
CORE_KEYWORDS = {
    "Data Analytics",
    "Business Intelligence (BI)",
    "Power BI",
    "SQL",
    "Data Warehousing",
    "Data Engineering",
    "ETL",
    "Data Modeling",
    "Dashboard Development",
    "Query Optimization",
    "KPI Reporting",
    "DAX",
    "Data Visualization",
    "Analytics",
    "Reporting",
}

# Specific tools and technologies
TOOLS = {
    "Power BI",
    "SQL Server",
    "Python",
    "Excel",
    "Power Query",
    "DAX",
    "Tableau",
    "Looker",
    "dbt",
    "Snowflake",
    "BigQuery",
    "AWS Redshift",
    "Azure Data Lake",
    "SSIS",
}

# Data modeling concepts
DATA_MODELING_CONCEPTS = {
    "Star Schema",
    "Snowflake Schema",
    "Fact Tables",
    "Dimension Tables",
    "Data Warehouse",
    "Schema Design",
    "Grain Definition",
    "Slowly Changing Dimensions (SCD)",
    "Normalization",
    "Denormalization",
}

# SQL concepts
SQL_CONCEPTS = {
    "CTEs (Common Table Expressions)",
    "Window Functions",
    "Stored Procedures",
    "Joins",
    "Subqueries",
    "Indexing",
    "Query Performance",
    "Query Optimization",
    "Execution Plan",
    "Triggers",
    "Views",
    "Aggregations",
}

# Power BI concepts
POWER_BI_CONCEPTS = {
    "DAX Measures",
    "Calculated Columns",
    "Row-Level Security (RLS)",
    "Power Query",
    "Data Modeling in Power BI",
    "Incremental Refresh",
    "DirectQuery",
    "Import Mode",
    "Dashboard Design",
    "Report Layout",
    "Slicers and Filters",
    "Power BI Service",
    "Power BI Desktop",
}

# ETL/Data Integration concepts
ETL_CONCEPTS = {
    "ETL Pipeline",
    "Data Integration",
    "Data Transformation",
    "Data Validation",
    "Data Quality",
    "Data Cleansing",
    "Incremental Load",
    "Full Load",
    "Change Data Capture (CDC)",
    "Data Synchronization",
}

# Business/Soft skills
BUSINESS_SKILLS = {
    "Stakeholder Management",
    "Requirements Gathering",
    "UAT Testing",
    "KPI Definition",
    "Business Analytics",
    "Decision Support",
    "Agile/Scrum",
    "Data Governance",
    "Data Security",
    "Compliance",
    "Communication",
    "Documentation",
}

# Performance metrics and achievements
ACHIEVEMENTS = {
    "40% efficiency improvement",
    "60% performance gain",
    "Query optimization",
    "Reporting automation",
    "Dashboard adoption",
    "Data quality improvement",
    "Performance tuning",
    "Cost reduction",
}

# Post types that align with this profile
RECOMMENDED_POST_TYPES = [
    "sql_tip",
    "power_bi_insight",
    "data_modeling_lesson",
    "etl_challenge",
    "kpi_strategy",
    "query_optimization",
    "stakeholder_management",
    "data_governance",
    "tool_comparison",
    "myth_busting",
    "contrarian_take",
    "framework",
    "bold_prediction",
    "market_observation",
]

# Post types to use less frequently (general purpose)
SECONDARY_POST_TYPES = [
    "personal_story",
    "failure_story",
    "career_journey",
    "values_statement",
    "before_after",
]

# Topic pool - posts should be generated around these topics
TOPIC_POOL = {
    "Query Optimization": {
        "keywords": ["SQL", "performance", "execution plan", "indexing", "optimization"],
        "post_types": ["sql_tip", "query_optimization"],
    },
    "Power BI Best Practices": {
        "keywords": ["Power BI", "DAX", "dashboard", "performance", "security"],
        "post_types": ["power_bi_insight", "framework"],
    },
    "Data Modeling": {
        "keywords": ["schema", "star schema", "fact table", "dimension", "modeling"],
        "post_types": ["data_modeling_lesson", "myth_busting"],
    },
    "ETL & Data Integration": {
        "keywords": ["ETL", "pipeline", "data quality", "transformation", "integration"],
        "post_types": ["etl_challenge", "contrarian_take"],
    },
    "KPI & Analytics Strategy": {
        "keywords": ["KPI", "metrics", "analytics", "reporting", "decision"],
        "post_types": ["kpi_strategy", "framework"],
    },
    "Stakeholder Communication": {
        "keywords": ["stakeholder", "requirements", "communication", "adoption"],
        "post_types": ["stakeholder_management", "market_observation"],
    },
    "Industry Insights": {
        "keywords": ["BI", "analytics", "trends", "future", "industry"],
        "post_types": ["market_observation", "bold_prediction"],
    },
    "Data Governance": {
        "keywords": ["governance", "security", "compliance", "policy", "standards"],
        "post_types": ["data_governance", "framework"],
    },
    "Tool Comparisons": {
        "keywords": ["Power BI", "Tableau", "Looker", "tools", "comparison"],
        "post_types": ["tool_comparison", "contrarian_take"],
    },
}

def get_all_brand_keywords():
    """Get all brand-related keywords in a single set."""
    all_keywords = (
        CORE_KEYWORDS | TOOLS | DATA_MODELING_CONCEPTS |
        SQL_CONCEPTS | POWER_BI_CONCEPTS | ETL_CONCEPTS |
        BUSINESS_SKILLS | ACHIEVEMENTS
    )
    return all_keywords


def validate_post_is_data_focused(post_content: str) -> bool:
    """
    Check if a post is focused on data analyst keywords.
    Returns True if post mentions at least 2+ keywords from the brand lexicon.
    """
    keywords = get_all_brand_keywords()
    keyword_count = 0
    
    content_lower = post_content.lower()
    for keyword in keywords:
        if keyword.lower() in content_lower:
            keyword_count += 1
    
    return keyword_count >= 2


if __name__ == "__main__":
    print("Data Analyst Brand Keywords Configuration")
    print("=" * 60)
    print(f"Total keywords: {len(get_all_brand_keywords())}")
    print(f"Recommended post types: {len(RECOMMENDED_POST_TYPES)}")
    print(f"Topics: {len(TOPIC_POOL)}")
    print("\nCore Keywords:")
    for keyword in sorted(CORE_KEYWORDS):
        print(f"  • {keyword}")
