#!/usr/bin/env python3
"""
Quick Reference: Data Analytics RSS Feed Packs

This shows how to use the 14 specialized RSS feed packs with 100+ feeds.
Each pack focuses on specific data analytics topics aligned with your brand.
"""

# ============================================================================
# PACK NAMES & FEED COUNTS
# ============================================================================

PACKS = {
    "data-analyst": 30,           # Complete analytics stack ⭐ RECOMMENDED
    "powerbi-dax": 8,             # Power BI & DAX expertise
    "sql-databases": 12,          # SQL & databases
    "etl-data-pipeline": 11,      # ETL & data pipelines
    "data-modeling-schema": 6,    # Data modeling
    "business-intelligence": 9,   # BI platforms
    "cloud-data-platforms": 8,    # Cloud data services
    "analytics-engineering": 7,   # Modern data stack
    "data-governance-quality": 7, # Governance & quality
    "data-science-ml": 10,        # Data science/ML
    "performance-optimization": 7, # Query optimization
    "analytics": 11,              # General analytics
    "technology-news": 5,         # Tech trends
}

# ============================================================================
# QUICK START COMMANDS
# ============================================================================

EXAMPLES = {
    "General Data Analytics (RECOMMENDED)": {
        "command": 'export POST_SOURCE_PACK="data-analyst"',
        "description": "30 feeds covering all data analytics topics",
        "best_for": "Establishing overall data analyst authority"
    },
    
    "Power BI Expert": {
        "command": 'export POST_SOURCE_PACK="powerbi-dax,data-modeling-schema"',
        "description": "14 feeds focused on Power BI & data modeling",
        "best_for": "Power BI expertise and DAX mastery"
    },
    
    "SQL Specialist": {
        "command": 'export POST_SOURCE_PACK="sql-databases,performance-optimization,data-modeling-schema"',
        "description": "31 feeds about SQL, optimization, and schema design",
        "best_for": "SQL expertise and query performance authority"
    },
    
    "ETL/Data Pipeline": {
        "command": 'export POST_SOURCE_PACK="etl-data-pipeline,analytics-engineering,data-modeling-schema"',
        "description": "24 feeds on ETL, dbt, and data transformation",
        "best_for": "Data engineering and modern data stack"
    },
    
    "BI Platform Focus": {
        "command": 'export POST_SOURCE_PACK="business-intelligence,cloud-data-platforms"',
        "description": "17 feeds on BI tools and cloud platforms",
        "best_for": "Business intelligence tools expertise"
    },
    
    "Complete Coverage": {
        "command": 'export POST_SOURCE_PACK="data-analyst,powerbi-dax,sql-databases,performance-optimization,data-governance-quality"',
        "description": "50+ feeds covering all major data topics",
        "best_for": "Comprehensive data analyst brand"
    },
    
    "Cloud Focus": {
        "command": 'export POST_SOURCE_PACK="cloud-data-platforms,analytics-engineering,etl-data-pipeline"',
        "description": "26 feeds about cloud data platforms",
        "best_for": "Cloud warehouse expertise (Snowflake, BigQuery, etc)"
    },
}

# ============================================================================
# TOPIC-SPECIFIC FEEDS
# ============================================================================

TOPIC_MAPPING = {
    "💎 Power BI & DAX": {
        "packs": ["powerbi-dax"],
        "feeds": 8,
        "sources": "Microsoft Power BI official, SQLBI, DAX.tips, Guy in a Cube"
    },
    
    "🔧 SQL & Query Optimization": {
        "packs": ["sql-databases", "performance-optimization"],
        "feeds": 19,
        "sources": "SQL Shack, Bren Ozar, SQL Performance, MSSQL Tips"
    },
    
    "📊 Data Modeling & Schema": {
        "packs": ["data-modeling-schema"],
        "feeds": 6,
        "sources": "SQLBI, Kimball Group, DataEdo, dbt"
    },
    
    "🔄 ETL & Data Pipelines": {
        "packs": ["etl-data-pipeline"],
        "feeds": 11,
        "sources": "dbt, Meltano, Fivetran, Talend, Apache Airflow"
    },
    
    "☁️ Cloud Data Platforms": {
        "packs": ["cloud-data-platforms"],
        "feeds": 8,
        "sources": "Google Cloud, AWS, Azure, Snowflake, Databricks"
    },
    
    "📈 Business Intelligence": {
        "packs": ["business-intelligence"],
        "feeds": 9,
        "sources": "Tableau, Looker, Sisense, Qlik, MicroStrategy"
    },
    
    "⚙️ Analytics Engineering": {
        "packs": ["analytics-engineering"],
        "feeds": 7,
        "sources": "dbt, Meltano, Holistics, Modern Data Stack"
    },
    
    "🔐 Data Governance": {
        "packs": ["data-governance-quality"],
        "feeds": 7,
        "sources": "Immuta, Collibra, Alation, DataIQ"
    },
    
    "🚀 Data Science & ML": {
        "packs": ["data-science-ml"],
        "feeds": 10,
        "sources": "OpenAI, FastAI, Towards Data Science, Medium"
    },
    
    "📚 General Analytics": {
        "packs": ["analytics"],
        "feeds": 11,
        "sources": "KDnuggets, Analytics Vidhya, Mode, DataCamp"
    },
}

# ============================================================================
# RSS FEED STATISTICS
# ============================================================================

def print_summary():
    total_feeds = sum(PACKS.values())
    total_packs = len(PACKS)
    
    print("\n" + "="*80)
    print("📰 DATA ANALYTICS RSS FEED SYSTEM SUMMARY")
    print("="*80)
    print(f"\n✅ Total RSS Feed Packs: {total_packs}")
    print(f"✅ Total Available Feeds: {total_feeds}+")
    print(f"✅ Topics Covered: {len(TOPIC_MAPPING)}")
    print(f"✅ Post Formats Supported: sql_tip, power_bi_insight, query_optimization,")
    print(f"   data_modeling_lesson, kpi_strategy, etl_challenge, data_governance,")
    print(f"   tool_comparison, and more!")
    
    print("\n" + "-"*80)
    print("AVAILABLE PACKS:")
    print("-"*80)
    for pack, count in sorted(PACKS.items(), key=lambda x: x[1], reverse=True):
        print(f"  • {pack:40s} ({count:2d} feeds)")
    
    print("\n" + "-"*80)
    print("RECOMMENDED CONFIGURATIONS:")
    print("-"*80)
    for name, config in EXAMPLES.items():
        print(f"\n  {name}")
        print(f"  Command: {config['command']}")
        print(f"  Best for: {config['best_for']}")
    
    print("\n" + "-"*80)
    print("TOPIC-SPECIFIC RESOURCES:")
    print("-"*80)
    for topic, details in TOPIC_MAPPING.items():
        packs = ", ".join(details["packs"])
        print(f"\n  {topic}")
        print(f"  Packs: {packs}")
        print(f"  Feeds: {details['feeds']}")
        print(f"  Sources: {details['sources'][:60]}...")
    
    print("\n" + "="*80)
    print("🚀 NEXT STEPS:")
    print("="*80)
    print("""
1. Choose your configuration from above
2. Set environment variable:
   export POST_SOURCE_PACK="data-analyst"
   
3. Run your automation:
   python3 post_data_analyst_news.py

4. Monitor which posts get best engagement
5. Adjust pack selection to focus on high-engagement topics

For detailed information, see: RSS_FEED_CONFIGURATION.md
""")
    print("="*80 + "\n")

if __name__ == "__main__":
    print_summary()
