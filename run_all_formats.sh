#!/bin/bash
# Run LinkedIn posting with all formats enabled
# This allows random selection from all available post formats

cd /workspaces/linkedin-devops-automation

# Check if environment file exists
if [ ! -f .env ]; then
    echo "❌ .env file not found! Please create it with LINKEDIN_ACCESS_TOKEN"
    echo "Example:"
    echo "LINKEDIN_ACCESS_TOKEN=your_token_here"
    echo "LINKEDIN_MEMBER_ID=your_member_id"
    exit 1
fi

# Source the environment
set -a
source .env
set +a

echo "🚀 Running LinkedIn posting with ALL FORMATS ENABLED"
echo "📊 Available formats will be randomly selected"
echo "🤖 AI Enhancement: ENABLED"
echo "🎯 Dynamic Personas: ENABLED"
echo ""

# Run the posting script (no FORCE_FORMAT to enable all)
python3 post_data_analyst_news.py

echo ""
echo "✅ Posting completed (check output above for results)"