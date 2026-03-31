#!/bin/bash
# QuotaDrift Setup Script (Linux/macOS)

echo "🗜️ Setting up QuotaDrift..."

# Create venv
python3 -m venv venv
source venv/bin/activate

# Install deps
pip install --upgrade pip
pip install -r requirements.txt

# Handle .env
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✓ Created .env from .env.example"
fi

echo "------------------------------------------------"
echo "✅ Setup complete."
echo "1. Edit .env with your API keys."
echo "2. Run the server: uvicorn main:app --reload"
echo "------------------------------------------------"
