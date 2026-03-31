# QuotaDrift Setup Script (Windows PowerShell)

Write-Host "🗜️ Setting up QuotaDrift..." -ForegroundColor Cyan

# Create venv
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install deps
python -m pip install --upgrade pip
pip install -r requirements.txt

# Handle .env
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "✓ Created .env from .env.example"
}

Write-Host "------------------------------------------------" -ForegroundColor White
Write-Host "✅ Setup complete." -ForegroundColor Green
Write-Host "1. Edit .env with your API keys."
Write-Host "2. Run the server: uvicorn main:app --reload"
Write-Host "------------------------------------------------"
